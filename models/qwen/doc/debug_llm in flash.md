# DDR/NVMe 权重分层 —— MoE 专家按需加载

## Context

Qwen3.6-35B-A3B (36 GB Q8_K_XL) 全量 mmap 加载后峰值 DDR ≈ 36 GB。该模型 256 个专家每 token 仅激活 8 个，专家权重占模型 ~31 GB（绝大多数内存），但每次推理只用其中 ~0.8 GB。目标：将专家权重改为 NVMe 按需加载 + 用完释放，DDR 常驻降至 ~5 GB。

## 策略

**纯 CPU 推理，同步 O_DIRECT 加载专家 slice，固定窗口缓存。**

核心流程：
1. 模型加载时：DDR 权重正常加载；NVMe 权重分配 full-size tensor buffer（MAP_ANONYMOUS，无物理页），注册到 NVMe provider
2. 推理时：每层 MoE gate 算出 expert IDs → NVMe provider O_DIRECT 读取对应专家数据 → copy 到 tensor buffer 对应 offset → 计算
3. 缓存窗口：保留最近 N 层（默认 5）的活跃专家在 DDR，淘汰更早层级的专家（madvise DONTNEED）

物理内存占用：DDR 权重 (~5 GB) + staging buffer (64 MB) + 窗口内活跃专家 (5 层 × ~42 MB ≈ 210 MB) + 计算缓冲区 ≈ **6-7 GB**。


## NVMe Pattern 默认规则

MoE 模型默认将以下 tensor name 模式匹配为 NVMe tier：
```
blk\.\d+\.ffn_(gate_up|gate|up|down)_exps\.weight
```
即所有层的专家 FFN 权重（`ffn_gate_up_exps` / `ffn_gate_exps` + `ffn_up_exps` / `ffn_down_exps`）。

DDR 常驻：
- `tok_embd`, `output`, `output_norm`（顶层）
- 所有 attention/SSM 权重（`wq`, `wk`, `wv`, `wo`, `wqkv`, SSM 相关）
- `ffn_gate_inp`（路由权重，小且每 token 必用）
- 共享专家权重（`ffn_*_shexp`，每 token 必用）
- 所有 norm 权重

## 内存估算 (Qwen3.6-35B-A3B, Q8_K_XL, cache_window=5)

| 组件 | DDR 占用 |
|------|---------|
| DDR 权重（attention + SSM + norms + gate + shared experts） | ~5 GB |
| NVMe tensor buffer 虚拟地址空间 | ~31 GB (无物理页) |
| Staging buffer | 64 MB |
| 缓存窗口内专家物理页 (5 层 × 8 experts × 3 tensors × ~1.7 MB) | ~210 MB |
| KV cache (n_ctx=4096, 10 层) | ~80 MB |
| 计算缓冲区 | ~0.5-2 GB |
| **总计峰值 DDR** | **~6-7.5 GB** |

对比全量 mmap 的 ~37 GB，节省 ~30 GB。窗口越大缓存命中率越高但内存越多，窗口=0 时最小内存 (~5.7 GB)。

## NVMe I/O 分析

### 单次搬运粒度 — 连续专家分组批量读取

`load_experts` 不是逐个专家单独读，而是将选中的 8 个专家 ID 排序后，把**连续的专家合并为一次 I/O**。

例如选中 `[3, 7, 15, 42, 43, 88, 101, 255]`，分组为：
- `[3]` → 1 expert × stride
- `[7]` → 1 expert × stride
- `[15]` → 1 expert × stride
- `[42, 43]` → **2 experts × stride**（一次连续读）
- `[88]` → 1 expert × stride
- `[101]` → 1 expert × stride
- `[255]` → 1 expert × stride

每个分组的 I/O 长度 = `n_consecutive × expert_stride + 512B padding`。对齐后直接 `pread` 到 tensor buffer 对应位置，大块时绕过 staging buffer 零拷贝。

Qwen3.6-35B-A3B（Q8_K_XL）每个 expert stride：

| Tensor | Shape | expert_stride (nb[2]) |
|--------|-------|-----------------------|
| `ffn_gate_up_exps` | [2048, 1024, 256] | ~2.2 MB |
| `ffn_down_exps` | [512, 2048, 256] | ~1.1 MB |

### 每层 I/O 量

- `ffn_gate_up_exps`：8 experts × ~2.2 MB = ~17.6 MB
- `ffn_down_exps`：8 experts × ~1.1 MB = ~8.8 MB
- **每层合计 ≈ ~26 MB**

### 全量推理 I/O（40 层，cache_window=0）

- 40 层 × ~26 MB = **~1.0 GB** NVMe 读取
- NVMe 3 GB/s 顺序读取：~330 ms（理论值，实际受 expert 分散布局影响）
- 首次推理时所有专家都需加载，后续推理可复用缓存窗口内的专家

### 数据路径

```
MoE gate 计算 → selected_experts [8, n_tokens] (I32)
     ↓ eval callback (ask=true, GGML_OP_MUL_MAT_ID 前)
load_experts: O_DIRECT pread → staging buffer → memcpy 到 tensor buffer 对应 slot
     ↓
ggml_mul_mat_id CPU 实现: 只访问 matrix_row_counts[cur_a] > 0 的专家
     (248 个未选中专家的内存不被触碰，不触发 page fault)
```

纯 CPU 路径下 scheduler 的 MoE expert copy 优化不触发（无跨设备拷贝），数据留在原 buffer，由 `ggml_mul_mat_id` 按索引直接访问。


## 实测数据（2026-05-14，初始实现）

> 注：此时存在多个 bug（per-layer expert 跟踪、view ne[0] 还原等），输出质量有问题。以下为历史参考。

环境：16 GB DDR，Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf（36 GB），CPU-only，`--nvme-tiers --nvme-cache-window 5 --no-mmap`

| 阶段 | DDR (VmRSS) | VmHWM | 说明 |
|------|-------------|-------|------|
| 加载完成 | 4.6 GB | **6.2 GB** | buffer 分配 36 GB 虚拟地址（VmSize=37.8 GB），仅 DDR 权重填充物理页 |
| 推理中峰值 | **~5.3 GB** | 6.2 GB | NVMe 按需加载专家，推理速度 ~545 ms/token |
| 推理后稳定 | **~4.5 GB** | 6.2 GB | cache_window=5 内专家驻留 |

Buffer 分配明细（server 日志）：
- CPU model buffer: 36,659 MiB（虚拟地址，大部分无物理页）
- CPU KV buffer: 80 MiB
- CPU RS buffer (SSM 状态): 251 MiB
- CPU compute buffer: 497 MiB
- NVMe staging: 64 MiB

**关键结论**：16 GB 系统成功运行 36 GB 模型，峰值 DDR 6.2 GB。全量 mmap 模式需 ~37 GB DDR，此系统无法运行。


## 实测数据（2026-05-18，修复后）

环境：Intel i7-14700KF + 62 GB DDR，Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf（36 GB），CPU-only，`--nvme-tiers`

| 指标 | 全量加载 (baseline) | NVMe 分层 |
|------|---------------------|-----------|
| VmHWM | ~37 GB | **~4.2 GB** |
| VmRSS (推理中) | ~37 GB | ~3.9 GB |
| 输出质量 | "The capital of France is **Paris**." | "The capital of France is **Paris**." ✅ |
| Prefill 速度 | 35.5 tok/s | 2.7 tok/s |
| Decode 速度 | 15.5 tok/s | 1.4 tok/s |
| 退出 | 正常 | 正常（double free 已修复） |

内存明细（server 日志）：
```
Host buffer:  71 GB total (虚拟地址), 67 MiB model + 248 MiB context = ~315 MiB 物理页
CPU_NVMe:     ~31 GB 虚拟地址空间 (MAP_NORESERVE, 无物理页)
KV cache:     ~80 MiB
Compute:      ~500 MiB
```

**关键结论**：DDR 峰值从 ~37 GB 降至 ~4.2 GB（9x 节省），输出正确性一致。解码速度受限于 NVMe I/O（每层 ~26 MB，64 层全量推理 ~1.7 GB 读取），约 11x 慢于全量 DDR。

## 并发 I/O 优化

### 问题

初始实现使用单个 staging buffer，所有 pread() 调用串行执行。每个 expert slice ~1-2 MB 的读取虽然足够发挥 NVMe 顺序带宽，但串行 issued 导致 SSD 在两次读取之间空闲，无法饱和 NVMe 队列深度。

### 方案：多 staging buffer + std::async 并发 pread

- 分配 `io_depth`（默认 4）个 staging buffer（每个 64 MB，posix_memalign 页对齐）
- `load_experts()` 将选中的 experts 排序、合并连续分组后，以 `io_depth` 为一批并发发出 pread()
- 每批最多 `io_depth` 个并发 I/O，确保每个 staging buffer 同一时间只被一个 pread() 使用
- 通过 `std::async(std::launch::async, ...)` 实现，无需 io_uring 等额外依赖

### 关键修复：staging buffer 竞争

初次实现中 `staging_bufs_[gi % n_bufs]` 在组数超过 io_depth 时，两个 async 任务会同时写入同一个 staging buffer，导致数据损坏（模型输出退化为重复文本）。

**修复**：改为分批处理，每批最多 `io_depth` 个组，批内 staging buffer 一一对应（`staging_bufs_[gi - batch_start]`），批间同步等待所有 future 完成。

```cpp
for (size_t batch_start = 0; batch_start < groups.size(); batch_start += n_bufs) {
    // batch_start 到 batch_end 的组并发执行
    // 每组使用 staging_bufs_[gi - batch_start]，无竞争
    for (auto & f : futures) f.get(); // 批间同步
}
```

### 新增 CLI 参数

`--nvme-io-depth N`（默认 4）：并发 I/O 队列深度，即 staging buffer 数量。

### 实测数据（2026-05-18，并发 I/O）

环境：同上，`--nvme-tiers --nvme-io-depth 4`

| 指标 | io-depth 1 (串行) | io-depth 4 (并发) | 全量加载 (baseline) |
|------|-------------------|-------------------|---------------------|
| Prefill 速度 | 1.4 tok/s | **3.5 tok/s** (2.5x) | 35.5 tok/s |
| Decode 速度 | 1.1 tok/s | **2.0 tok/s** (1.8x) | 15.5 tok/s |
| VmHWM | ~4.2 GB | ~4.2 GB | ~37 GB |
| 输出质量 | ✅ | ✅ | ✅ |
| Staging buffer | 64 MB × 1 | 64 MB × 4 = 256 MB | N/A |

并发 I/O 带来 ~1.8-2.5x 提速，DDR 占用仅增加 ~192 MB（多 3 个 staging buffer）。

## 性能瓶颈分析

相比全量 DDR 加载（15.5 tok/s decode），NVMe 分层后（2.0 tok/s）性能下降约 7.7x。瓶颈定位如下。

### 每个前向传播的 I/O 量

```
64 层 × cache_window=5 → 每次需加载 ~59 层专家
每层: 8 experts × 3 tensors (gate_up, gate, down) × ~1.1-2.2 MB/expert ≈ 26 MB
总计: 59 × 26 MB ≈ 1.3 GB / token
```

### NVMe 实测带宽

设备：NVMe SSD 931.5 GB，`/dev/nvme0n1p2`，ext4。

| 模式 | 带宽 | 测试方法 |
|------|------|---------|
| 顺序 1MB 块 | 3.5 GB/s | `dd iflag=direct bs=1M` |
| 顺序 2MB pread | 1.8 GB/s | 500× pread(fd, 2MB, sequential offset) |
| **随机 2MB pread** | **1.53 GB/s** | 500× pread(fd, 2MB, random offset)，模拟专家分散访问 |

专家分布在 36 GB 文件中的不同偏移，属于随机读模式。

### 瓶颈：NVMe 物理带宽是硬天花板

```
理论 decode 下限 = 1.3 GB ÷ 1.53 GB/s = 850 ms/token → 1.2 tok/s
实测 decode = 2.0 tok/s (500 ms/token)
```

实测值高于理论下限，因为：
- 专家分组后连续的 experts 合并为一次 pread，部分恢复顺序读
- io_depth=4 并发 I/O 饱和 NVMe 队列深度
- cache_window=5 避免了最近 5 层的重复加载

**结论：无论怎样优化软件调度，每 token 1.3 GB 的数据量受 NVMe 物理带宽限制，是根本瓶颈。CPU 计算被 I/O 完全饿死。**

### 可能的提速方向

| 方向 | 预期效果 | 代价 |
|------|---------|------|
| 专家量化降级（Q8→Q4，stride 减半） | 2-4x decode | 精度损失 |
| 增大 cache_window（更多层驻留 DDR） | 线性减少 I/O 量 | DDR 占用增加（每层 ~26 MB） |
| Prefetch 下一层专家 ID（利用前几层 attention 结果预测） | 隐藏 I/O 延迟 | 实现复杂，预测准确率不确定 |
| PCIe 5.0 NVMe（随机读 ~12 GB/s） | ~3-4x | 硬件升级 |
| RAMDisk / 内存盘（专家热数据缓存） | 接近 DDR 速度 | 需要额外 RAM |

核心矛盾：**MoE 每次只用 8/256 专家（3.1%），但仍需从 NVMe 加载 ~1.3 GB 数据/token**。NVMe 随机读带宽决定了性能天花板。
