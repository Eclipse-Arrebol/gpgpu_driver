# gpgpu-driver

一个基于 QEMU 虚拟 GPU 的自研 GPGPU 软件栈。项目从 Linux PCI 驱动、用户态运行时、VRAM 分配器、手写 RV32 GPU kernel，一直做到 Qwen2-0.5B 的 token-level greedy 推理 demo。

这个项目的目标不是做一个高性能推理框架，而是把 GPU 软件栈的关键路径拆开，从底层设备模型到上层 transformer 推理完整跑通。

# qemu
项目地址：本项目是 QEMU 训练营 2026 GPU赛题的基础上改的
https://github.com/gevico/qemu-camp-2026-exper-Eclipse-Arrebol


## 当前状态

当前已经跑通：

- QEMU 虚拟 PCI GPGPU 设备上的 kernel dispatch
- Linux PCI 字符设备驱动
- 用户态 `libgpgpu` runtime
- `mmap` 访问 VRAM
- `ioctl` 发起 kernel launch
- 中断通知 kernel 完成
- Buddy + slab VRAM 分配器
- 多个手写 RV32 GPU kernel
- Qwen2-0.5B FP32 权重加载
- KV cache
- 单 token decode transformer pipeline
- host tokenizer + VM token-level greedy generation + host detokenizer 的 toy 推理闭环

注意：README 里的“INT8 推理”是后续目标。当前已跑通的是 **FP32 展开权重路径**，不是 INT8 路径。

## 架构

```text
文本 / token ids
    ↓
tools/tokenize_qwen.py 或手写 token ids
    ↓
infer_tokens / libgpgpu
    ↓ mmap + ioctl
gpgpu_drv.ko
    ↓ MMIO + interrupt
QEMU virtual PCI GPGPU
    ↓
RV32 SIMT interpreter
    ↓
hand-written GPU kernels
```

核心调用路径：

```text
gpuInit()
    open("/dev/gpgpu")
    mmap VRAM

weight_loader_load()
    load Qwen2-0.5B weight binaries into VRAM

transformer_init()
    allocate scratch buffers
    load GPU kernel binaries
    build RoPE sin/cos table

transformer_step()
    embedding
    24 x decoder layer
    final RMSNorm
    lm_head
    argmax
```

## 设计笔记：从驱动到 transformer

这一节按实现顺序记录项目里几块核心设计。它更像开发笔记，不追求抽象得很漂亮，而是尽量解释“为什么这个 toy GPU 软件栈能跑起来”。

### `gpgpu_drv.ko`：一个最小可用的 PCI GPU 驱动

驱动入口是一个标准 Linux PCI driver。内核发现 vendor/device id 匹配的 PCI 设备后，会调用 `gpgpu_probe()`。这个函数做了几件事：

1. `pci_enable_device()` 启用 PCI 设备。
2. `pci_request_regions()` 认领 BAR 资源。
3. `pci_iomap()` 映射 BAR0/BAR2/BAR4。
4. 读取控制寄存器里的 device id 和 VRAM size。
5. 注册字符设备 `/dev/gpgpu`。
6. 注册中断，当前使用 Legacy INTx。
7. 打开设备侧的 interrupt enable。

这个驱动没有试图实现完整 DRM/GPU 子系统，而是选择了一个很直接的路径：把虚拟 GPU 暴露成一个字符设备。用户态 runtime 打开 `/dev/gpgpu`，通过 `mmap` 拿到 VRAM 映射，通过 `ioctl` 发起 kernel dispatch。

### `file_operations`：为什么没有走 `read/write`

当前驱动的 `file_operations` 只实现了：

```c
open
release
mmap
unlocked_ioctl
```

没有实现 `read` / `write`。这是有意简化：GPU 数据传输不走字符设备的字节流接口，而是直接把 BAR2，也就是 VRAM，映射到用户态地址空间。

因此用户态的 `gpuMemcpy()` 本质上只是：

```c
memcpy(ctx->vram_mmap + offset, host_ptr, size);  // H2D
memcpy(host_ptr, ctx->vram_mmap + offset, size);  // D2H
```

这种方式非常粗糙，但很适合 toy GPGPU：

- 不需要先实现 DMA engine。
- 不需要设计复杂的 copy command queue。
- 调试时可以直接 D2H dump 任意 VRAM 区域。
- 性能不理想，但软件栈路径很清楚。

`open()` 做的事情也很少：从 inode 里找到 `gpgpu_dev`，放进 `file->private_data`。之后 `mmap()` 和 `ioctl()` 都从这里拿设备上下文。

### `mmap`：把 BAR 映射给用户态

驱动的 `mmap` 根据 `vm_pgoff` 区分映射哪个 BAR：

```text
vm_pgoff == 0  -> BAR0 控制寄存器
vm_pgoff == 1  -> BAR2 VRAM
```

用户态 `gpuInit()` 里映射 VRAM 时用的是：

```c
mmap(..., fd, 1 * PAGE_SIZE)
```

也就是让 `vm_pgoff == 1`，从而映射 BAR2。之前踩过一个坑：如果 offset 用 0，就会映射到 BAR0 控制寄存器，用户态以为自己在写 VRAM，实际是在写控制寄存器窗口。

真正建立映射的是：

```c
io_remap_pfn_range()
```

它把 PCI BAR 的物理页框映射到用户进程的虚拟地址空间。这样 `libgpgpu` 就可以把 VRAM 当成一段普通内存来读写。

### `ioctl`：一次 kernel launch 的控制路径

`ioctl` 当前只定义了一个核心命令：

```c
GPGPU_IOC_DISPATCH
```

用户态传入：

```text
kernel_addr
kernel_args
grid_dim[3]
block_dim[3]
shared_mem_size
```

驱动收到后，把这些字段写进 BAR0 的 MMIO 控制寄存器：

```text
KERNEL_ADDR_LO / HI
KERNEL_ARGS_LO / HI
GRID_DIM_X/Y/Z
BLOCK_DIM_X/Y/Z
SHARED_MEM_SIZE
DISPATCH
```

最后写 `DISPATCH` 寄存器启动设备侧执行。然后驱动在 wait queue 上睡眠：

```c
wait_event_interruptible(wait_queue, kernel_done);
```

QEMU 虚拟 GPU 执行完 kernel 后触发中断。中断 handler 里只做一件事：设置 `kernel_done = 1`，唤醒等待队列。于是 `ioctl` 返回，用户态认为这次 kernel launch 完成。

这条路径很像一个极简 CUDA launch：

```text
用户态准备 args
    ↓
ioctl 写 dispatch registers
    ↓
设备执行 kernel
    ↓
中断通知完成
    ↓
ioctl 返回
```

### QEMU 设备侧：把一个虚拟 GPU 挂到 PCI 总线上

设备侧代码位于 QEMU 源码树的 `hw/gpgpu/`。它主要分成两层：

```text
gpgpu.c       QEMU PCI device / BAR / MMIO / interrupt
gpgpu_core.c  RV32 SIMT core interpreter
```

在 QEMU 里注册设备用的是 QOM。`gpgpu_type_info` 里声明：

```text
name          = "gpgpu"
parent        = TYPE_PCI_DEVICE
instance_size = sizeof(GPGPUState)
class_init    = gpgpu_class_init
```

然后通过：

```c
type_register_static(&gpgpu_type_info);
type_init(gpgpu_register_types)
```

把这个设备类型注册进 QEMU。`gpgpu_class_init()` 里填 PCI 设备信息：

```text
vendor_id
device_id
revision
class_id
realize
exit
reset
properties
```

真正创建设备实例时，QEMU 会调用 `gpgpu_realize()`。这里完成几件关键事情：

1. `g_malloc0()` 分配一大块 host 内存作为 VRAM。
2. `memory_region_init_io()` 创建 BAR0 控制寄存器窗口。
3. `pci_register_bar(..., 0, ...)` 把 BAR0 挂到 PCI 配置空间。
4. `memory_region_init_io()` 创建 BAR2 VRAM 窗口。
5. `pci_register_bar(..., 2, ...)` 把 BAR2 暴露给 guest。
6. 创建 BAR4 doorbell 窗口。
7. 初始化 MSI/MSI-X 相关结构。
8. 创建 kernel completion timer。

所以 guest 里的 Linux 驱动看到的是一个正常 PCI 设备；驱动 `probe()` 之后映射 BAR，用户态再通过 `/dev/gpgpu` 访问它。

### QEMU MMIO：控制寄存器、VRAM 和中断

QEMU 侧 BAR0 是控制寄存器。`gpgpu_ctrl_read()` / `gpgpu_ctrl_write()` 负责处理 guest 对控制寄存器的读写。

驱动写这些寄存器：

```text
KERNEL_ADDR_LO / HI
KERNEL_ARGS_LO / HI
GRID_DIM_X/Y/Z
BLOCK_DIM_X/Y/Z
SHARED_MEM_SIZE
DISPATCH
IRQ_ENABLE
```

当 guest 写 `GPGPU_REG_DISPATCH` 时，QEMU 侧会：

```text
gpgpu_dispatch_validate()
gpgpu_core_exec_kernel()
timer_mod(kernel_timer, now + 100ms)
```

`gpgpu_core_exec_kernel()` 是同步执行的。它直接在 QEMU 进程里解释执行 kernel。执行结束后，timer 触发 `gpgpu_kernel_complete()`，设置 `KERNEL_DONE` 状态，并通过：

```c
pci_irq_assert()
pci_irq_deassert()
```

给 guest 拉一次中断线。guest 驱动里的 interrupt handler 被唤醒，`ioctl` 返回。

BAR2 是 VRAM。当前 QEMU 侧 VRAM 也很直接：

```c
memcpy(&val, s->vram_ptr + addr, size);      // read
memcpy(s->vram_ptr + addr, &val, size);      // write
```

也就是说，guest 对 BAR2 的 load/store 最终就是访问 QEMU 进程里 `vram_ptr` 指向的 host 内存。

### RV32 SIMT 解释器：用 C 模拟一颗小 GPU core

`gpgpu_core.c` 是设备侧最有意思的部分。它没有真正生成机器码，也没有接入 TCG，而是手写了一个 RV32 指令解释器。

当前支持的常见指令包括：

- load/store：`lb/lbu/lw/sb/sw`
- integer imm：`addi/slli/slti/sltiu/xori/ori/andi/srli/srai`
- integer reg：`add/sub/mul/sll/slt/sltu/xor/div/srl/sra/or/rem/and`
- branch：`beq/bne/blt/bge/bltu/bgeu`
- jump：`jal/jalr`
- CSR：主要用 `mhartid`
- float load/store：`flw/fsw`
- float arithmetic：`fadd.s/fsub.s/fmul.s/fdiv.s/fsqrt.s/fmin/fmax`
- float compare / move / convert：`flt/feq/fsgnj/fmv/fcvt`
- fused multiply-add：`fmadd.s`

为了支撑模型算子，还加了一些项目自定义或扩展指令，例如：

- `vexp.s`：用 host `expf()` 实现指数函数，用于 softmax。
- bf16/fp8 相关转换实验。
- `vshfl`：warp 内 shuffle。

这里有一个很容易踩坑的点：FP register 里存的是 32-bit bit pattern，不是 C 的 `float` 类型。因此解释 FP 指令时要反复做：

```c
u32_to_float()
float_to_u32()
```

之前 RMSNorm、fmv/fsgnj 一类问题就暴露过这个边界。

### warp、lane 和 `mhartid`

设备侧用 `GPGPUWarp` 表示一个 warp，用 `GPGPULane` 表示一个 lane。每个 lane 有自己的：

```text
gpr[32]
fpr[32]
pc
active
mhartid
```

`gpgpu_core_exec_kernel()` 会按 grid/block/warp 遍历：

```text
for block z/y/x
    for warp in block
        init warp
        a0 = kernel_args
        exec warp
```

初始化 warp 时会把 block id、warp id、thread id 编进 `mhartid`。GPU kernel 里通过读 `mhartid` 得到当前 lane 的身份，再推导 thread id、block id 等信息。这个设计很简化，但足够支撑目前的手写算子。

### warp shuffle：用 snapshot 做 lane 间通信

RMSNorm、softmax 这类算子需要 warp 内规约。为了让 lane 之间交换数据，设备侧实现了一个自定义 `vshfl`。

实现思路是：

1. 先把当前 warp 所有 lane 的源寄存器拍成 snapshot。
2. 对当前活跃 mask 里的 lane 执行：

```text
src_lane = lane_id ^ laneMask
dst = snapshot[src_lane]
```

这类似 CUDA 里的 shuffle xor 用法，可以用来写树形规约。

当前实现有一个简化假设：shuffle 最好发生在 warp-uniform 的上下文里，也就是参与 shuffle 的 lane 集合要一致。如果在严重分歧路径里做 shuffle，结果可能不符合真实 GPU 的语义。对于当前手写 kernel，我尽量让需要 shuffle 的地方保持全 warp 参与。

### 分歧执行：min-PC scheduling

真实 GPU 对 warp divergence 有 reconvergence stack 或类似机制。这个 toy core 没有实现完整 reconvergence，而是用了一个简单策略：每轮从 active lanes 里找最小 PC，只执行 PC 等于这个最小值的 lane 集合。

伪代码大概是：

```text
while active_mask != 0:
    min_pc = min(lane.pc for active lane)
    current_mask = lanes whose pc == min_pc
    insn = fetch(min_pc)
    execute insn for current_mask
```

这个策略有个好处：实现非常简单。lane 分歧后，不同 PC 的 lane 会被分批执行；当它们之后回到同一个 PC，又会自然一起执行。

但它也有明显问题：如果某些 lane 进入一个 PC 较小的循环，比如 `while (...) { ... }`，而另一些 lane 已经跳到更大的 PC，那么调度器会一直优先执行小 PC 的 lane。只要小 PC lane 不退出循环，大 PC lane 就可能长期得不到执行，表现得像饥饿。

所以当前 min-PC scheduling 适合教学和简单算子，但不是完整 GPU 分歧模型。后续如果要更接近真实硬件，需要实现 reconvergence token/stack，或者至少引入更公平的 PC group 调度。

### VRAM 分配：参考 Linux buddy + slab

用户态 `libgpgpu` 负责 VRAM 分配。这里没有向驱动申请内存，也没有真正的 GPU page table；分配器只是管理 `mmap` 出来的 BAR2 offset。

大块内存走 buddy allocator。基本思路和 Linux buddy 类似：

- VRAM 按 4KB page 管理。
- 每个 order 对应 `2^order` 个 page。
- 分配时找到能容纳请求大小的最小 order。
- 如果只有更大的块，就不断二分，把右半边挂回低一级 free list。
- 释放时找 buddy，如果 buddy 也空闲就合并成更大的块。

小块内存走 slab cache。当前有几档固定 object size：

```text
32, 64, 128, 256, 512, 2048 bytes
```

每个 slab cache 从 buddy 拿一页，再切成固定大小的小 slot，用链表管理空闲 slot。这样 kernel args 这种 32 bytes 小对象就不用浪费整页。

这个实现目前还有明显缺陷：buddy 是按 2 的幂向上取整的。比如一个 4GB VRAM，如果已经分掉了一个接近 2GB 的大块，剩余空间被切成若干块之后，再想分配一个“略大于 1/4 VRAM”的连续大块，就可能失败。不是总容量不够，而是连续 buddy block 不够。这个问题在加载大模型权重时很真实，所以 `weight_loader` 后来把权重拆成多个 region，而不是一次性申请一个巨大的连续 region。

也就是说，当前分配器更像“能支撑实验的 toy allocator”，不是成熟 GPU memory manager。

### `weight_loader`：把 Qwen 权重搬进 VRAM

`weight_loader` 的职责很单纯：把 host 文件系统里的权重二进制读出来，放到 VRAM 的固定位置，并提供“某层某个 tensor 在 VRAM 哪里”的查询接口。

它不做这些事：

- 不做 tokenizer。
- 不做 safetensors 解析。
- 不做 transpose。
- 不做 dtype 转换。
- 不做 lazy loading。
- 不做运行时 shape 推导。

这些工作都被提前放到 dump 权重的 Python 工具里完成。到了 C 侧，权重已经被整理成一组固定二进制文件：

```text
embedding.bin
lm_head.bin
final_norm.bin
layer_00.bin
...
layer_23.bin
```

每个 `layer_NN.bin` 内部的 tensor offset 是固定的，定义在 `weight_layout.h`。例如：

```text
input_norm
q_proj.w / q_proj.b
k_proj.w / k_proj.b
v_proj.w / v_proj.b
o_proj.w
post_norm
gate_proj.w
up_proj.w
down_proj.w
```

这里一个重要约定是：所有 `*_proj.weight` 已经提前转成 device `gemm.S` 能直接读取的 `[K, N]` 布局。也就是说，GPU kernel 不负责转置，`weight_loader` 也不负责转置。

加载过程大概分 6 步：

1. 根据 `weight_layout.h` 计算每个 region 需要多少字节。
2. `stat()` 检查所有权重文件大小，文件尺寸不对就直接失败。
3. 调 `gpuMalloc()` 申请 VRAM region。
4. 在每个 region 内计算 embedding、lm_head、final_norm、每层 layer base。
5. `fread()` 整个文件到 host buffer。
6. `gpuMemcpy(..., GPU_MEMCPY_H2D)` 把文件内容复制到 VRAM。

查询时不使用字符串，而是使用 enum：

```c
weight_loader_layer(wl, 12, WL_DOWN_PROJ_W)
weight_loader_embedding(wl)
weight_loader_lm_head(wl)
weight_loader_final_norm(wl)
```

这样 tensor 名写错会更早暴露，而不是运行时拼字符串失败。

#### 为什么权重要拆成 8 个 region

一开始最自然的想法是：把所有权重当成一个 2.35 GiB 的大块，一次 `gpuMalloc()`，然后按 offset 排进去。但当前 buddy allocator 会把大块请求向上取整到 2 的幂 order，2.35 GiB 这种请求会非常不友好。

后来实现改成 8 个 region：

```text
region[0] = embedding
region[1] = lm_head
region[2] = final_norm + layer 0..3
region[3] = layer 4..7
region[4] = layer 8..11
region[5] = layer 12..15
region[6] = layer 16..19
region[7] = layer 20..23
```

这样做的原因不是模型结构天然需要 8 块，而是为了绕开 buddy 大块分配的碎片和 round-up 问题。每 4 层一组大约 228 MB，buddy 会 round 到 256 MB，整体还能给 scratch、KV cache、kernel binaries 和 args buffer 留出空间。

这个设计也让调试更直观：打印 layout 时可以看到每个 region base 和每层 layer base。如果某一层权重对不上，可以直接 D2H 对应 VRAM 区间，和 host 文件某个 offset 做 bit-exact 对比。

#### 权重加载最容易踩的坑

这个项目里实际踩过一次很典型的问题：VM 里的 `/root/weights/*.bin` 和 host 端 `weights/*.bin` 不一致。表现出来像是 layer 12 的 `down_proj` GEMM 算错，但最后发现是 VM 里的 `layer_12.bin` 文件本身中间有一段全 0。

所以权重部署后建议做 checksum gate：

```bash
cd ~/gpgpu-driver
md5sum weights/*.bin

ssh -p 12055 root@localhost 'cd /root && md5sum weights/*.bin'
```

只有 host 和 VM 的权重文件一致，后面的逐层 bit-exact 对照才有意义。

### GPU kernel 算子：手写 RV32 汇编

`kernels/` 里每个 `.S` 都是一个 GPU kernel。它们不是 CUDA，也不是 OpenCL，而是设备侧 SIMT 解释器能执行的一小段 RV32 程序。

目前 transformer 路径里最关键的 kernel 包括：

- `embedding_lookup.S`：根据 token id 取 embedding 行。
- `rmsnorm.S`：做 RMSNorm，warp 内需要规约平方和。
- `gemm.S`：矩阵乘法，是 q/k/v/o/ffn/lm_head 的核心。
- `broadcast_add.S`：加 bias 或 residual。
- `rope.S`：对 Q/K 做 rotary position embedding。
- `qkt_decode.S`：decode 阶段计算 QK attention score。
- `softmax_decode.S`：对当前 token 可见的 score 做 softmax。
- `pv_decode.S`：用 attention probability 加权 V cache。
- `silu.S`：FFN gate 的激活。
- `vmul.S`：`silu(gate) * up`。
- `argmax.S`：从 logits 里取 greedy token。

每个 kernel 的参数不是 C ABI 结构自动传递，而是由用户态把 args struct 写到 VRAM，kernel 从 `a0` 指向的 args buffer 里按固定 offset `lw` 字段。因此 C 里的 args struct 字段顺序和 `.S` 里的加载顺序必须严格一致。这类 bug 非常隐蔽：字段错位通常不会 crash，只会让某个 lane 拿到奇怪地址或奇怪 shape。

### KV cache：最简单的一直追加

当前 KV cache 管理非常朴素：初始化时一次性分配 K/V 两个大 buffer：

```text
K: [num_layers, num_kv_heads, max_seq, head_dim]
V: [num_layers, num_kv_heads, max_seq, head_dim]
```

每生成一个 token，每层都会产生新的 K/V。当前做法是：

1. `SCRATCH_K` / `SCRATCH_V` 在 device 上得到当前层当前 token 的 K/V。
2. D2H 拷到 host 临时 buffer。
3. host 调 `kv_cache_append_layer()` 把它写回 KV cache 对应位置。
4. 一个 token 的 24 层都 append 完后，`kv_cache_commit_token()` 推进 `filled`。

它不会释放旧 token，也不会做 sliding window；就是从 position 0 一直追加到 `max_seq`。这和 toy decode 很匹配，但性能很差，因为每层都有 K/V 的 D2H/H2D 往返。后续如果要优化，应该写一个 device-side `kv_cache_append` kernel，让 K/V 直接在 VRAM 内搬到 cache。

### transformer：把一堆 kernel 串成 decode pipeline

`transformer.c` 做的事情不是实现一个高级框架，而是把固定 Qwen2-0.5B decode 图手工展开。当前只支持 `S_max = 1` 的 pure decode 路径。

每个 token 的流程是：

```text
embedding
    ↓
for layer in 0..23:
    input rmsnorm
    q_proj / k_proj / v_proj
    q/k bias add
    q/k rope
    append K/V cache
    qkt_decode
    softmax_decode
    pv_decode
    o_proj
    residual add
    post rmsnorm
    gate_proj / up_proj
    silu
    vmul
    down_proj
    residual add
    ↓
final rmsnorm
lm_head
argmax
```

为了省显存，workspace 里只维护几块 scratch buffer，例如：

```text
RESID_A / RESID_B
SCRATCH_X
SCRATCH_Q / SCRATCH_K / SCRATCH_V
SCRATCH_SCORES
SCRATCH_FFN_A / SCRATCH_FFN_B
LOGITS
```

这些 buffer 在不同算子之间反复复用。比如 `SCRATCH_X` 一会儿是 RMSNorm 输出，一会儿是 attention 输出，一会儿又是 down_proj 输出。这样代码可读性一般，但显存占用可控，而且很接近手写推理 engine 的真实状态。

当前 toy 推理入口 `infer_tokens` 只做 greedy generation。tokenizer 和 detokenizer 放在 host Python 侧，VM 里只处理 token id。这是为了尽快完成“文本 -> token -> GPGPU 推理 -> token -> 文本”的闭环，而不是一开始就把 tokenizer 也塞进 C 里。

## 目录结构

```text
gpgpu-driver/
├── gpgpu_drv.c                 # Linux PCI driver
├── Makefile                    # kernel module build
├── kernels/                    # hand-written RV32 GPU kernels
│   ├── gemm.S
│   ├── rmsnorm.S
│   ├── rope.S
│   ├── qkt_decode.S
│   ├── softmax_decode.S
│   ├── pv_decode.S
│   └── ...
├── lib/
│   ├── libgpgpu.c              # user-space runtime
│   ├── libgpgpu.h
│   ├── include/
│   │   ├── buddy.h
│   │   ├── slab_cache.h
│   │   ├── kv_cache.h
│   │   ├── weight_loader.h
│   │   ├── weight_layout.h
│   │   └── transformer.h
│   ├── src/
│   │   ├── buddy.c
│   │   ├── slab_cache.c
│   │   ├── kv_cache.c
│   │   ├── weight_loader.c
│   │   └── transformer.c
│   └── test/
│       ├── test_transformer.c  # reference-oriented end-to-end test
│       └── infer_tokens.c      # toy token-level inference entry
├── tools/
│   ├── dump_weights.py
│   ├── tokenize_qwen.py
│   ├── detokenize_qwen.py
│   └── check_layer.py
├── weights/                    # generated model weight binaries, not committed
├── ref/                        # reference dumps, optional
└── docs/                       # development notes
```

QEMU 设备侧代码不在本仓库中。需要配套的 QEMU virtual GPGPU device，设备侧通常包含 PCI BAR、VRAM、kernel dispatch、SIMT interpreter 等逻辑。

## 环境要求

开发环境：

- Linux host
- RISC-V Linux VM，例如 openEuler RISC-V
- QEMU，且包含配套的 virtual GPGPU device
- RISC-V Linux 交叉编译工具链：`riscv64-linux-gnu-gcc`
- RISC-V bare-metal 工具链：`riscv64-unknown-elf-gcc`
- Python 3
- Python `transformers`，用于 host 端 tokenizer / detokenizer

设备约定：

| 项目 | 值 |
| --- | --- |
| PCI Vendor ID | `0x1234` |
| PCI Device ID | `0x1337` |
| BAR0 | 控制寄存器 |
| BAR2 | VRAM |
| BAR4 | Doorbell |
| 设备节点 | `/dev/gpgpu` |

## 构建

### 编译驱动

```bash
cd ~/gpgpu-driver
make driver
```

默认 `Makefile` 使用仓库内的 `kernel-headers/`。如果你使用自己的目标内核构建目录，请修改 `KDIR`。

### 编译 GPU kernels

```bash
cd ~/gpgpu-driver/kernels
make
```

这个目录里的 `.S` 是设备侧执行的 RV32 kernel。`make` 会对每个 `.S` 生成两个文件：

```text
xxx.bin   真正加载到 VRAM 里执行的 kernel binary
xxx.dump  反汇编文本，用来检查指令编码和调试
```

例如：

```text
gemm.S
gemm.bin
gemm.dump
```

运行推理时，VM 里的 `infer_tokens` / `test_transformer` 需要能在 `<kernel_bin_dir>` 下找到这些 `.bin` 文件。前面的示例里 `<kernel_bin_dir>` 是 `/tmp`，所以一般会把 kernel binaries 传到 VM 的 `/tmp`：

```bash
cd ~/gpgpu-driver/kernels
make
scp -P 12055 *.bin root@localhost:/tmp/
```

`.dump` 不参与运行，只是给人看的反汇编。它适合用来确认伪指令展开、`funct7/funct3` 编码、分支 offset、args 结构体字段读取顺序等。

### 编译用户态 runtime 和测试程序

```bash
cd ~/gpgpu-driver/lib
make

cd ~/gpgpu-driver/lib/test
make infer_tokens
make test_transformer
```

默认编译不会打开中间 tensor dump。这个版本跑得更快，适合普通 toy 推理。

如果要做逐层 bit-exact 对照，可以加 `-DDUMP_INTERMEDIATES`：

```bash
cd ~/gpgpu-driver/lib/test
make test_transformer CFLAGS="-I.. -I../include -DDUMP_INTERMEDIATES"
```

打开后，`transformer.c` 会把中间结果写到 `dump/step_xx/`。这对定位算子错误很有用，但会显著变慢，也会生成大量 `.bin` dump 文件。普通推理不建议打开。

## 权重文件

模型权重二进制不应提交到 Git。

当前 loader 期望 `weights/` 下有以下文件：

```text
embedding.bin
lm_head.bin
final_norm.bin
layer_00.bin
layer_01.bin
...
layer_23.bin
manifest.json
```

权重 layout 定义在：

```text
lib/include/weight_layout.h
```

当前实现使用 FP32 权重布局。部分权重来源于 bf16 dump 后按 FP32 位模式展开，`weight_loader` 不做 transpose 或 dtype 转换，只按固定 offset 将文件加载到 VRAM。

可以用 `tools/dump_weights.py` 从 Hugging Face / safetensors 模型导出当前 loader 需要的二进制布局：

```bash
cd ~/gpgpu-driver
python3 tools/dump_weights.py --help
```

这个脚本依赖：

```bash
pip install torch safetensors huggingface_hub
```

它做的事情包括：

- 下载或读取 Qwen2-0.5B-Instruct 权重。
- 将 bf16/fp16/fp32 tensor 转成当前项目使用的 FP32 bytes。
- 将各个 `*_proj.weight` transpose 成 device `gemm.S` 直接读取的 `[K, N]` 布局。
- 按 `weight_layout.h` 约定的 offset 生成 `layer_NN.bin`。
- 生成 `manifest.json`，用于记录 shape、hash 和 sanity 信息。

权重 dump 产物比较大，`weights/*.bin` 不应该提交到 Git。

部署到 VM 后，建议先校验 host 和 VM 的权重一致性：

```bash
cd ~/gpgpu-driver
md5sum weights/*.bin

ssh -p 12055 root@localhost 'cd /root && md5sum weights/*.bin'
```

如果权重文件不一致，后续算子调试会被错误输入带偏。

## 运行 toy 推理

当前 toy 推理采用：

```text
host Python tokenizer
    ↓
input_tokens.txt
    ↓
VM /root/infer_tokens
    ↓
output_tokens.txt
    ↓
host Python detokenizer
```

### 1. host 端生成 token ids

`infer_tokens` 不直接处理中文文本，它只认识 token id。文本到 token id 的转换由 host 端 Python 脚本完成：

```bash
cd ~/gpgpu-driver

python3 tools/tokenize_qwen.py \
  --model Qwen/Qwen2-0.5B-Instruct \
  --chat \
  --text "你好，请介绍一下你自己" \
  --out /tmp/input_tokens.txt
```

`--chat` 会调用 Qwen tokenizer 的 chat template，适合 Qwen2-Instruct 这类模型。它会自动拼出类似 system/user/assistant 的特殊 token 格式，并在末尾添加 generation prompt。

如果只是想把普通文本直接编码，不套 chat template，可以不用 `--chat`：

```bash
python3 tools/tokenize_qwen.py \
  --model Qwen/Qwen2-0.5B-Instruct \
  --text "你好" \
  --out /tmp/input_tokens.txt
```

如果本地已有 tokenizer，可以把 `--model` 换成本地模型/tokenizer 目录，例如：

```bash
python3 tools/tokenize_qwen.py \
  --model /home/hp/models/Qwen2-0.5B-Instruct \
  --chat \
  --text "你好，请介绍一下你自己" \
  --out /tmp/input_tokens.txt
```

检查输出应该是一串数字：

```bash
cat /tmp/input_tokens.txt
```

例如：

```text
151644 8948 198 ... 151644 77091 198
```

如果看到的是 `input_ids attention_mask` 这种文本，说明 tokenizer 返回值没有被正确取出，需要检查 `tools/tokenize_qwen.py` 或 transformers 版本。

### 2. 上传输入 token ids

```bash
scp -P 12055 /tmp/input_tokens.txt root@localhost:/root/input_tokens.txt
```

### 3. VM 端运行推理

先部署驱动、kernel binaries、权重和 `infer_tokens` 到 VM。

VM 内执行：

```bash
rmmod gpgpu_drv 2>/dev/null
insmod /root/gpgpu_drv.ko

cd /root
./infer_tokens ./weights /tmp /root/input_tokens.txt 16 /root/output_tokens.txt 151645
```

参数含义：

```text
./infer_tokens <weights_dir> <kernel_bin_dir> <input_tokens.txt> <max_new_tokens> <output_tokens.txt> [eos_token_id]
```

其中 `151645` 是 Qwen2 常见的 eos token id。也可以不传 eos，只按 `max_new_tokens` 固定生成。

### 4. 拉回输出并 detokenize

`infer_tokens` 输出的仍然是 token id，不是中文文本。先把 VM 生成的 token 文件拉回 host：

```bash
scp -P 12055 root@localhost:/root/output_tokens.txt /tmp/output_tokens.txt
cat /tmp/output_tokens.txt
```

然后用同一个 tokenizer 做 detokenize：

```bash
cd ~/gpgpu-driver
python3 tools/detokenize_qwen.py \
  --model Qwen/Qwen2-0.5B-Instruct \
  --ids /tmp/output_tokens.txt \
  --skip-special
```

如果使用本地 tokenizer，`--model` 同样换成本地路径：

```bash
python3 tools/detokenize_qwen.py \
  --model /home/hp/models/Qwen2-0.5B-Instruct \
  --ids /tmp/output_tokens.txt \
  --skip-special
```

当前 toy 入口只把“新生成 token”写入 `output_tokens.txt`，不包含 prompt token。如果你想看完整上下文，可以把 `input_tokens.txt` 和 `output_tokens.txt` 拼起来再 detokenize。

## 测试

端到端 transformer sanity test：

```bash
cd ~/gpgpu-driver/lib/test
make test_transformer
scp -P 12055 test_transformer root@localhost:/root/

ssh -p 12055 root@localhost
cd /root
./test_transformer ./weights /tmp
```

预期至少看到：

```text
prefill last (expected 3):    3  PASS
decode step 0 (expected 323): 323  PASS
```

逐层 dump 对照工具：

```bash
python3 tools/check_layer.py
```

## 已实现的 GPU kernels

当前 `kernels/` 中包含：

- `gemm.S`
- `rmsnorm.S`
- `rope.S`
- `embedding_lookup.S`
- `broadcast_add.S`
- `silu.S`
- `vmul.S`
- `qkt_decode.S`
- `softmax_decode.S`
- `pv_decode.S`
- `argmax.S`
- 以及若干单算子测试 kernel

## 已知限制

- 这是实验项目，不是生产推理框架。
- 当前 Qwen 路径是 FP32 权重，不是 INT8。
- tokenizer / detokenizer 在 host Python 侧运行。
- 生成策略只有 greedy / argmax。
- 不支持 batch。
- `prefill` 当前按多次 single-token step 实现。
- KV cache append 仍走 host-mediated D2H/H2D 路径。
- 每个算子独立 kernel launch，性能不是当前目标。
- 依赖配套 QEMU virtual GPGPU device，单独 clone 本仓库不能直接在普通机器上运行 GPU 部分。

## 路线图

- [x] PCI driver + char device
- [x] BAR mmap
- [x] ioctl kernel dispatch
- [x] interrupt-based completion
- [x] user-space runtime
- [x] Buddy + slab allocator
- [x] RV32 GPU kernel toolchain
- [x] core transformer kernels
- [x] Qwen2-0.5B FP32 weight loader
- [x] token-level greedy inference demo
- [ ] device-side KV cache append
- [ ] host tokenizer 集成到完整 demo 脚本
- [ ] INT8 weight layout
- [ ] INT8 GEMM / dequant
- [ ] kernel fusion / launch overhead optimization
- [ ] 更完整的 reference regression suite

## 调试经验

这个项目中遇到过一些典型底层问题：

- RISC-V virt PLIC 不支持 MSI-X，驱动改用 Legacy INTx。
- `mmap` offset 需要区分 BAR0 和 BAR2，否则会把控制寄存器当 VRAM。
- FP register 在解释器中存的是 bit pattern，所有 FP 读写必须显式转换。
- `kernel_args` MMIO register 漏实现时，kernel 会读到错误 args 地址。
- VM 里的权重文件可能和 host 不一致，必须用 checksum gate 防止错误输入污染算子调试。

## License

尚未选择许可证。公开发布前建议添加 `LICENSE` 文件，例如 MIT 或 Apache-2.0。
