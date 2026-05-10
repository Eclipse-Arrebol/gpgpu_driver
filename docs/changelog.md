# 4.30
- 实现了gpu中的分歧时如何执行的，采用了min-PC scheduling 的实现方式，但是这种实现方式可以会导致一直执行pc持续较小的lane，从而饿死另一组lane

- 编写了对min-PC 实现的测试脚本，测试没有问题

- 对RMSNORM需要获取不同线程的信息，就要实现线程中的信息通信，我使用的一个warp中简单的Shuffle 来实现线程中的信息交换，其原理和树有点像
    - 但是shuffle有个缺点就是一个warp中必须全部活跃才能实现，信息的交换，不然结果可能会错
    - 早期 CUDA 文档明确要求 shuffle 必须在"warp-uniform 上下文"中使用。新版（Volta+）引入了带 mask 的 __shfl_sync(mask, ...) 来允许部分参与，但那是后话。
    - 目前的设计照 "必须全 lane 参与" 的简单约束来

# 5.2
    这写rmsnorm算子的时候遇到了一个问题就是slab是分配一页的大小即4096，但是如果分配的内存超过4096，之前的逻辑就是没有实现扩容，导致分配空间不足，返回错误，为了实现这个扩容的功能，新增了一个数据结果叫做slab_meta,用于记录每一个slab的大小和空间和类型，扩容的时候需要将slab切割，然后将其切割的部分加入空闲列表，但是free的时候如果整个页都空闲的话，一开始的设计是将整个页都回收，但是怎么找到整个页的内容呢，其在空闲链表中是散的，只能使用遍历地址，看看满不满足meta中的数据信息，但是这有引发了一个问题就是一空闲就回收会到致系统频繁的回收分配内存，所有最终就使用了保留一个页的方式。

    在重新测试 test_fmadd 时,32 个 c 元素读出来全是 1.0,说明 kernel 的 fsw 没生效。我做了两轮 debug:
    在 host 直接读 vram[c_off],仍然是 1.0,确认是 kernel 没写,不是 D2H 拷贝问题
    让 kernel 把 a0 / a3 / t4 写到已知位置,host 读出来全是 0x40000000,而这正好是 float 2.0f 的位模式——意味着 kernel 的 debug sw 根本没改 vram,反推 a0 错了

    查 QEMU gpgpu_ctrl_write 发现 GPGPU_REG_KERNEL_ARGS_LO/HI 没有 case,驱动写过来的 args 地址被静默丢弃,s->kernel.kernel_args 始终是 reset 时的 0。
    但这个 bug 一直存在,之前测试为什么过了?因为旧 slab 的切片顺序导致 args_off 恰好是 0,和 QEMU 这边的 0 巧合一致,kernel 用 a0=0 读 vram[0] 也能拿到 args。slab 重构改了切片顺序(头插),args_off 变成 0xfe0,巧合不再成立,bug 暴露。
    slab 重构没引入 bug,它揭露了潜伏的 bug。

# 5.3 
    今天测试了rmsnorm 发现测试失败，反推实现的全局平方和为32，不是896，首先怀疑shuffle有问题，但是测试没有问题，那就是之前的ft0是1，为什么了，发现fsgnj.s (funct7=0x10) 在 QEMU 解释器里没实现，fmv.s ft0, ft1 是 fsgnj.s ft0, ft1, ft1 的伪指令(funct7=0x10, funct3=0)。你的 exec_fp 里没有处理这个 funct7,指令被静默吞掉。

    吸取了之前的教训，我现在实现了一个小功能就测试一下，避免找bug麻烦，今天是实现了softmax和silu，感觉都没什么困难

# 5.4
    今天实现了attention中的一部分qkt_scale 怎么说呢，感觉QK矩阵的乘法还是有点难懂，加上还要算输入的base，还要一堆的寄存器要记，感觉写汇编还是难度很大的，
    总的来说，工作主要是一个循环算输出的8个元素，内部的循环算一行的点积然后乘以scale


# 5.5
    今天实现了softmax16和pv模块，和完整拼好了attention模块，但是这次的汇编基本都是ai写的，让我深感对自己算法能力的欠缺，我决定每天刷两道算法题，虽然不知道能坚持多久


# 5.6
    今天实现了embedding_lookup vmul 重写了gemm ，开始感觉汇编写的越来越流畅了，


# 5.8 
    实现了repo的kernel部分，我发现实现代码其实是比较简单的，关键是要理解算子的内存分布





