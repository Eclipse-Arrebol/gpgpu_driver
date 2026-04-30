# 4.30
- 实现了gpu中的分歧时如何执行的，采用了min-PC scheduling 的实现方式，但是这种实现方式可以会导致一直执行pc持续较小的lane，从而饿死另一组lane

- 编写了对min-PC 实现的测试脚本，测试没有问题

- 对RMSNORM需要获取不同线程的信息，就要实现线程中的信息通信，我使用的一个warp中简单的Shuffle 来实现线程中的信息交换，其原理和树有点像
    - 但是shuffle有个缺点就是一个warp中必须全部活跃才能实现，信息的交换，不然结果可能会错
    - 早期 CUDA 文档明确要求 shuffle 必须在"warp-uniform 上下文"中使用。新版（Volta+）引入了带 mask 的 __shfl_sync(mask, ...) 来允许部分参与，但那是后话。
    - 目前的设计照 "必须全 lane 参与" 的简单约束来

