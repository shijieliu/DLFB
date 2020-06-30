<!--
 * @Author: liushijie
 * @Date: 2020-06-22 11:22:04
 * @LastEditTime: 2020-06-25 10:31:47
 * @LastEditors: liushijie
 * @Description: 
 * @FilePath: /LightLR/README.md
--> 
1. 管理 node 的功能
2. run graph 的功能

# 6.17
1. 底层优化，avx 支持向量化，openmp 支持。这部分可以尽量用现有的实现，不强求最优
2. 图级别：目前的算子不够；backward 没写；提供 Layer 级别的抽象；Opr 支持参数。这部分，需要尽快完成一个小的 cnn demo，因为目前 loss 和 optimizer 还没有实现
3. 多机：需要增加对 net 的更多的支持，目前的 api 还太底层；定义支持 allreduce 的抽象层级。这部分应该在2的基础上，可以复制 graph，然后通过 graph 的接口进行处理
4. 量化：PQ 可以加上吗？主要用在 search 上？简易的分布式 search？可以支持一个 demo
5. benchmark

# 6.22
1. conv 的实现；cuda 支持；如何在上层能够调用；需要条件编译；从底层的数据开始，cpu/gpu 需要区分
2. 图级别：还需要 optimizer 的支持；需要 train/val flag 区分
3. 基本只需要实现 allreduce 即可

# 6.25
1. 完成 conv，batchnorm，pooling 等 op 和 layer
2. 完成optim
3. 完成 softmax loss
4. 完成一个简单的 cnn demo