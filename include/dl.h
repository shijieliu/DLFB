/*
 * @Author: liushijie
 * @Date: 2020-06-15 21:08:48
 * @Last Modified by:   liushijie
 * @Last Modified time: 2020-06-15 21:08:48
 */
#pragma once

#include "dag/graph.h"
#include "dag/layer/activation.h"
#include "dag/layer/batchnorm.h"
#include "dag/layer/conv.h"
#include "dag/layer/crossentropy.h"
#include "dag/layer/dropout.h"
#include "dag/layer/linear.h"
#include "dag/layer/pooling.h"
#include "dag/layer/reshape.h"
#include "dag/layer/softmax.h"
#include "dag/optim.h"
#include "dag/node.h"
#include "dag/tensor.h"
#include "io.h"
#include "random.h"

namespace dl {
inline void Init() { Graph::GetInstance(); }

inline Graph::GraphExecutor Compile(std::initializer_list<DataNode *> src, std::initializer_list<DataNode *> dst) {
    Graph &graph = Graph::GetInstance();
    return graph.compile(src, dst);
}
}