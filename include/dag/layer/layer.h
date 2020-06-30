/*
 * @Author: liushijie
 * @Date: 2020-06-20 18:43:52
 * @LastEditTime: 2020-06-29 19:24:16
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/layer/layer.h
 */
#pragma once

#include "dag/graph.h"
#include "dag/node.h"
#include "dag/operator/activation.h"
#include "dag/operator/add.h"
#include "dag/operator/conv.h"
#include "dag/operator/crossentropy.h"
#include "dag/operator/dropout.h"
#include "dag/operator/mul.h"
#include "dag/operator/norm.h"
#include "dag/operator/pooling.h"
#include "dag/operator/reshape.h"
#include "dag/operator/softmax.h"
#include "dag/operator/sub.h"
#include "dag/operator/reduce.h"
#include "dag/operator/dataprovider.h"
#include "random.h"

namespace dl {
namespace nn {
class LayerBase {
  protected:
    LayerBase() {}
    ~LayerBase() {}

  public:
    LayerBase(const LayerBase &)  = delete;
    LayerBase(const LayerBase &&) = delete;
    LayerBase &operator=(const LayerBase &) = delete;
    LayerBase &operator=(const LayerBase &&) = delete;
};
}
}