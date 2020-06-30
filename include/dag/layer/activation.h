/*
 * @Author: liushijie
 * @Date: 2020-06-20 19:02:32
 * @LastEditTime: 2020-06-28 19:33:40
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/layer/activation.h
 */
#pragma once
#include "dag/layer/layer.h"

namespace dl {
namespace nn {
template <typename ActivationImpl> class Activation : public LayerBase {
  public:
    Activation() {}

    DataNode *operator()(DataNode *pre_activation) {
        return CreateNode<ActivationImpl>({pre_activation});
    }
};

using ReLU = Activation<ReLUImpl>;

}
}