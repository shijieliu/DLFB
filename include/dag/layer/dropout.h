/*
 * @Author: liushijie
 * @Date: 2020-06-20 15:30:38
 * @LastEditTime: 2020-06-22 10:47:30
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/layer/dropout.h
 */
#pragma once
#include "dag/layer/layer.h"

namespace dl {
namespace nn {
class Dropout : public LayerBase {
  public:
    explicit Dropout(float p)
        : mP(p) {}

    DataNode *operator()(DataNode *inp_node) {
        return CreateNode<DropoutImpl, float>({inp_node}, mP);
    }

  private:
    float mP;
};

} // namespace nn
}