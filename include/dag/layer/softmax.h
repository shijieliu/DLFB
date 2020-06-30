/*
 * @Author: liushijie
 * @Date: 2020-06-23 18:21:28
 * @LastEditTime: 2020-06-29 15:52:28
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/layer/softmax.h
 */

#pragma once
#include "dag/layer/layer.h"

namespace dl {
namespace nn {
class Softmax : public LayerBase {
  public:
    explicit Softmax() {}
    DataNode *operator()(DataNode *x) {
        DataNode *out = CreateNode<SoftmaxImpl>({x});
        LOG_INFO("\n\tsoftmax input shape:%s\n\tsoftmax output shape:%s",
                 FormatShape(x->tensor()->shape()).c_str(),
                 FormatShape(out->tensor()->shape()).c_str());
        return out;
    }
};
}
}