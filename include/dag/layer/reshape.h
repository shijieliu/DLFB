/*
 * @Author: liushijie
 * @Date: 2020-06-29 14:57:14
 * @LastEditTime: 2020-06-29 15:02:46
 * @LastEditors: liushijie
 * @Description: 
 * @FilePath: /LightLR/include/dag/layer/reshape.h
 */ 

#pragma once
#include "dag/layer/layer.h"

namespace dl {
namespace nn {

class Reshape : public LayerBase {
  public:
    explicit Reshape(std::initializer_list<int64_t> shape)
        : mOutShape(shape.begin(), shape.end()) {
        LOG_INFO("\n\treshape args:\n\t\tout shape:%s", FormatShape(mOutShape).c_str());
    }

    DataNode *operator()(DataNode *x) {
        DataNode *out_x = CreateNode<ReshapeImpl, Shape>({x}, mOutShape);
        LOG_INFO("\n\tinput shape:%s\n\toutput shape:%s", FormatShape(x->tensor()->shape()).c_str(), FormatShape(out_x->tensor()->shape()).c_str());
        return out_x;
    }

  Shape mOutShape;
};
}
}