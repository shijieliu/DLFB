/*
 * @Author: liushijie
 * @Date: 2020-06-25 22:39:32
 * @LastEditTime: 2020-06-29 15:55:37
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/layer/crossentropy.h
 */

#pragma once

#include "dag/layer/layer.h"

namespace dl {
namespace nn {
class CrossEntropy : public LayerBase {
  public:
    CrossEntropy() {}
    DataNode *operator()(DataNode *logits, DataNode *label) {
        DataNode *loss = CreateNode<CrossEntropyImpl>({logits, label});
        LOG_INFO("\n\tcrossentropy logits shape:%s\n\tcrossentropy label "
                 "shape:%s\n\tcrossentropy loss shape:%s",
                 FormatShape(logits->tensor()->shape()).c_str(),
                 FormatShape(label->tensor()->shape()).c_str(),
                 FormatShape(loss->tensor()->shape()).c_str());
        return loss;
    }
};
}
}