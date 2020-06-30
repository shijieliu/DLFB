/*
 * @Author: liushijie
 * @Date: 2020-06-25 22:43:08
 * @LastEditTime: 2020-06-29 19:13:23
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/operator/crossentropy.h
 */

#pragma once

#include "dag/node.h"

namespace dl {
class CrossEntropyImpl : public OperatorNodeBase {
  public:
    CrossEntropyImpl(int64_t uid)
        : OperatorNodeBase(uid) {}
    ~CrossEntropyImpl() = default;

    void forward(const std::vector<const Tensor *> &inps,
                 Tensor *                           out) override {
        CHECK_EQ(inps.size(), 2);
        const Tensor *logits = inps[0]; // (n, c)
        const Tensor *labels = inps[1]; // (n)

        int64_t num_classes = logits->shape()[1];
        for (int64_t n = 0; n < logits->shape()[0]; ++n) {
            out->data()[n] =
                logits->data()[n * num_classes +
                               static_cast<int64_t>(labels->data()[n])];
        }
        mLabelIdx = *labels;
    }

    void backward(const Tensor *diff, std::vector<Tensor *> &grads) override {
        Tensor *grad_logits = grads[0];
        int64_t num_classes = grad_logits->shape()[1];
        for(int64_t n = 0; n < diff->shape()[0]; ++n){
            grad_logits->data()[n * num_classes + static_cast<int64_t>(mLabelIdx.data()[n])] = diff->data()[n];
        }
    }
    Shape inferenceShape(const std::vector<const Tensor *> &inps) override {
        return Shape({inps[0]->shape()[0]});
    }

    Tensor mLabelIdx;
};
}