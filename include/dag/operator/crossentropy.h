/*
 * @Author: liushijie
 * @Date: 2020-06-25 22:43:08
 * @LastEditTime: 2020-07-01 05:36:08
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/operator/crossentropy.h
 */

#pragma once

#include "dag/node.h"

namespace dl {
class CrossEntropyImpl : public OperatorNodeBase {
  public:
    CrossEntropyImpl(int uid)
        : OperatorNodeBase(uid) {}
    ~CrossEntropyImpl() = default;

    void forward(const std::vector<const Tensor *> &inps,
                 Tensor *                           out) override {
        CHECK_EQ(inps.size(), 2);
        const Tensor *logits = inps[0]; // (n, c)
        const Tensor *labels = inps[1]; // (n)
        mLabelIdx.clear();
        
        int num_classes = logits->shape()[1];
        for (int n = 0; n < logits->shape()[0]; ++n) {
            int idx = static_cast<int>(labels->data()[n]);
            if(idx >= num_classes){
                LOG_ERROR("idx %d out of range %d", idx, num_classes);
            }
            float activation_value = logits->data()[n * num_classes + idx] + 1e-12;
            out->data()[n] =
                -std::log2(activation_value);
            mLabelIdx[n] = GradPack(idx, 1. / activation_value);
        }
    }

    void backward(const Tensor *diff, std::vector<Tensor *> &grads) override {
        Tensor *grad_logits = grads[0];
        int num_classes = grad_logits->shape()[1];
        for(int n = 0; n < diff->shape()[0]; ++n){
            grad_logits->data()[n * num_classes + mLabelIdx[n].idx] = diff->data()[n] * mLabelIdx[n].grad;
        }
    }
    
    Shape inferenceShape(const std::vector<const Tensor *> &inps) override {
        return Shape({inps[0]->shape()[0]});
    }
    struct GradPack{
        GradPack() {}
        GradPack(int idx_, float grad_): idx(idx_), grad(grad_) {}
        int idx;
        float grad;
    };
    std::unordered_map<int, GradPack> mLabelIdx;
};
}