/*
 * @Author: liushijie
 * @Date: 2020-06-20 19:07:24
 * @LastEditTime: 2020-06-30 13:36:45
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/operator/activation.h
 */

#pragma once

#include "dag/node.h"
#include <numeric>
#include <unordered_set>

namespace dl {
class ReLUImpl : public OperatorNodeBase {
  public:
    ReLUImpl(int64_t uid)
        : OperatorNodeBase(uid) {}
    virtual ~ReLUImpl() = default;

    void forward(const std::vector<const Tensor *> &inps,
                 Tensor *                           outs) override {
        mActivationIndex.clear();
        CHECK_EQ(inps.size(), 1);
        const Tensor *before_activation = inps[0];
        CHECK_EQ(before_activation->size(), outs->size());
        std::vector<int64_t> index(outs->size());
        std::iota(index.begin(), index.end(), 0);

        std::transform(index.begin(), index.end(), outs->data(),
                       [&, this](int64_t idx) -> float {
                           if (before_activation->data()[idx] > 0) {
                               mActivationIndex.insert(idx);
                               return before_activation->data()[idx];
                           } else {
                               return 0.0f;
                           }
                       });
    }

    void backward(const Tensor *diff, std::vector<Tensor *> &grads) override {
        CHECK_EQ(grads.size(), 1);
        Tensor *             delta = grads[0];
        std::vector<int64_t> index(delta->size());
        std::iota(index.begin(), index.end(), 0);

        std::transform(index.begin(), index.end(), delta->data(),
                       [&](int64_t idx) -> float {
                           if (mActivationIndex.find(idx) !=
                               mActivationIndex.end()) {
                               return diff->data()[idx];
                           } else {
                               return 0.0f;
                           }
                       });
    }

    Shape inferenceShape(const std::vector<const Tensor *> &inps) override {
        CHECK_EQ(inps.size(), 1);
        return inps[0]->shape();
    }

    std::unordered_set<int64_t> mActivationIndex;
};
}