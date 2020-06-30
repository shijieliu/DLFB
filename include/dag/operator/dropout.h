/*
 * @Author: liushijie
 * @Date: 2020-06-22 10:41:00
 * @LastEditTime: 2020-06-23 18:30:10
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/operator/dropout.h
 */
#pragma once
#include "dag/node.h"

namespace dl {
class DropoutImpl final : public OperatorNodeBase {
  public:
    DropoutImpl(int uid, float p)
        : OperatorNodeBase(uid)
        , mP(p) {}
    virtual ~DropoutImpl() = default;

    void forward(const std::vector<const Tensor *> &inps,
                 Tensor *                           outs) override {
        CHECK_EQ(inps.size(), 1);
        const Tensor *before_dropout = inps[0];
        CHECK_EQ(before_dropout->size(), outs->size());
        mDropoutIndex.clear();
        if (mTrainFlag) {
            std::vector<int> index(outs->size());
            std::iota(index.begin(), index.end(), 0);

            std::transform(index.begin(), index.end(), outs->data(),
                           [&](int idx) -> float {
                               if (static_cast<float>(rand()) / RAND_MAX <
                                   0.5) {
                                   mDropoutIndex.insert(idx);
                                   return 0.0f;
                               } else {
                                   return before_dropout->data()[idx];
                               }
                           });
        } else {
            // eval mode
            outs->copyFrom(before_dropout->data(),
                              before_dropout->data() + before_dropout->size());
        }
    }

    void backward(const Tensor *diff, std::vector<Tensor *> &grads) override {
        CHECK_EQ(grads.size(), 1);
        Tensor *            grad = grads[0];
        std::vector<int> index(grad->size());
        std::iota(index.begin(), index.end(), 0);

        std::transform(index.begin(), index.end(), grad->data(),
                       [&](int idx) -> float {
                           if (mDropoutIndex.find(idx) != mDropoutIndex.end()) {
                               return diff->data()[idx];
                           } else {
                               return 0.0f;
                           }
                       });
    }

    Shape inferenceShape(const std::vector<const Tensor *> &inps) override {
        Shape res;
        for (const Tensor *t : inps) {
            if (res.size() == 0) {
                res = t->shape();
            }
            assert(res == t->shape());
        }
        return res;
    }
    float                      mP;
    std::unordered_set<int> mDropoutIndex;
};
}