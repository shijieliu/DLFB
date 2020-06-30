/*
 * @Author: your name
 * @Date: 2020-06-20 13:58:16
 * @LastEditTime: 2020-06-25 21:17:33
 * @LastEditors: liushijie
 * @Description: In User Settings Edit
 * @FilePath: /LightLR/include/dag/operator/reduce.h
 */
#pragma once
#include "dag/node.h"

namespace dl {
enum ReduceType { Sum, Mean };

template <ReduceType reduce> class ReduceOpImpl : public OperatorNodeBase {
  public:
    ReduceOpImpl(int64_t uid)
        : OperatorNodeBase(uid) {}

    ~ReduceOpImpl() = default;

    void forward(const std::vector<const Tensor *> &inps,
                 Tensor *                           out) override {
        CHECK_EQ(inps.size(), 1);
        float sum =
            std::accumulate(inps[0]->data(), inps[0]->data() + inps[0]->size(), 0);
        LOG_DEBUG("reduce sum %f", sum);
        if (reduce == ReduceType::Mean) {
            sum /= inps[0]->size();
        }
        *(out->data()) = sum;
    }
    
    void backward(const Tensor *diff, std::vector<Tensor *> &grads) override {
        CHECK_EQ(grads.size(), 1);
        CHECK_EQ(diff->size(), 1);
        auto g = grads[0];
        for (int64_t i = 0; i < g->size(); ++i) {
            if (reduce == ReduceType::Mean) {
                g->data()[i] = diff->data()[0] / g->size();
            }
            if (reduce == ReduceType::Sum) {
                g->data()[i] = diff->data()[0];
            }
        }
    }

    Shape inferenceShape(const std::vector<const Tensor *> &inps) override {
        return Shape({1});
    }
};

using ReduceMeanImpl = ReduceOpImpl<ReduceType::Mean>;

using ReduceSumImpl = ReduceOpImpl<ReduceType::Sum>;
}