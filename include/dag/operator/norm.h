/*
 * @Author: liushijie
 * @Date: 2020-06-20 14:35:52
 * @LastEditTime: 2020-06-25 16:39:53
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/operator/norm.h
 */
#pragma once
#include "dag/node.h"

namespace dl {
class BatchNormImpl : public OperatorNodeBase {
  public:
    BatchNormImpl(int uid, float eps, float momentum, bool track_running_stats)
        : OperatorNodeBase(uid)
        , mEps(eps)
        , mMomentum(momentum)
        , mTrackRunningStats(track_running_stats) {}

    ~BatchNormImpl() = default;

    void forward(const std::vector<const Tensor *> &inps,
                 Tensor *                           norm_result) override {
        CHECK_EQ(inps.size(), 3);
        const Tensor *norm_tensor = inps[0];
        const Tensor *moving_mean = inps[1];
        const Tensor *moving_var  = inps[2];
        Sub(*norm_tensor, *moving_mean, norm_result);
        Div(*norm_result, *moving_var, norm_result);
    }

    void backward(const Tensor *diff, std::vector<Tensor *> &grads) override {}

    Shape inferenceShape(const std::vector<const Tensor *> &inps) override {
        return inps[0]->shape();
    }

    float mEps;
    float mMomentum;
    bool  mTrackRunningStats;
};
}