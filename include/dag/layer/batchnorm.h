/*
 * @Author: liushijie
 * @Date: 2020-06-20 15:30:24
 * @LastEditTime: 2020-06-25 16:41:07
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/layer/batchnorm.h
 */
#pragma once
#include "dag/layer/layer.h"
#include "dag/operator/add.h"
#include "dag/operator/mul.h"
#include "dag/operator/norm.h"
namespace dl {
namespace nn {
class BatchNorm2D : public LayerBase {
  public:
    BatchNorm2D(int num_features, float eps = 1e-05, float momentum = 0.1,
                bool affine = true, bool track_running_stats = true)
        : mNumFeatures(num_features)
        , mEps(eps)
        , mMonmentum(momentum)
        , mAffine(affine)
        , mTrackRunningStats(track_running_stats)
        , mGamma(nullptr)
        , mBeta(nullptr) {
        if (mAffine) {
            mGamma = CreateNode({mNumFeatures});
            mBeta  = CreateNode({mNumFeatures});
        }
    }
    /**
     * @description:
     * @param {type} inp_node (n, c, h, w)
     * @return:
     */
    DataNode *operator()(DataNode *inp_node) {
        DataNode *norm_result = CreateNode<BatchNormImpl, float, float, bool>(
            {inp_node, mMovingMean, mMovingVar}, mEps, mMonmentum,
            mTrackRunningStats);

        if (mAffine) {
            norm_result = CreateNode<MulImpl>({norm_result, mGamma});
            norm_result = CreateNode<AddImpl>({norm_result, mBeta});
        }
        return norm_result;
    }

  private:
    int    mNumFeatures;
    float     mEps;
    float     mMonmentum;
    bool      mAffine;
    bool      mTrackRunningStats;
    DataNode *mGamma;
    DataNode *mBeta;
    DataNode *mMovingMean;
    DataNode *mMovingVar;
};
}
}