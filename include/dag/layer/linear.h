/*
 * @Author: liushijie
 * @Date: 2020-06-20 18:50:56
 * @LastEditTime: 2020-06-29 17:42:43
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/layer/linear.h
 */
#pragma once
#include "dag/layer/layer.h"

namespace dl {
namespace nn {

class Linear : public LayerBase {
  public:
    explicit Linear(int64_t in_features, int64_t out_features, bool bias)
        : mInFeatures(in_features)
        , mOutFeatures(out_features)
        , mBiasFlag(bias)
        , mWeights(nullptr)
        , mBias(nullptr) {
        mWeights = CreateNode({mInFeatures, mOutFeatures}, true);
        LOG_INFO("\n\tlinear args:\n\t\tin_features:%lu\n\t\tout_features:%lu\n\t\tbias:%s", in_features, out_features, bias ? "true": "false");
        Random(mWeights->tensor()->data(), mWeights->tensor()->size());
        if (mBiasFlag) {
            mBias = CreateNode({mOutFeatures}, true);
            Random(mBias->tensor()->data(), mBias->tensor()->size());
        }
    }

    DataNode *operator()(DataNode *x) {
        DataNode *out_x = CreateNode<MatMulImpl>({x, mWeights});
        if (mBiasFlag) {
            out_x = CreateNode<AddImpl>({out_x, mBias});
        }
        LOG_INFO("\n\tlinear input shape:%s\n\tlinear output shape:%s", FormatShape(x->tensor()->shape()).c_str(), FormatShape(out_x->tensor()->shape()).c_str());
        return out_x;
    }

  private:
    int64_t    mInFeatures;
    int64_t    mOutFeatures;
    bool      mBiasFlag;
    DataNode *mWeights;
    DataNode *mBias;
};
}
}