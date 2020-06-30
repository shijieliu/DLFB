/*
 * @Author: liushijie
 * @Date: 2020-06-20 15:30:30
 * @LastEditTime: 2020-06-29 12:01:26
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/layer/conv.h
 */
#pragma once
#include "dag/layer/layer.h"

namespace dl {
namespace nn {
class Conv2D : public LayerBase {
  public:
    explicit Conv2D(int in_channels, int out_channels, int kernel_size,
                    int stride = 1, int padding = 0, bool bias = true,
                    const char *padding_mode = "zeros")
        : mInChannels(in_channels)
        , mOutChannels(out_channels)
        , mKernelSize(kernel_size)
        , mStride(stride)
        , mPadding(padding)
        , mBiasFlag(bias)
        , mPaddingMode(padding_mode)
        , mWeight(nullptr)
        , mBias(nullptr) {
        mWeight =
            CreateNode({mOutChannels, mInChannels, mKernelSize, mKernelSize}, true);
        Random(mWeight->tensor()->data(), mWeight->tensor()->size());

        LOG_INFO("\n\tconv args:\n\t\tin_channels:%d\n\t\tout_channels:%d\n\t\tkernel_size:%d\n\t\tstride:%d\n\t\tpadding:%d\n\t\tpaddingmode:%s\n\t\tbias:%s", mInChannels, mOutChannels, mKernelSize, mStride, mPadding, mPaddingMode.c_str(), bias ? "true": "false");
        if (mBiasFlag) {
            mBias = CreateNode({mOutChannels}, true);
            Random(mBias->tensor()->data(), mBias->tensor()->size());
        }
    }

    DataNode *operator()(DataNode *inp_node) {
        DataNode *out_node =
            CreateNode<Conv2DImpl, int, int, int, const char *>(
                {inp_node, mWeight}, mKernelSize, mStride, mPadding,
                mPaddingMode.c_str());

        if (mBias) {
            out_node = CreateNode<AddImpl>({out_node, mBias});
        }
        LOG_INFO("\n\tconv input shape: %s\n\tconv output shape: %s", FormatShape(inp_node->tensor()->shape()).c_str(), FormatShape(out_node->tensor()->shape()).c_str());
        return out_node;
    }

  private:
    int      mInChannels;
    int      mOutChannels;
    int      mKernelSize;
    int      mStride;
    int      mPadding;
    bool        mBiasFlag;
    std::string mPaddingMode;
    DataNode *  mWeight;
    DataNode *  mBias;
};
}
}