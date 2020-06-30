/*
 * @Author: liushijie
 * @Date: 2020-06-22 10:47:59
 * @LastEditTime: 2020-06-29 14:37:04
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/layer/pooling.h
 */
#pragma once
#include "dag/layer/layer.h"

namespace dl {
namespace nn {
class Pool2D : public LayerBase {
  public:
    explicit Pool2D(int64_t kernel_size, int64_t stride, int64_t padding)
        : mKernelSize(kernel_size)
        , mStride(stride)
        , mPadding(padding) {}

  protected:
    int64_t mKernelSize;
    int64_t mStride;
    int64_t mPadding;
};

class Maxpool2d : public Pool2D {
  public:
    explicit Maxpool2d(int64_t kernel_size, int64_t stride, int64_t padding)
        : Pool2D(kernel_size, stride, padding) {
        LOG_INFO("\n\tMaxpool2d "
                 "args:\n\t\tkernel_size:%lu\n\t\tstride:%lu\n\t\tpadding:%lu",
                 kernel_size, stride, padding);
    }

    DataNode *operator()(DataNode *inp) {
        DataNode *out = CreateNode<MaxPool2DImpl, int64_t, int64_t, int64_t>(
            {inp}, mKernelSize, mStride, mPadding);
        LOG_INFO("\n\tmaxpool2d input shape:%s\n\tmaxpool2d output shape:%s", FormatShape(inp->tensor()->shape()).c_str(), FormatShape(out->tensor()->shape()).c_str());
        return out;
    }
};
}
}
