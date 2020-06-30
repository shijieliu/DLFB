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
    explicit Pool2D(int kernel_size, int stride, int padding)
        : mKernelSize(kernel_size)
        , mStride(stride)
        , mPadding(padding) {}

  protected:
    int mKernelSize;
    int mStride;
    int mPadding;
};

class Maxpool2d : public Pool2D {
  public:
    explicit Maxpool2d(int kernel_size, int stride, int padding)
        : Pool2D(kernel_size, stride, padding) {
        LOG_INFO("\n\tMaxpool2d "
                 "args:\n\t\tkernel_size:%d\n\t\tstride:%d\n\t\tpadding:%d",
                 kernel_size, stride, padding);
    }

    DataNode *operator()(DataNode *inp) {
        DataNode *out = CreateNode<MaxPool2DImpl, int, int, int>(
            {inp}, mKernelSize, mStride, mPadding);
        LOG_INFO("\n\tmaxpool2d input shape:%s\n\tmaxpool2d output shape:%s", FormatShape(inp->tensor()->shape()).c_str(), FormatShape(out->tensor()->shape()).c_str());
        return out;
    }
};
}
}
