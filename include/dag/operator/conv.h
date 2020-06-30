/*
 * @Author: liushijie
 * @Date: 2020-06-15 21:14:52
 * @Last Modified by: liushijie
 * @Last Modified time: 2020-06-15 21:17:00
 */

#pragma once

#include "dag/node.h"

namespace dl {
class Conv2DImpl : public OperatorNodeBase {
  public:
    Conv2DImpl(const int uid, int kernel_size, int stride,
               int padding = 0, const char *paddingmode = "zeros")
        : OperatorNodeBase(uid)
        , mKernel(kernel_size)
        , mStride(stride)
        , mPadding(padding)
        , mPaddingMode(paddingmode) {}
    virtual ~Conv2DImpl() = default;

    void forward(const std::vector<const Tensor *> &inps,
                 Tensor *                           outs) override {
        CHECK_EQ(inps.size(), 2);
        Conv2D(*inps[0], *inps[1], outs, mStride, mPadding, mPaddingMode);
        minp  = *inps[0];
        mWeight = *inps[1];
    }

    void backward(const Tensor *diff, std::vector<Tensor *> &grads) override {
        CHECK_EQ(grads.size(), 2);
        Tensor *grad_x = grads[0];
        Tensor *grad_w = grads[1];
        Conv2D(minp, *diff, grad_w, mStride, mPadding, mPaddingMode);
        Tensor padding_and_rotate_weight({mWeight.shape()[0],
                                          mWeight.shape()[1],
                                          mWeight.shape()[2] + 2 * mPadding,
                                          mWeight.shape()[3] + 2 * mPadding});
        Padding(mWeight, &padding_and_rotate_weight, mPadding,
                mPaddingMode.c_str());
        Rotate(mWeight, &padding_and_rotate_weight);
        Conv2D(mWeight, *diff, grad_x, mStride, mKernel - 1, "zeros");
    }

    Shape inferenceShape(const std::vector<const Tensor *> &inps) override {
        const Tensor *x = inps[0]; // (n, c_in, h, w)
        const Tensor *w = inps[1]; // (c_out, c_in, k, k)
        LOG_DEBUG("\n\tx shape:%s\n\tw shape:%s", FormatShape(x->shape()).c_str(), FormatShape(w->shape()).c_str());
        return Shape(
            {x->shape()[0], w->shape()[0],
             (x->shape()[2] + 2 * mPadding - w->shape()[2]) / mStride + 1,
             (x->shape()[3] + 2 * mPadding - w->shape()[3]) / mStride + 1});
    }

    int      mKernel;
    int      mStride;
    int      mPadding;
    std::string mPaddingMode;
    Tensor      minp;
    Tensor      mWeight;
};
}