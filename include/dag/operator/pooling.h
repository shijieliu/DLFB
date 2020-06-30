/*
 * @Author: liushijie
 * @Date: 2020-06-22 10:53:08
 * @LastEditTime: 2020-06-30 19:48:37
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/operator/pooling.h
 */
#pragma once
#include "dag/node.h"

namespace dl {

class PoolImpl : public OperatorNodeBase {
  public:
    PoolImpl(int uid, int kernel_size, int stride, int padding = 0)
        : OperatorNodeBase(uid)
        , mKernel(kernel_size)
        , mStride(stride)
        , mPadding(padding) {}
    virtual ~PoolImpl() = default;

    Shape inferenceShape(const std::vector<const Tensor *> &inps) override {
        CHECK_EQ(inps.size(), 1);
        const Tensor *x = inps[0]; // (n, c_in, h, w)
        return Shape(
            {x->shape()[0], x->shape()[1],
             (x->shape()[2] + 2 * mPadding - mKernel) / mStride + 1,
             (x->shape()[3] + 2 * mPadding - mKernel) / mStride + 1});
    }

    int mKernel;
    int mStride;
    int mPadding;
};

class MaxPool2DImpl : public PoolImpl {
  public:
    MaxPool2DImpl(int uid, int kernel_size, int stride,
                  int padding = 0)
        : PoolImpl(uid, kernel_size, stride, padding) {}
    virtual ~MaxPool2DImpl() = default;
    
    void forward(const std::vector<const Tensor *> &inps,
                 Tensor *                           outs) override {
        CHECK_EQ(inps.size(), 1);
        const Tensor *inp = inps[0];
        const Shape& inp_shape = inp->shape();
        int out_height = (inp_shape[2] + 2 * mPadding - mKernel) / mStride + 1;
        int out_width = (inp_shape[3] + 2 * mPadding - mKernel) / mStride + 1;

        mArgmaxIndex.clear();
        for(int n = 0; n < inp_shape[0]; ++n){
            for(int c = 0; c < inp_shape[1]; ++c){
                for(int h = 0; h < out_height; ++h){
                    for(int w = 0; w < out_width; ++w){
                        int argmax_idx = 0;
                        float max_num = 1.175494e-38;
                        for(int k1 = 0; k1 < mKernel; ++k1){
                            for(int k2 = 0; k2 < mKernel; ++k2){
                                int inp_offset = Expand(k2 + w * mStride, inp_shape[3], k1 + h * mStride, inp_shape[2], c, inp_shape[1], n);
                                if(max_num < inp->data()[inp_offset]){
                                    max_num = inp->data()[inp_offset];
                                    argmax_idx = inp_offset;
                                }
                            }
                        }

                        int out_offset = Expand(w, out_width, h, out_height, c, inp_shape[1], n);
                        outs->data()[out_offset] = max_num;
                        mArgmaxIndex[argmax_idx] = out_offset;
                    }
                }
            }
        }
    }

    void backward(const Tensor *diff, std::vector<Tensor *> &grads) override {
        Tensor* grad = grads[0];
        for(auto it : mArgmaxIndex){
            int argmax_idx = it.first;
            int diff_idx = it.second;
            grad->data()[argmax_idx] = diff->data()[diff_idx];
        }
    }

    std::unordered_map<int, int> mArgmaxIndex;
};

} // namespace dl
