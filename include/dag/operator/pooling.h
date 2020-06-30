/*
 * @Author: liushijie
 * @Date: 2020-06-22 10:53:08
 * @LastEditTime: 2020-06-29 14:38:27
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/operator/pooling.h
 */
#pragma once
#include "dag/node.h"

namespace dl {

class PoolImpl : public OperatorNodeBase {
  public:
    PoolImpl(int64_t uid, int64_t kernel_size, int64_t stride, int64_t padding = 0)
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

    int64_t mKernel;
    int64_t mStride;
    int64_t mPadding;
};

class MaxPool2DImpl : public PoolImpl {
  public:
    MaxPool2DImpl(int64_t uid, int64_t kernel_size, int64_t stride,
                  int64_t padding = 0)
        : PoolImpl(uid, kernel_size, stride, padding) {}
    virtual ~MaxPool2DImpl() = default;
    
    void forward(const std::vector<const Tensor *> &inps,
                 Tensor *                           outs) override {
        CHECK_EQ(inps.size(), 1);
        const Tensor *inp = inps[0];
        const Shape& inp_shape = inp->shape();
        int64_t out_height = (inp_shape[2] + 2 * mPadding - mKernel) / mStride + 1;
        int64_t out_width = (inp_shape[3] + 2 * mPadding - mKernel) / mStride + 1;

        mArgmaxIndex.clear();
        for(int64_t n = 0; n < inp_shape[0]; ++n){
            for(int64_t c = 0; c < inp_shape[1]; ++c){
                for(int64_t h = 0, out_h = 0; h + mKernel < inp_shape[2]; h += mStride, ++out_h){
                    for(int64_t w = 0, out_w = 0; w + mKernel < inp_shape[3]; w += mStride, ++out_w){
                        int64_t argmax_idx = 0;
                        float max_num = 1.175494e-38;
                        for(int64_t k1 = 0; k1 < mKernel; ++k1){
                            for(int64_t k2 = 0; k2 < mKernel; ++k2){
                                int64_t inp_offset = expand(k2 + w, inp_shape[3], k1 + h, inp_shape[2], c, inp_shape[1], n);
                                if(max_num < inp->data()[inp_offset]){
                                    max_num = inp->data()[inp_offset];
                                    argmax_idx = inp_offset;
                                }
                            }
                        }

                        int64_t out_offset = expand(out_w, out_width, out_h, out_height, c, inp_shape[1], n);
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
            int64_t argmax_idx = it.first;
            int64_t diff_idx = it.second;
            grad->data()[argmax_idx] = diff->data()[diff_idx];
        }
    }

    std::unordered_map<int64_t, int64_t> mArgmaxIndex;
};

} // namespace dl
