/*
 * @Author: liushijie
 * @Date: 2020-06-20 17:31:18
 * @LastEditTime: 2020-07-09 20:33:57
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/operator/mul.h
 */
#pragma once
#include "dag/node.h"
#include "cuda/cuda_mul.h"
namespace dl {

class MatMulImpl final : public OperatorNodeBase {
  public:
    MatMulImpl(int uid)
        : OperatorNodeBase(uid) {}
    virtual ~MatMulImpl() = default;

    void forward(const std::vector<const Tensor *> &inps,
                 Tensor *                           outs) override {
        if(inps.size() != 2){
            LOG_ERROR("invalid input size");
        }
        const Tensor *x = inps[0];
        const Tensor *w = inps[1];
        if(x->shape().size() != 2){
            LOG_ERROR("invalid x shape %s", FormatShape(x->shape()).c_str());
        }
        if(w->shape().size() != 2){
            LOG_ERROR("invalid w shape %s", FormatShape(w->shape()).c_str());
        }
        if(mDeviceType == DeviceType::GPU){
            cuda::CudaMat(*x, *w, outs);
        }else{
            Mat(*x, *w, outs);
        }
        m_transpose_w.reshape({w->shape()[1], w->shape()[0]});
        Transpose(*w, &m_transpose_w);
        m_transpose_x.reshape({x->shape()[1], x->shape()[0]});
        Transpose(*x, &m_transpose_x);
    }

    void backward(const Tensor *diff, std::vector<Tensor *> &grads) override {
        // diff (n, dim) grad_x (n, m) grad_w (m, dim) m_transpose_x (m, n) m_transpose_w (dim, m)
        Tensor *grad_x = grads[0];
        Tensor *grad_w = grads[1];
        if(mDeviceType == DeviceType::GPU){
            cuda::CudaMat(m_transpose_x, *diff, grad_w);
            cuda::CudaMat(*diff, m_transpose_w, grad_x);
        }else{
            Mat(m_transpose_x, *diff, grad_w);
            Mat(*diff, m_transpose_w, grad_x);
        }
    }

    Shape inferenceShape(const std::vector<const Tensor *> &inps) override {
        return Shape({inps[0]->shape()[0], inps[1]->shape()[1]});
    }

    Tensor m_transpose_x;
    Tensor m_transpose_w;
};

class MulImpl final : public OperatorNodeBase {
  public:
    MulImpl(int uid)
        : OperatorNodeBase(uid) {}
    virtual ~MulImpl() = default;

    void forward(const std::vector<const Tensor *> &inps,
                 Tensor *                           outs) override {
        for (const Tensor *t : inps) {
            if(mDeviceType == DeviceType::GPU){
                cuda::CudaMul(*outs, *t, outs);
            }else{
                Mul(*outs, *t, outs);
            }
        }
        mMulActivation.clear();
        for (const Tensor *t : inps) {
            mMulActivation.emplace_back(t->shape());
            Div(*outs, *t, &mMulActivation.back());
        }
    }
    void backward(const Tensor *diff, std::vector<Tensor *> &grads) override {
        for (int i = 0; i < grads.size(); ++i){
            if(mDeviceType == DeviceType::GPU){
                cuda::CudaMul(*diff, mMulActivation[i], grads[i]);
            }else{
                Mul(*diff, mMulActivation[i], grads[i]);
            }
        }
    }

    Shape inferenceShape(const std::vector<const Tensor *> &inps) override {
        Shape res;
        for (const Tensor *t : inps) {
            if (res.size() == 0) {
                res = t->shape();
            }
            assert(res == t->shape());
        }
        return res;
    }
    std::vector<Tensor> mMulActivation;
};
} // namespace dl
