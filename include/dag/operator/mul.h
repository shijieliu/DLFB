/*
 * @Author: liushijie
 * @Date: 2020-06-20 17:31:18
 * @LastEditTime: 2020-06-29 17:34:34
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/operator/mul.h
 */
#pragma once
#include "dag/node.h"

namespace dl {

class MatMulImpl final : public OperatorNodeBase {
  public:
    MatMulImpl(int64_t uid)
        : OperatorNodeBase(uid) {}
    virtual ~MatMulImpl() = default;

    void transpose(Tensor *inp){
        CHECK_EQ(inp->shape().size(), 2);
        Tensor cpy = *inp;
        for(int64_t n = 0; n < inp->shape()[0]; ++n){
            for(int64_t m = 0; m < inp->shape()[1]; ++m){
                inp->data()[n, inp->shape()[0], m] = cpy.data()[expand(m, inp->shape()[1], n)];
            }
        }
    }

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

        Mat(*x, *w, outs);
        m_transpose_w = *w;
        transpose(&m_transpose_w);
        m_transpose_x = *x;
        transpose(&m_transpose_x);
    }

    void backward(const Tensor *diff, std::vector<Tensor *> &grads) override {
        // diff (n, dim) grad_x (n, m) grad_w (m, dim)
        Tensor *grad_x = grads[0];
        Tensor *grad_w = grads[1];
        Mat(m_transpose_x, *diff, grad_w);
        Mat(*diff, m_transpose_w, grad_x);
    }

    Shape inferenceShape(const std::vector<const Tensor *> &inps) override {
        return Shape({inps[0]->shape()[0], inps[1]->shape()[1]});
    }

    Tensor m_transpose_x;
    Tensor m_transpose_w;
};

class MulImpl final : public OperatorNodeBase {
  public:
    MulImpl(int64_t uid)
        : OperatorNodeBase(uid) {}
    virtual ~MulImpl() = default;

    void forward(const std::vector<const Tensor *> &inps,
                 Tensor *                           outs) override {
        for (const Tensor *t : inps) {
            Mul(*outs, *t, outs);
        }
        mMulActivation.clear();
        for (const Tensor *t : inps) {
            mMulActivation.emplace_back(t->shape());
            Div(*outs, *t, &mMulActivation.back());
        }
    }
    void backward(const Tensor *diff, std::vector<Tensor *> &grads) override {
        for (int64_t i = 0; i < grads.size(); ++i)
            Mul(*diff, mMulActivation[i], grads[i]);
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
