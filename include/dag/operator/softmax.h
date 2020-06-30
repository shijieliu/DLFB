/*
 * @Author: liushijie
 * @Date: 2020-06-25 22:38:55
 * @LastEditTime: 2020-06-30 17:19:53
 * @LastEditors: liushijie
 * @Description: 
 * @FilePath: /LightLR/include/dag/operator/softmax.h
 */ 

#pragma once

#include "dag/node.h"

namespace dl {
class SoftmaxImpl : public OperatorNodeBase {
  public:
    SoftmaxImpl(int uid)
        : OperatorNodeBase(uid) {}

    ~SoftmaxImpl() = default;

    void forward(const std::vector<const Tensor *> &inps,
                 Tensor *                           out) override {
        if(inps.size() != 1){
            LOG_ERROR("invalid input size %lu", inps.size());
        }
        const Tensor *inp = inps[0];
        int row = inp->shape()[0];
        int col = inp->shape()[1];
        Tensor        sum_tensor(inp->shape());
        if(inp->shape().size() != 2){
            LOG_ERROR("invalid input shape %s", FormatShape(inp->shape()).c_str());
        }
        
        for(int n = 0; n < inp->shape()[0]; ++n){
            float * inp_st = inp->data() + n * col;
            float *inp_ed = inp->data() + (n + 1) * col;
            float row_max_value = *std::max_element(inp_st, inp_ed);
            std::transform(inp_st, inp_ed, out->data() + n * col, [row_max_value](float v) -> float {return std::exp(v - row_max_value);});
            float sum_value =
                std::accumulate(out->data() + n * col,
                                out->data() + (n + 1) * col, 0);
            std::transform(out->data() + n * col, out->data() + (n + 1) * col, out->data() + n * col, [sum_value](float v) -> float {return v/sum_value;});
        }

        mForwardLogits = *out;
    }

    void backward(const Tensor *diff, std::vector<Tensor *> &grads) override {
        Tensor *grad = grads[0];
        Tensor dot;
    }
    Shape inferenceShape(const std::vector<const Tensor *> &inps) override {
        return inps[0]->shape();
    }

    Tensor mForwardLogits;
    Tensor mLabel;
};
}