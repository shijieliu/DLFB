/*
 * @Author: your name
 * @Date: 2020-06-20 07:05:48
 * @LastEditTime: 2020-06-30 17:12:21
 * @LastEditors: liushijie
 * @Description: In User Settings Edit
 * @FilePath: /LightLR/include/dag/operator/add.h
 */
//
// Created by 刘仕杰 on 2020/4/6.
//
#pragma once
#include "dag/node.h"

namespace dl {

class AddImpl final : public OperatorNodeBase {
  public:
    AddImpl(int uid)
        : OperatorNodeBase(uid) {}
    virtual ~AddImpl() = default;

    void forward(const std::vector<const Tensor *> &inps,
                 Tensor *                           outs) override {
        for (const Tensor *t : inps) {
            if(t->shape() == outs->shape()){
                Add(*outs, *t, outs);
            }else if(outs->shape().size() == 4 && t->shape().size() == 1 && t->shape()[0] == outs->shape()[1]){
                // t (c)
                // outs (n, c, h, w)
                Tensor expand_t(outs->shape());
                for(int n = 0; n < outs->shape()[0]; ++n){
                    for(int c = 0; c < outs->shape()[1]; ++c){
                        for(int h = 0; h < outs->shape()[2]; ++h){
                            for(int w = 0; w < outs->shape()[3]; ++w){
                                expand_t.data()[Expand(w, outs->shape()[3], h, outs->shape()[2], c, outs->shape()[1], n)] = t->data()[c];
                            }
                        }
                    }
                }
                Add(*outs, expand_t, outs);
            }else{
                LOG_ERROR("do not support data shape %s + %s", FormatShape(outs->shape()).c_str(), FormatShape(t->shape()).c_str());
            }
        }
    }

    void backward(const Tensor *diff, std::vector<Tensor *> &grads) override {
        for (Tensor *grad : grads) {
            if(grad->shape() == diff->shape()){
                (*grad) = (*diff);
            }else{
                
            }
        }
    }

    Shape inferenceShape(const std::vector<const Tensor *> &inps) override {
        return inps[0]->shape();
    }
};
} // namespace dl
