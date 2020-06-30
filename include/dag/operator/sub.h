/*
 * @Author: liushijie
 * @Date: 2020-06-15 20:57:53
 * @Last Modified by: liushijie
 * @Last Modified time: 2020-06-15 21:04:13
 */

#pragma once
#include "dag/node.h"

namespace dl {

class SubImpl : public OperatorNodeBase {
    SubImpl(const int uid);
    virtual ~SubImpl();

    void forward(const std::vector<const Tensor *> &inps, Tensor *outs) {
        Sub(*inps[0], *inps[1], outs);
    }

    void backward(const Tensor *diff, std::vector<Tensor *> &grads) {
        (*grads[0]) = *diff;
        (*grads[1]) = *diff;
        std::transform(
            grads[1]->data(), grads[1]->data() + grads[1]->size(),
            grads[1]->data(), [](float x) -> float { return -x; });
    }
    
    Shape inferenceShape(const std::vector<const Tensor *> &inps) {
        Shape res;
        for (const Tensor *t : inps) {
            if (res.size() == 0) {
                res = t->shape();
            }
            assert(res == t->shape());
        }
        return res;
    }
};
}