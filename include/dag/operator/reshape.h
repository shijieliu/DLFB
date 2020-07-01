/*
 * @Author: liushijie
 * @Date: 2020-06-29 14:42:18
 * @LastEditTime: 2020-07-01 15:07:16
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dag/operator/reshape.h
 */

#pragma once

#include "dag/node.h"

namespace dl {
class ReshapeImpl : public OperatorNodeBase {
  public:
    ReshapeImpl(int uid, const Shape &shape)
        : OperatorNodeBase(uid)
        , mOutShape(shape) {}
    virtual ~ReshapeImpl() = default;

    void forward(const std::vector<const Tensor *> &inps,
                 Tensor *                           outs) override {
        CHECK_EQ(inps.size(), 1);
        const Tensor *inp = inps[0];
        CHECK_EQ(inp->size(), outs->size());
        outs->copyFrom(inp->data(), inp->data() + inp->size());
    }

    void backward(const Tensor *diff, std::vector<Tensor *> &grads) override {
        grads[0]->copyFrom(diff->data(), diff->data() + diff->size());
    }

    Shape inferenceShape(const std::vector<const Tensor *> &inps) override {
        return mOutShape;
    }

    Shape mOutShape;
};
}