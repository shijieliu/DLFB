//
// Created by 刘仕杰 on 2020/4/6.
//

#pragma once

#include "dag/tensor.h"
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dl {

class NodeBase {
  public:
    explicit NodeBase(int64_t uid_)
        : mUID(uid_) {}

    virtual ~NodeBase() = default;
    int64_t mUID;
};

enum DeviceType{
    CPU,
    GPU
};
class DataNode : public NodeBase {
  public:
    DataNode(const int64_t uid, const Shape &shape, bool require_grad, DeviceType device_type=DeviceType::CPU)
        : NodeBase(uid)
        , mTensor(std::make_shared<Tensor>(shape))
        , mGradient(std::make_shared<Tensor>(shape))
        , mRequireGrad(require_grad)
        , mDeviceType(device_type) {
        LOG_DEBUG("tensor size %zu", mTensor->size());
        LOG_DEBUG("gradient size %zu", mGradient->size());
    }
    virtual ~DataNode() { }

    Tensor* tensor() const { return mTensor.get();}
    Tensor* grad() const {return mGradient.get();}

  protected:
    std::shared_ptr<Tensor> mTensor;
    std::shared_ptr<Tensor> mGradient;
    bool                    mRequireGrad;
    DeviceType mDeviceType;
};

class OperatorNodeBase : public NodeBase {
  public:
    OperatorNodeBase(const int64_t uid)
        : NodeBase(uid)
        , mTrainFlag(true) {}
    virtual ~OperatorNodeBase() = default;

    virtual void forward(const std::vector<const Tensor *> &inps,
                         Tensor *                           outs) = 0;
    virtual void backward(const Tensor *diff, std::vector<Tensor *> &grads) = 0;
    virtual Shape inferenceShape(const std::vector<const Tensor *> &inps) = 0;

    void applyForward() {
        std::vector<const Tensor *> forward_inps(mInNodes.size());
        std::transform(mInNodes.begin(), mInNodes.end(), forward_inps.begin(),
                       [](const DataNode *n) { return n->tensor(); });

        forward(forward_inps, mOutNodes->tensor());
    }

    void applyBackward() {
        std::vector<Tensor *> backward_outputs(mInNodes.size());
        std::transform(mInNodes.begin(), mInNodes.end(),
                       backward_outputs.begin(),
                       [](DataNode *n) { return n->grad(); });
        backward(mOutNodes->grad(), backward_outputs);
    }

    Shape applyInferenceShape() {
        assert(mInNodes.size() > 0);
        std::vector<const Tensor *> forward_inps(mInNodes.size());
        std::transform(mInNodes.begin(), mInNodes.end(), forward_inps.begin(),
                       [](const DataNode *n) { return n->tensor(); });
        return inferenceShape(forward_inps);
    }

    void setInNodes(const std::vector<DataNode *> &prenodes) {
        mInNodes = prenodes;
    }

    template <typename ForwardIt>
    void setInNodes(ForwardIt first, ForwardIt last) {
        while (first != last) {
            mInNodes.push_back(*first);
            ++first;
        }
    }

    void setOutNodes(DataNode *nextnodes) { mOutNodes = nextnodes; }

    const DataNode *getOutNodes() { return mOutNodes; }

    void train() { mTrainFlag = true; }

    void eval() { mTrainFlag = false; }

  protected:
    std::vector<DataNode *> mInNodes;
    DataNode *              mOutNodes;
    bool                    mTrainFlag;
};

} // namespace dl
