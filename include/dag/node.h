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
enum DeviceType { CPU, GPU };

class NodeBase {
  public:
    explicit NodeBase(int uid_)
        : mUID(uid_) {}

    virtual ~NodeBase() = default;
    int mUID;
};

class DataNode : public NodeBase {
  public:
    DataNode(const int uid, const Shape &shape, bool require_grad)
        : NodeBase(uid)
        , mTensor(std::make_shared<Tensor>(shape))
        , mGradient(std::make_shared<Tensor>(shape))
        , mRequireGrad(require_grad) {
        LOG_DEBUG("tensor size %lu", mTensor->size());
        LOG_DEBUG("gradient size %lu", mGradient->size());
    }
    virtual ~DataNode() {}

    Tensor *tensor() const { return mTensor.get(); }
    Tensor *grad() const { return mGradient.get(); }
    bool    requires_grad() const { return mRequireGrad; }

  protected:
    std::shared_ptr<Tensor> mTensor;
    std::shared_ptr<Tensor> mGradient;
    bool                    mRequireGrad;
};

class OperatorNodeBase : public NodeBase {
  public:
    OperatorNodeBase(const int uid)
        : NodeBase(uid)
        , mTrainFlag(true)
        , mDeviceType(DeviceType::CPU) {}
    virtual ~OperatorNodeBase() = default;

    virtual void forward(const std::vector<const Tensor *> &inps,
                         Tensor *                           outs) = 0;
    virtual void backward(const Tensor *diff, std::vector<Tensor *> &grads) = 0;
    virtual void gpuForward(const std::vector<const Tensor *> &inps,
                            Tensor *                           outs) {
        forward(inps, outs);
    }
    virtual void gpuBackward(const Tensor *diff, std::vector<Tensor *> &grads) {
        backward(diff, grads);
    }
    virtual Shape inferenceShape(const std::vector<const Tensor *> &inps) = 0;

    void applyForward() {
        std::vector<const Tensor *> forward_inps(mInNodes.size());
        std::transform(mInNodes.begin(), mInNodes.end(), forward_inps.begin(),
                       [](const DataNode *n) { return n->tensor(); });
        if (mDeviceType == DeviceType::CPU) {
            forward(forward_inps, mOutNodes->tensor());
        } else if (mDeviceType == DeviceType::GPU) {
            gpuForward(forward_inps, mOutNodes->tensor());
        }else{
            LOG_ERROR("dont support device type %d", mDeviceType);
        }
    }

    void applyBackward() {
        std::vector<Tensor *> backward_outputs(mInNodes.size());
        std::transform(mInNodes.begin(), mInNodes.end(),
                       backward_outputs.begin(),
                       [](DataNode *n) { return n->grad(); });
        if(mDeviceType == DeviceType::CPU){
            backward(mOutNodes->grad(), backward_outputs);
        }else if(mDeviceType == DeviceType::GPU){
            gpuBackward(mOutNodes->grad(), backward_outputs);
        }
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

    void cpu() { mDeviceType = DeviceType::CPU; }

    void cuda() { mDeviceType = DeviceType::GPU; }

  protected:
    std::vector<DataNode *> mInNodes;
    DataNode *              mOutNodes;
    bool                    mTrainFlag;
    DeviceType              mDeviceType;
};

} // namespace dl
