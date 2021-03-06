/*
 * @Author: your name
 * @Date: 2020-02-20 07:06:22
 * @LastEditTime: 2020-07-01 18:16:30
 * @LastEditors: liushijie
 * @Description: In User Settings Edit
 * @FilePath: /LightLR/include/dag/optim.h
 */
#pragma once
#include "dag/node.h"
#include "random.h"

namespace dl {
class Optimizer {
  public:
    Optimizer(const std::vector<DataNode *>& params_list);
    virtual ~Optimizer() = default;
    virtual void step() = 0;
    void zeroGrad();

  protected:
    std::vector<DataNode *> mParams;
};

Optimizer::Optimizer(const std::vector<DataNode *>& params_list)
    : mParams(params_list.begin(), params_list.end()) {}

void Optimizer::zeroGrad(){
  for(DataNode * param : mParams){
    Zeros(param->grad()->data(), param->grad()->size());
  }
}

class SGD: public Optimizer{
  public:
    SGD(const std::vector<DataNode *>& params_list, float lr, float momentum, float weight_decay, bool nesterov=false);
    virtual ~SGD() = default;

    virtual void step() override;
  protected:
    float mLr;
    float mMomentum;
    float mWeightDecay;
    bool mNesterov;
};

SGD::SGD(const std::vector<DataNode *>& params_list, float lr, float momentum, float weight_decay, bool nesterov) : Optimizer(params_list), mLr(lr), mMomentum(momentum), mWeightDecay(weight_decay), mNesterov(nesterov) {}

void SGD::step(){
  for(DataNode *node: mParams){
    if(node->requires_grad()){
      Tensor update(node->tensor()->shape());
      Mul(*(node->grad()), mLr, &update);
      Add(*(node->tensor()), update, node->tensor());
    }
  }
}

}
