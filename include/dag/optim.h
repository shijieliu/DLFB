/*
 * @Author: your name
 * @Date: 2020-06-20 07:06:22
 * @LastEditTime: 2020-06-20 07:06:23
 * @LastEditors: your name
 * @Description: In User Settings Edit
 * @FilePath: /LightLR/include/dag/optim.h
 */
//
// Created by 刘仕杰 on 2020/2/2.
//
#pragma once
#include "tensor.h"
#include <vector>

namespace dl {
class Node;
class Optimizer {
  public:
    Optimizer();
    virtual void step();
    virtual void zeroGrad(Node *endpoint);
};
}
