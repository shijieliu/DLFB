/*
 * @Author: liushijie
 * @Date: 2020-07-01 18:05:00
 * @LastEditTime: 2020-07-11 15:30:54
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dist/dist_optim.h
 */

#pragma once
#include "dag/graph.h"
#include "dag/optim.h"
#include "dist/comm.h"
#include <cstdlib>

namespace dl {
namespace dist {
class DistributedOptimizer {
  public:
    DistributedOptimizer(std::unique_ptr<Optimizer>& optim);

    void step();
    void zeroGrad() { mOptimizer->zeroGrad(); }

  private:
    void                  init();
    std::unique_ptr<Optimizer>           mOptimizer;
    std::unique_ptr<Comm> mComm;
};

DistributedOptimizer::DistributedOptimizer(std::unique_ptr<Optimizer>& optim)
    : mOptimizer(std::move(optim))
    , mComm(new Comm()) {}

void DistributedOptimizer::step() {
    Graph &graph = Graph::GetInstance();
    // all reduce gradient

    AllReduce(graph.params(), mComm.get());
    // update
    mOptimizer->step();
}
}
}
