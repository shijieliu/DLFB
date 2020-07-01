/*
 * @Author: liushijie
 * @Date: 2020-07-01 18:05:00
 * @LastEditTime: 2020-07-01 18:09:36
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
    DistributedOptimizer(Optimizer *optim)
        : mOptimizer(optim) {}
    void step() {
        Graph &graph = Graph::GetInstance();
        // all reduce gradient
        
        // update
        mOptimizer->step();
    }
    void zeroGrad() {
      mOptimizer->zeroGrad();
    }
  private:
    int     mRank;
    int     mWorldSize;
    Optimizer *mOptimizer;
};
}
}
