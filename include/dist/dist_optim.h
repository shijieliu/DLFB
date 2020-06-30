
#pragma once
#include "dag/graph.h"
#include "dag/optim.h"
#include "dist/comm.h"
#include <cstdlib>

namespace dl {
namespace dist {
class DistributedOptimizer : public Optimizer {
  public:
    DistributedOptimizer(Optimizer *optim)
        : mOprimizer(optim) {}
    void step() override {
        Graph &graph = Graph::GetInstance();
        // all reduce gradient

        // update
        mOprimizer->step();
    }

  private:
    int64_t     mRank;
    int64_t     mWorldSize;
    Optimizer *mOprimizer;
};
}
}
