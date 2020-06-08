//
// Created by 刘仕杰 on 2020/2/2.
//

#ifndef LIGHTLR_OPTIM_H
#define LIGHTLR_OPTIM_H

#include <vector>
#include "tensor.h"

namespace dl {
    class Node;
    class Optimizer {
    public:
        Optimizer();
        void step();
        void zeroGrad(Node* endpoint);
    };
}

#endif //LIGHTLR_OPTIM_H
