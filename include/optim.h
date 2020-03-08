//
// Created by 刘仕杰 on 2020/2/2.
//

#ifndef LIGHTLR_OPTIM_H
#define LIGHTLR_OPTIM_H

#include <vector>
#include "tensor.h"

namespace lr {
    template<typename T>
    class Optimizer {
    public:
        Optimizer();

        void step();

    };
}

#endif //LIGHTLR_OPTIM_H
