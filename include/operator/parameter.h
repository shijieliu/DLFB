//
// Created by 刘仕杰 on 2020/4/6.
//

#ifndef LIGHTLR_PARAMETER_H
#define LIGHTLR_PARAMETER_H

#include "node.h"

namespace lr {
    class Parameter : public Node {
    private:
        Tensor m_data;
        Tensor m_gradient;
        bool m_requires_grad;
    };
}
#endif //LIGHTLR_PARAMETER_H
