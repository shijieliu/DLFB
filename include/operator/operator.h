//
// Created by 刘仕杰 on 2020/4/6.
//

#ifndef LIGHTLR_OPERATOR_H
#define LIGHTLR_OPERATOR_H

#include "node.h"

namespace lr {
    class OperatorNode : public Node {
    public:
        virtual void Forward();

        virtual void Backward();
    };
}
#endif //LIGHTLR_OPERATOR_H
