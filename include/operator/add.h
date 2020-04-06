//
// Created by 刘仕杰 on 2020/4/6.
//

#ifndef LIGHTLR_ADD_H
#define LIGHTLR_ADD_H

#include "operator/operator.h"

namespace lr {
    class AddNode : public OperatorNode {
        AddNode();

        void Forward();

        void Backward();
    };
}
#endif //LIGHTLR_ADD_H
