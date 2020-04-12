//
// Created by 刘仕杰 on 2020/4/6.
//

#ifndef LIGHTLR_ADD_H
#define LIGHTLR_ADD_H

#include "node.h"

namespace lr {
    class Add : public OperatorNode {
    public:
        Add(std::string name) : OperatorNode(name) {}

        void Forward() override;

        void Backward() override;

        Shape GetOutShape() override;

        virtual ~Add(){
            printf("[Add deconstruct]\n");
        }

    };
}
#endif //LIGHTLR_ADD_H
