//
// Created by 刘仕杰 on 2020/4/6.
//

#ifndef ADD_H
#define ADD_H

#include "node.h"

namespace dl {
    class Add : public OperatorNode {
    public:
        Add(std::string name) : OperatorNode(name) {}

        void Forward() override;

        void Backward() override;

        Shape GetOutShape() override;

        virtual ~Add() {
            printf("[Add deconstruct]\n");
        }

    };

    class Sub : public OperatorNode {
    public:
        Sub(std::string name) : OperatorNode(name) {}

        void Forward() override;

        void Backward() override;

        Shape GetOutShape() override;

        virtual ~Sub() {
            printf("[Sub deconstruct]\n");
        }
    };
}
#endif //LIGHTLR_ADD_H
