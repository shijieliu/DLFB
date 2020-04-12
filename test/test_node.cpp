//
// Created by 刘仕杰 on 2020/4/6.
//

#include "tensor.h"
#include "node.h"
#include "graph.h"
#include "operator/add.h"

int main() {
    lr::Parameter a1("a1", lr::Shape{3});
    lr::Parameter a2("a2", lr::Shape{3});
    lr::Add add("add");
    std::unique_ptr<lr::Parameter> out = add(&a1, &a2);

    out->Forward();
//    out->Data().display();
//    out.backward();
//    out.GradData().display();
//    optimizer.step(xx.parameters());

}