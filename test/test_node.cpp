//
// Created by 刘仕杰 on 2020/4/6.
//

#include "tensor.h"
#include "node.h"
#include "graph.h"
#include "operator/dataprovider.h"
#include "operator/parameter.h"
#include "operator/add.h"

int main() {
    lr::DataProvider a1;
    lr::DataProvider a2;
    lr::Parameter out = lr::Add(a1, a2);
}