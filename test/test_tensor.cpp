//
// Created by 刘仕杰 on 2020/3/8.
//

#include <vector>
#include "tensor.h"

using lr::Tensor;
using lr::Shape;

int main() {
    Shape shape{3, 2};
    Tensor tensor{shape};
    tensor.ones();
    tensor.display();
    tensor.zeros();
    tensor.display();
    tensor.uniform();
    tensor.display();
    tensor.gussian();
    tensor.display();
}
