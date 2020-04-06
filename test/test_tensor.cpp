//
// Created by 刘仕杰 on 2020/3/8.
//

#include <vector>
#include "tensor.h"
#include <iostream>

using lr::Tensor;
using lr::Shape;
using std::cout;
using std::endl;

void test_add(const Tensor &lhs, const Tensor &rhs) {
    Tensor res = lhs + rhs;
    cout << "test add" << endl;
    lhs.display();
    rhs.display();
    res.display();
}


void test_sub(const Tensor &lhs, const Tensor &rhs) {
    Tensor res = lhs - rhs;
    cout << "test sub" << endl;
    lhs.display();
    rhs.display();
    res.display();
}

void test_mul(const Tensor &lhs, const Tensor &rhs) {
    Tensor res = lhs * rhs;
    cout << "test mul" << endl;
    lhs.display();
    rhs.display();
    res.display();
}

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

    Tensor other{shape};
    cout << "init other" << endl;
    other.gussian();
    test_add(tensor, other);
    test_sub(tensor, other);
    test_mul(tensor, other);
}
