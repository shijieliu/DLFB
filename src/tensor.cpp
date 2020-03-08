//
// Created by 刘仕杰 on 2020/2/2.
//

#include "tensor.h"

namespace lr {
    Tensor::Tensor(const Shape &shape) : _shape(shape) {
        int size = getSize();
        _data = new double[size];
    }

//    Tensor Tensor::operator+(const Tensor &other) {
//        // impletement tensor + tensor
//
//        this->_data
//    }

    Tensor::~Tensor() {
        delete[] _data;
    }
}