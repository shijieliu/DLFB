//
// Created by 刘仕杰 on 2020/2/2.
//

#include "tensor.h"

namespace lr {
    Tensor::Tensor(const Shape &shape) : _shape(shape) {
        int size = getSize();
        _data = new double[size];
    }

    Tensor Tensor::element_op(const Tensor &other, Operator op) const {
        if (!std::equal(this->shape().begin(), this->shape().end(), other.shape().begin())) {
            throw std::out_of_range("element operator shape not equal");
        }
        auto element_op_func = [op](double l, double r) -> double {
            switch (op) {
                case Add:
                    return l + r;
                case Sub:
                    return l - r;
                case Mul:
                    return l * r;
            }

        };
        Tensor res{this->shape()};
        for (int i = 0; i < this->getSize(); ++i) {
            res.setValue(i, element_op_func(this->getValue(i), other.getValue(i)));
        }
        return res;
    }

    Tensor Tensor::element_op(const int scale, Operator op) const{
        auto element_op_func = [op](double l, double r) -> int {
            switch (op) {
                case Add:
                    return l + r;
                case Sub:
                    return l - r;
                case Mul:
                    return l * r;
            }

        };
        Tensor res{this->shape()};
        for (int i = 0; i < this->getSize(); ++i) {
            res.setValue(i, element_op_func(this->getValue(i), scale));
        }
        return res;
    }

    Tensor Tensor::operator+(const Tensor &other) const {
        return element_op(other, Add);
    }

    Tensor Tensor::operator+(const int scale) const {
        return element_op(scale, Add);
    }

    Tensor Tensor::operator-(const Tensor &other) const {
        return element_op(other, Sub);
    }

    Tensor Tensor::operator-(const int scale) const {
        return element_op(scale, Sub);
    }

    Tensor Tensor::operator*(const Tensor &other) const{
        return element_op(other, Mul);
    }

    Tensor Tensor::operator*(const int scale) const{
        return element_op(scale, Mul);
    }

//    Tensor Tensor::operator&(const Tensor &other) {
//
//    }

    Tensor::~Tensor() {
        delete[] _data;
    }
}