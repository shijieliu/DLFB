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

    Tensor Tensor::element_op(const int scale, Operator op) const {
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
            res.setValue(i, element_op_func(this->getValue(i), scale));
        }
        return res;
    }

    Tensor Tensor::operator+(const Tensor &other) const {
        return element_op(other, Add);
    }

    Tensor Tensor::operator+(int scale) const {
        return element_op(scale, Add);
    }

    Tensor Tensor::operator-(const Tensor &other) const {
        return element_op(other, Sub);
    }

    Tensor Tensor::operator-(int scale) const {
        return element_op(scale, Sub);
    }

    Tensor Tensor::operator*(const Tensor &other) const {
        return element_op(other, Mul);
    }

    Tensor Tensor::operator*(int scale) const {
        return element_op(scale, Mul);
    }

    Tensor Tensor::matrix_mul(const Tensor &other) const {
        if(this->shape().size() != 2 || other.shape().size() != 2){
            throw std::runtime_error("matrix mul shape must be 2");
        }
        if(this->shape()[1] != other.shape()[0]){
            throw std::runtime_error("matrix mul col row must equal");
        }
        int row = this->shape()[0];
        int col = other.shape()[1];
        int common_size = this->shape()[1];
        Shape tmpshape(2, 0);
        Tensor res{Shape (row, col)};
        res.zeros();
        for(int r = 0; r < row; ++r){
            for(int c = 0; c < col; ++c){
                for(int i = 0; i < common_size; ++i){
                    tmpshape[0] = r;
                    tmpshape[1] = i;
                    double first = this->getValue(tmpshape);
                    tmpshape[0] = i;
                    tmpshape[1] = c;
                    double second = other.getValue(tmpshape);
                    tmpshape[0] = r;
                    tmpshape[1] = c;
                    int current = res.getValue(tmpshape);
                    res.setValue(tmpshape, current + first * second);
                }
            }
        }
        return res;
    }

    Tensor::~Tensor() {
        delete[] _data;
    }
}