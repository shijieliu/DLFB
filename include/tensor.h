//
// Created by 刘仕杰 on 2020/1/25.
//

#ifndef LIGHTLR_TENSOR_H
#define LIGHTLR_TENSOR_H

#include <iostream>
#include <memory>
#include <vector>
#include <initializer_list>
#include <cassert>
#include <algorithm>
#include <cstring>
#include "utils/random.h"

namespace lr {
    using Shape = std::vector<int>;

    
    //
//        Tensor element_op(const Tensor &other, Operator op) const;
//
//        Tensor element_op(int scale, Operator op) const;
//
//        Tensor operator+(const Tensor &other) const; // 加法
//        Tensor operator+(int scale) const;
//
//        Tensor operator-(const Tensor &other) const; // 减法
//        Tensor operator-(int scale) const;
//
//        Tensor operator*(const Tensor &other) const; // 逐元素乘法
//        Tensor operator*(int scale) const;
//
//        Tensor matrix_mul(const Tensor &other) const;

    class Tensor {
    public:
        explicit Tensor(const Shape &shape);

        ~Tensor();

        inline int getSize() const {
            int size = 1;
            for (int i : _shape) {
                size *= i;
            }
            return size;
        }

        inline const Shape &shape() const {
            return _shape;
        }

        inline int _getIdxOffset(const Shape &idx) const {
            int offset = 0;
            for (int i = 0; i < _shape.size() - 1; ++i) {
                offset += idx[i] * _shape[i];
            }
            return offset + idx.back();
        }

        inline void setValue(const Shape &idx, double value) {
            int offset = _getIdxOffset(idx);
            setValue(offset, value);
        }

        inline void setValue(int idx, double value) {
            if (idx < 0 || idx >= this->getSize()) {
                throw std::out_of_range("set value out of range");
            }
            _data[idx] = value;
        }

        inline double getValue(const Shape &idx) const {
            int offset = _getIdxOffset(idx);
            return getValue(offset);
        }

        inline double getValue(int idx) const {
            if (idx < 0 || idx >= this->getSize()) {
                throw std::out_of_range("get value out of range");
            }
            return _data[idx];
        }

        inline void ones() {
            for (int i = 0; i < getSize(); ++i) {
                _data[i] = 1;
            }
        }

        inline void uniform(double low = 0, double high = 1) { // 均值分布
            for (int i = 0; i < getSize(); ++i) {
                _data[i] = lr::uniform(low, high);
            }
        }

        inline void gussian(double mean = 0, double var = 1) {
            for (int i = 0; i < getSize(); ++i) {
                _data[i] = lr::gussian(mean, var);
            }
        }

        inline void zeros() {
            memset(reinterpret_cast<void *>(_data), 0, sizeof(double) * getSize());
        }

        inline void display() const {
            printf("size is %d\n", getSize());
            for (int i = 0; i < getSize(); ++i) {
                printf("%f\t", _data[i]);
            }
            printf("\n");
        }

    private:
        Shape _shape;
        double *_data;
    };


}

#endif //LIGHTLR_TENSOR_H
