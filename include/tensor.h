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

    class Tensor {
        Shape _shape;
        double *_data;

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

        inline int _getIdxOffset(const Shape &idx) {
            int offset = 0;
            for (int i = 0; i < _shape.size() - 1; ++i) {
                offset += idx[i] * _shape[i];
            }
            return offset + idx.back();
        }

        inline void setValue(const Shape &idx, int value) {
            for (int i = 0; i < idx.size(); ++i) {
                assert(idx[i] < _shape[i]);
            }
            int offest = _getIdxOffset(idx);
            _data[offest] = value;
        }

        inline void ones(){
            for(int i = 0; i < getSize(); ++i){
                _data[i] = 1;
            }
        }

        inline void uniform(double low = 0, double high = 1){ // 均值分布
            for(int i = 0; i < getSize(); ++i){
                _data[i] = lr::uniform(low, high);
            }
        }

        inline void gussian(double mean = 0, double var = 1){
            for(int i = 0; i < getSize(); ++i){
                _data[i] = lr::gussian(mean, var);
            }
        }

        inline void zeros(){
            memset(reinterpret_cast<void*>(_data), 0, sizeof(double) * getSize());
        }

        inline void display(){
            printf("size is %d\n", getSize());
            for(int i = 0; i < getSize(); ++i){
                printf("%f\t", _data[i]);
            }
            printf("\n");
        }

        Tensor operator+(const Tensor &other); // 加法
        Tensor operator+(const int scale);

        Tensor operator-(const Tensor &other); // 减法
        Tensor operator-(const int scale);

        Tensor operator*(const Tensor &other); // 矩阵乘法
        Tensor operator*(const int scale);

        Tensor operator&(const Tensor &other); // 逐元素乘法

    };


}

#endif //LIGHTLR_TENSOR_H
