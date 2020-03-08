//
// Created by 刘仕杰 on 2020/1/25.
//

#ifndef LIGHTLR_UTILS_H
#define LIGHTLR_UTILS_H

#include "tensor.h"
#include <string>
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>

namespace lr {




    template<typename T>
    T CalcLoss(Tensor<T> *predict, const Tensor<T> &label, Loss<T>* lossFunc) {
        auto loss = lossFunc->Forward(*predict, label);
        const auto& lossData = loss.Data();
        T value = 0;
        for (const auto& r: lossData){
            for (const auto c: r){
                value += c;
            }
        }
        return value;
    }

    template<typename T>
    void CalcGradient(const Tensor<T> &predict, const Tensor<T> &input, Tensor<T> *weights) {
        assert(predict.Row() == input.Row());
        for (int r = 0; r < weights->Row(); ++r) {
            for (int c = 0; c < weights->Col(); ++c) {
                T gradient = 0;
                for (int b = 0; b < predict.Row(); ++b) { // batchsize
                    gradient += input.Data()[b][r] * predict.Gradient()[b][c];
                }
                weights->SetGradient(r, c, gradient);
            }
        }
    }
}

#endif //LIGHTLR_UTILS_H
