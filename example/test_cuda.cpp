/*
 * @Author: liushijie
 * @Date: 2020-06-28 15:25:14
 * @LastEditTime: 2020-07-08 20:22:04
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/example/test_cuda.cpp
 */
#include "cuda/cuda_add.h"
#include "cuda/cuda_conv.h"
#include "macro.h"
#include "random.h"
#include <algorithm>
#include <vector>
using std::vector;
using dl::Tensor;

void test_add() {
    int        n = 1000;
    dl::Tensor data1({n});
    dl::Tensor data2({n});
    dl::Ones(data2.data(), 10);

    for (int i = 0; i < n; ++i) {
        data1.data()[i] = i;
    }
    dl::Tensor res({n});
    dl::Time("cuda add", [&](){
        dl::cuda::CudaAdd(data1, data2, &res);
    });
    for (int i = 0; i < n; ++i) {
        CHECK_NE(std::abs(res.data()[i] - i - 1), 1e-5);
    }
    LOG_INFO("cuda add pass");
}

int main() {
    test_add();
}