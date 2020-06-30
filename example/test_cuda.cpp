/*
 * @Author: liushijie
 * @Date: 2020-06-28 15:25:14
 * @LastEditTime: 2020-06-28 15:26:22
 * @LastEditors: liushijie
 * @Description: 
 * @FilePath: /LightLR/example/test_cuda.cpp
 */ 
#include "cuda/cudalib.h"
#include "macro.h"
#include <algorithm>
#include <vector>
using std::vector;

int main() {
    int           n = 10;
    vector<float> data1;
    data1.reserve(n);
    vector<float> data2;
    data2.reserve(n);

    for (int i = 0; i < n; ++i) {
        data1[i] = i;
        data2[i] = 1;
    }

    vector<float> res;
    res.reserve(n);

    dl::cuda::CudaAdd(data1.data(), data2.data(), res.data(), res.size());
    for (int i = 0; i < n; ++i) {
        CHECK_EQ(res[i], i + 1);
    }
    LOG_INFO("cuda add pass");
}