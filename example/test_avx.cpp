/*
 * @Author: your name
 * @Date: 2020-06-20 07:35:23
 * @LastEditTime: 2020-06-30 11:32:20
 * @LastEditors: liushijie
 * @Description: In User Settings Edit
 * @FilePath: /LightLR/example/test_avx.cpp
 */
#include "avx.h"
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

    dl::SIMD::AvxVecAdd(data1.data(), data2.data(), res.data(), n);
    for (int i = 0; i < n; ++i) {
        CHECK_EQ(res[i], i + 1);
    }
    LOG_INFO("add pass");

    dl::SIMD::AvxVecSub(data1.data(), data2.data(), res.data(), n);
    for (int i = 0; i < n; ++i) {
        CHECK_EQ(res[i], i - 1);
    }
    LOG_INFO("sub pass");

    dl::SIMD::AvxVecMul(data1.data(), data2.data(), res.data(), n);
    for (int i = 0; i < n; ++i) {
        CHECK_EQ(res[i], i);
    }
    LOG_INFO("mul pass");

    dl::SIMD::AvxVecDiv(data1.data(), data2.data(), res.data(), n);
    for (int i = 0; i < n; ++i) {
        CHECK_EQ(res[i], i);
    }
    LOG_INFO("div pass");

    float dot_product =
        dl::SIMD::AvxVecDotProduct(data1.data(), data2.data(), data1.size());
    CHECK_EQ(dot_product, std::accumulate(data1.begin(), data1.end(), 0));
    LOG_INFO("dot product pass");
}