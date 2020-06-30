/*
 * @Author: liushijie
 * @Date: 2020-03-08 14:30:40
 * @LastEditTime: 2020-06-23 15:21:01
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/random.h
 */
#pragma once

#include "dag/tensor.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <random>

namespace dl {

void Random(float *data, int64_t n, float low = 0, float high = 1) {
    srand((unsigned) time(NULL));
    std::transform(data, data + n, data, [&](float) {
        return static_cast<float>(rand()) / RAND_MAX * (high - low) + low;
    });
}

void Ones(float *data, int64_t n) {
    std::transform(data, data + n, data, [](float) { return 1.0; });
}
}
