//
// Created by 刘仕杰 on 2020/3/8.
//

#ifndef LIGHTLR_RANDOM_H
#define LIGHTLR_RANDOM_H

#include <random>
#include <cstdio>

namespace lr {
    static std::default_random_engine generator;

    double uniform(double low, double high);

    double gussian(double mean, double var);
}

#endif //LIGHTLR_RANDOM_H
