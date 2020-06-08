//
// Created by 刘仕杰 on 2020/3/8.
//

#ifndef RANDOM_H
#define RANDOM_H

#include <random>
#include <cstdio>

namespace dl {
    static std::default_random_engine generator;

    inline double uniform(double low, double high) {
        std::uniform_real_distribution<double> uniform{low, high};
        return uniform(generator);
    }

    inline double gussian(double mean, double var) {
        std::normal_distribution<double> norm{mean, var};
        return norm(generator);
    }
}

#endif //LIGHTLR_RANDOM_H
