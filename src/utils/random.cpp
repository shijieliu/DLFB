//
// Created by 刘仕杰 on 2020/3/8.
//

#include "utils/random.h"

namespace lr {
    double uniform(double low, double high) {
        std::uniform_real_distribution<double> uniform{low, high};
        return uniform(generator);
    }

    double gussian(double mean, double var) {
        std::normal_distribution<double> norm{mean, var};
        return norm(generator);
    }
}