//
// Created by 刘仕杰 on 2020/2/2.
//

#ifndef LIGHTLR_UTILS_H
#define LIGHTLR_UTILS_H


#include "core/tensor.h"
#include <string>
#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>

namespace dl{
    using std::string;

    template<typename T>
    T StringToNum(const string &str) {
        T num;
        std::istringstream iss(str);
        iss >> num;
        return num;
    }

    template<typename T>
    std::vector<T> split(string &str, const char deli = '\t') {
        std::vector<T> res;
        std::istringstream strstream(str);
        string token;
        while (std::getline(strstream, token, deli)) {
            res.push_back(StringToNum<T>(token));
        }
        return res;
    }


}
#endif //LIGHTLR_UTILS_H
