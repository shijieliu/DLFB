/*
 * @Author: liushijie
 * @Date: 2020-06-22 12:02:43
 * @LastEditTime: 2020-06-29 11:49:09
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/utils.h
 */
//
// Created by 刘仕杰 on 2020/2/2.
//
#pragma once
#include <cstdlib>
#include <string>

namespace dl {
// using std::string;

// template <typename T> T StringToNum(const string &str) {
//     T                  num;
//     std::istringstream iss(str);
//     iss >> num;
//     return num;
// }

// template <typename T> std::vector<T> split(string &str, const char deli =
// '\t') {
//     std::vector<T>     res;
//     std::istringstream strstream(str);
//     string             token;
//     while (std::getline(strstream, token, deli)) {
//         res.push_back(StringToNum<T>(token));
//     }
//     return res;
// }

/** bare-bones unique_ptr
 * this one deletes with delete [] */
template <class T> struct ScopeDeleter {
    const T *ptr;
    explicit ScopeDeleter(const T *ptr = nullptr)
        : ptr(ptr) {}
    // void release () {ptr = nullptr; }
    // void set (const T * ptr_in) { ptr = ptr_in; }
    // void swap (ScopeDeleter<T> &other) {std::swap (ptr, other.ptr); }
    ~ScopeDeleter() { delete[] ptr; }
};

std::string GetEnv(const char *env_var, const char *default_) {
    char *val = std::getenv(env_var);
    if (val == nullptr) {
        return default_;
    } else {
        return val;
    }
}

template <typename T> static constexpr T expand(T hd) { return hd; }

template <typename T, typename... Args>
static constexpr T expand(T hd1, T hd2, Args... tl) {
    return hd1 + hd2 * expand(tl...);
}

}