/*
 * @Author: liushijie
 * @Date: 2020-06-22 12:02:43
 * @LastEditTime: 2020-06-30 20:58:16
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
#include <chrono>

namespace dl {
std::string GetEnv(const char *env_var, const char *default_) {
    char *val = std::getenv(env_var);
    if (val == nullptr) {
        return default_;
    } else {
        return val;
    }
}

template <typename T> static constexpr T Expand(T hd) { return hd; }

template <typename T, typename... Args>
static constexpr T Expand(T hd1, T hd2, Args... tl) {
    return hd1 + hd2 * Expand(tl...);
}

template <typename F, typename... Args>
void Time(const char *str, F &&func, Args&& ... args) {
    auto start = std::chrono::system_clock::now();
    func(args...);
    auto end = std::chrono::system_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    LOG_INFO("%s time spend: %f", str,
             double(duration.count()) * std::chrono::microseconds::period::num /
                 std::chrono::microseconds::period::den);
}
}