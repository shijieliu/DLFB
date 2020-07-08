/*
 * @Author: liushijie
 * @Date: 2020-06-22 12:02:43
 * @LastEditTime: 2020-07-08 11:34:47
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
    
template <typename T> static constexpr int Expand(T hd) { return static_cast<int>(hd); }

template <typename T, typename U, typename... Args>
static constexpr int Expand(T hd1, U hd2, Args... tl) {
    return static_cast<int>(hd1) + static_cast<int>(hd2) * Expand(tl...);
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