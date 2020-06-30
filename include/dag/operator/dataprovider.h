/*
 * @Author: your name
 * @Date: 2020-06-20 07:05:29
 * @LastEditTime: 2020-06-20 07:05:29
 * @LastEditors: your name
 * @Description: In User Settings Edit
 * @FilePath: /LightLR/include/dag/operator/dataprovider.h
 */
//
// Created by 刘仕杰 on 2020/4/6.
//
#pragma once
#include "dag/node.h"
#include "macro.h"

namespace dl {

class DataProviderImpl : public DataNode {
  public:
    DataProviderImpl(const int64_t uid, const Shape &shape)
        : DataNode(uid, shape, false) {}
    virtual ~DataProviderImpl() = default;
};
} // namespace dl