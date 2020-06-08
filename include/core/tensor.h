//
// Created by 刘仕杰 on 2020/1/25.
//

#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <memory>
#include <vector>
#include <initializer_list>
#include <cassert>
#include <algorithm>
#include <cstring>
#include "macro.h"

namespace dl
{
using Shape = std::vector<size_t>;

class Tensor
{
public:
    explicit Tensor(std::vector<float> &data, const Shape &shape) : mShape(shape)
    {
        for (size_t s : shape)
        {
            mSize *= s;
        }
        copyFrom(data.begin(), data.end());
    }

    explicit Tensor(const Shape &shape) : mShape(shape)
    {
        for (size_t s : shape)
        {
            mSize *= s;
        }
        float *data = new float[mSize];
        reset(data);
        memset((void *)data, 0, mSize * sizeof(float));
    }

    Tensor(const Tensor &other) = delete;

    Tensor(Tensor &&other) = default;

    Tensor &operator=(const Tensor &other) = delete;

    template <typename ForwardIt>
    void copyFrom(const ForwardIt &first, const ForwardIt &last)
    {
        int size = static_cast<int>(std::distance(first, last));
        float *data = new float[size];
        reset(data);
        auto it = first;
        while (size-- > 0)
        {
            *data = *it;
            ++data;
            ++it;
        }
    }

    const Shape &shape() const
    {
        return mShape;
    }

    const size_t size() const
    {
        return mSize;
    }

    float &operator[](size_t idx)
    {
        return mData.get()[idx];
    }

    const float &operator[](size_t idx) const
    {
        return mData.get()[idx];
    }

    void reset(float *data)
    {
        mData = std::unique_ptr<float, Deleter>(data, [](float *data) { delete[] data; });
    }

private:
    Shape mShape;
    size_t mSize = 1;

    using Deleter = std::function<void(float *)>;
    std::unique_ptr<float, Deleter> mData;
};

inline Tensor Add(const Tensor &lhs, const Tensor &rhs)
{
    CHECK_EQ(lhs.shape(), rhs.shape());
    CHECK_EQ(lhs.size(), rhs.size());
    Tensor res(lhs.shape());
    for (size_t idx = 0; idx < lhs.size(); ++idx)
    {
        res[idx] = lhs[idx] + rhs[idx];
    }
    return std::move(res);
}

inline Tensor Sub(const Tensor &lhs, const Tensor &rhs)
{
    CHECK_EQ(lhs.shape(), rhs.shape());
    CHECK_EQ(lhs.size(), rhs.size());
    Tensor res(lhs.shape());
    for (size_t idx = 0; idx < lhs.size(); ++idx)
    {
        res[idx] = lhs[idx] - rhs[idx];
    }
    return std::move(res);
}

inline Tensor Mul(const Tensor &lhs, const Tensor &rhs)
{
    CHECK_EQ(lhs.shape(), rhs.shape());
    CHECK_EQ(lhs.size(), rhs.size());
    Tensor res(lhs.shape());
    for (size_t idx = 0; idx < lhs.size(); ++idx)
    {
        res[idx] = lhs[idx] * rhs[idx];
    }
    return std::move(res);
}

inline Tensor Div(const Tensor &lhs, const Tensor &rhs)
{
    CHECK_EQ(lhs.shape(), rhs.shape());
    CHECK_EQ(lhs.size(), rhs.size());
    Tensor res(lhs.shape());
    for (size_t idx = 0; idx < lhs.size(); ++idx)
    {
        res[idx] = lhs[idx] / rhs[idx];
    }
    return std::move(res);
}

inline Tensor Mat(const Tensor &lhs, const Tensor &rhs)
{
    CHECK_EQ(lhs.shape().size(), 2);
    Tensor res(lhs.shape());
    return std::move(res);
    //    Tensor Tensor::matrix_mul(const Tensor &other) const {
    //        if(this->shape().size() != 2 || other.shape().size() != 2){
    //            throw std::runtime_error("matrix mul shape must be 2");
    //        }
    //        if(this->shape()[1] != other.shape()[0]){
    //            throw std::runtime_error("matrix mul col row must equal");
    //        }
    //        int row = this->shape()[0];
    //        int col = other.shape()[1];
    //        int common_size = this->shape()[1];
    //        Shape tmpshape(2, 0);
    //        Tensor res{Shape (row, col)};
    //        res.zeros();
    //        for(int r = 0; r < row; ++r){
    //            for(int c = 0; c < col; ++c){
    //                for(int i = 0; i < common_size; ++i){
    //                    tmpshape[0] = r;
    //                    tmpshape[1] = i;
    //                    double first = this->getValue(tmpshape);
    //                    tmpshape[0] = i;
    //                    tmpshape[1] = c;
    //                    double second = other.getValue(tmpshape);
    //                    tmpshape[0] = r;
    //                    tmpshape[1] = c;
    //                    int current = res.getValue(tmpshape);
    //                    res.setValue(tmpshape, current + first * second);
    //                }
    //            }
    //        }
    //        return res;
    //    }
}

} // namespace dl

#endif //TENSOR_H
