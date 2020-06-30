//
// Created by 刘仕杰 on 2020/1/25.
//
#pragma once

#include "avx.h"
#include "macro.h"
#include "utils.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <vector>

namespace dl {
using Shape = std::vector<int64_t>;

std::string FormatShape(const Shape& shape){
    std::string res("(");
    for(int64_t s : shape){
        res += std::to_string(s);
        res += ",";
    }
    res += ")";
    return res;
}
class Tensor {
  public:
    Tensor() {}
    explicit Tensor(std::vector<float> &                 data,
                    const std::initializer_list<int64_t> shape)
        : Tensor(data, Shape(shape.begin(), shape.end())) {}

    explicit Tensor(std::vector<float> &data, const Shape &shape) {
        reshape(shape);
        copyFrom(data.begin(), data.end());
    }

    explicit Tensor(const std::initializer_list<int64_t> shape)
        : Tensor(Shape(shape.begin(), shape.end())) {}

    explicit Tensor(const Shape &shape) {
        reshape(shape);
        float *data = new float[mSize];
        reset(data);
        memset((void *) data, 0, mSize * sizeof(float));
    }

    Tensor(const Tensor &other) {
        mShape = other.shape();
        mSize  = other.size();
        copyFrom(other.data(), other.data() + mSize);
    }

    Tensor &operator=(const Tensor &other) {
        mShape = other.shape();
        mSize  = other.size();
        copyFrom(other.data(), other.data() + mSize);
        return *this;
    }

    Tensor(Tensor &&other) = default;

    template <typename ForwardIt>
    void copyFrom(const ForwardIt &first, const ForwardIt &last) {
        int    size = static_cast<int>(std::distance(first, last));
        float *data = new float[size];
        reset(data);
        auto it = first;
        while (size-- > 0) {
            *data = *it;
            ++data;
            ++it;
        }
    }

    void reshape(const Shape &shape) {
        mShape = shape;
        mSize  = 1;
        for (int64_t s : shape) {
            mSize *= s;
        }
        mData.release();
        float *data = new float[mSize];
        reset(data);
        memset((void *) mData.get(), 0, mSize * sizeof(float));
    }

    void reshape(std::initializer_list<int64_t> shape) {
        reshape(Shape(shape.begin(), shape.end()));
    }

    const Shape &shape() const { return mShape; }

    int64_t size() const { return mSize; }

    float *data() const { return mData.get(); }

    void reset(float *data) {
        mData = std::unique_ptr<float, Deleter>(
            data, [](float *data) { delete[] data; });
    }

  private:
    Shape  mShape;
    int64_t mSize  = 1;
    using Deleter = std::function<void(float *)>;
    std::unique_ptr<float, Deleter> mData;
    int32_t mDevice;
};


void DisplayTensor(const dl::Tensor* t, const std::string& tensor_name="Tensor") {
    std::string shape_info = FormatShape(t->shape());

    std::string value_info("[");
    for (int64_t i = 0; i < t->size(); ++i) {
        value_info.append(std::to_string(t->data()[i]));
        if(i < 10){
            value_info.append(",");
        }else{
            value_info.append("...");break;
        }
    }
    value_info.append("]");
    LOG_INFO("\n\t%s info:\n\t\tshape:%s\n\t\tvalue:%s", tensor_name.c_str(), shape_info.c_str(), value_info.c_str());
}


inline void Add(const Tensor &lhs, const Tensor &rhs, Tensor *out) {
    CHECK_EQ(lhs.shape(), rhs.shape());
    CHECK_EQ(out->shape(), lhs.shape());
    CHECK_EQ(lhs.size(), rhs.size());
    SIMD::AvxVecAdd(lhs.data(), rhs.data(), out->data(), out->size());
}

inline void Sub(const Tensor &lhs, const Tensor &rhs, Tensor *out) {
    CHECK_EQ(lhs.shape(), rhs.shape());
    CHECK_EQ(out->shape(), lhs.shape());
    CHECK_EQ(lhs.size(), rhs.size());
    SIMD::AvxVecSub(lhs.data(), rhs.data(), out->data(), out->size());
}

inline void Mul(const Tensor &lhs, const Tensor &rhs, Tensor *out) {
    CHECK_EQ(lhs.shape(), rhs.shape());
    CHECK_EQ(out->shape(), lhs.shape());
    CHECK_EQ(lhs.size(), rhs.size());
    SIMD::AvxVecMul(lhs.data(), rhs.data(), out->data(), out->size());
}


inline void Mul(const Tensor &lhs, const float rhs, Tensor *out) {
    CHECK_EQ(out->shape(), lhs.shape());
    SIMD::AvxVecMul(lhs.data(), rhs, out->data(), out->size());
}


inline void Div(const Tensor &lhs, const Tensor &rhs, Tensor *out) {
    CHECK_EQ(lhs.shape(), rhs.shape());
    CHECK_EQ(out->shape(), lhs.shape());
    CHECK_EQ(lhs.size(), rhs.size());
    SIMD::AvxVecDiv(lhs.data(), rhs.data(), out->data(), out->size());
}

/**
 * @description:
 * @param {type}  lhs:(row, len) rhs:(len, col) out:(row, col)
 * @return:
 */
inline void Mat(const Tensor &lhs, const Tensor &rhs, Tensor *out) {
    CHECK_EQ(lhs.shape().size(), 2);
    CHECK_EQ(lhs.shape()[1], rhs.shape()[0]);

    int64_t row = lhs.shape()[0];
    int64_t col = rhs.shape()[1];
    int64_t len = lhs.shape()[1];
    CHECK_EQ(out->shape()[0], row);
    CHECK_EQ(out->shape()[1], col);

    float *             transpose_rhs = new float[rhs.size()];
    ScopeDeleter<float> delete_transpose_rhs(transpose_rhs);
    for (int64_t r = 0; r < len; ++r) {
        for (int64_t c = 0; c < col; ++c) {
            transpose_rhs[expand(r, len, c)] = rhs.data()[expand(c, col, r)];
        }
    }
    for (int64_t r = 0; r < row; ++r) {
        for (int64_t c = 0; c < col; ++c) {
            out->data()[r * col + c] = SIMD::AvxVecDotProduct(
                lhs.data() + r * len, transpose_rhs + c * len, len);
        }
    }
}

/**
 * @description:
 * @param {type}  inp (n, c, h, w), out (n, c, h + 2 * padding, w + 2 * padding)
 * @return:
 */
inline void Padding(const Tensor &inp, Tensor *out, int64_t padding,
                    const char *padding_mode) {
    int64_t n         = inp.shape()[0];
    int64_t c         = inp.shape()[1];
    int64_t h         = inp.shape()[2];
    int64_t w         = inp.shape()[3];
    int64_t padding_h = h + 2 * padding;
    int64_t padding_w = w + 2 * padding;
    memset(out->data(), 0, sizeof(float) * out->size());

    for (int64_t step_n = 0; step_n < n; ++step_n) {
        for (int64_t step_c = 0; step_c < c; ++step_c) {
            for (int64_t step_h = 0; step_h < h; ++step_h) {
                memcpy(out->data() + expand(padding, padding_w,
                                            padding + step_h, padding_h, step_c,
                                            c, step_n),
                       inp.data() + expand(step_h, h, step_c, c, step_n),
                       sizeof(float) * w);
            }
        }
    }
}
/**
 * @description:
 * @param {type} out (c_out, c_in, k, k)
 * @return:
 */
inline void Rotate(const Tensor &inp, Tensor *out) {
    CHECK_EQ(inp.shape().size(), 4);
    CHECK_EQ(inp.shape(), out->shape());

    int64_t c_out = inp.shape()[0];
    int64_t c_in  = inp.shape()[1];
    int64_t k     = inp.shape()[2];
    for (int64_t step_c_out = 0; step_c_out < c_out; ++step_c_out) {
        for (int64_t step_c_in = 0; step_c_in < c_in; ++step_c_in) {
            for (int64_t k1 = 0; k1 < k; ++k1) {
                for (int64_t k2 = 0; k2 < k; ++k2) {
                    out->data()[expand(k1, k, k2, k, step_c_in, c_in,
                                       step_c_out)] =
                        inp.data()[expand(k2, k, k1, k, step_c_in, c_in,
                                          step_c_out)];
                }
            }
        }
    }
}
/**
 * @description:
 * @param {type} x (n, c_in, h, w), weight (c_out, c_in, k, k) out(n, c_out, h*,
 * w*)
 * @return:
 */
inline void Conv2D(const Tensor &x, const Tensor &weight, Tensor *out,
                   int64_t stride, int64_t padding,
                   const std::string &padding_mode) {
    CHECK_EQ(x.shape().size(), 4);
    CHECK_EQ(weight.shape().size(), 4);
    LOG_DEBUG("\n\tTensor Conv2D args:\n\t\tx shape:(%lu, %lu, %lu, %lu)\n\t\tweight shape:(%lu, %lu, %lu, %lu)\n\t\tstride:%lu\n\t\tpadding:%lu", x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3], weight.shape()[0], weight.shape()[1], weight.shape()[2], weight.shape()[3], stride, padding);

    CHECK_EQ(x.shape()[1], weight.shape()[1]);

    int64_t              c_out = weight.shape()[0];
    int64_t              c_in  = weight.shape()[1];
    int64_t              k     = weight.shape()[2];
    int64_t              n     = x.shape()[0];
    int64_t              h     = x.shape()[2];
    int64_t              w     = x.shape()[3];
    int64_t              h_out     = (h + 2 * padding - k) / stride + 1;
    int64_t              w_out     = (w + 2 * padding - k) / stride + 1;
    LOG_DEBUG("(h_out: %lu, w_out: %lu)", h_out, w_out);
    CHECK_EQ(out->shape()[0], n);
    CHECK_EQ(out->shape()[1], c_out);
    CHECK_EQ(out->shape()[2], h_out);
    CHECK_EQ(out->shape()[3], w_out);

    Tensor padding_x;
    if (padding > 0) {
        if (padding_mode == "zeros") {
            h += 2 * padding;
            w += 2 * padding;
            padding_x.reshape({n, c_in, h, w});
            Padding(x, &padding_x, padding, padding_mode.c_str());
        } else {
            LOG_ERROR("dont support padding mode %s", padding_mode.c_str());
            assert(0);
        }
    } else {
        padding_x = x;
    }

    Tensor flatten_x({c_in * k * k, n * h_out * w_out});

    // build flatten_x
    {

        for (int64_t n_step = 0; n_step < n; ++n_step) {
            for (int64_t h_step = 0; h_step <= h - k; h_step += stride) {
                for (int64_t w_step = 0; w_step <= w - k; w_step += stride) {
                    for (int64_t c_step = 0; c_step < c_in; ++c_step) {
                        for (int64_t k1_step = 0; k1_step < k; ++k1_step) {
                            for (int64_t k2_step = 0; k2_step < k; ++k2_step) {
                                int64_t offset_flatten_x = expand(
                                    w_step, w - k + 1, h_step, h - k + 1,
                                    n_step, n, k2_step, k, k1_step, k, c_step);
                                int64_t offset_x = expand(w_step + k2_step, w,
                                                         h_step + k1_step, h,
                                                         c_step, c_in, n_step);
                                *(flatten_x.data() + offset_flatten_x) =
                                    *(x.data() + offset_x);
                            }
                        }
                    }
                }
            }
        }
    }

    Tensor flatten_weight({c_out, c_in * k * k});
    flatten_weight.copyFrom(weight.data(), weight.data() + weight.size());
    Tensor tmp_out({c_out, n * h_out * w_out});
    Mat(flatten_weight, flatten_x, &tmp_out);

    for (int64_t c_step = 0; c_step < c_out; ++c_step) {
        for (int64_t n_step = 0; n_step < n; ++n_step) {
            memcpy(out->data() + expand(c_step, c_out, n_step),
                   tmp_out.data() + expand(n_step, n, c_step),
                   sizeof(float) * (h - k + 1) * (w - k + 1));
        }
    }
}
} // namespace dl
