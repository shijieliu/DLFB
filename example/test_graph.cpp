/*
 * @Author: your name
 * @Date: 2020-06-20 07:17:31
 * @LastEditTime: 2020-07-01 05:36:54
 * @LastEditors: liushijie
 * @Description: In User Settings Edit
 * @FilePath: /LightLR/example/test_graph.cpp
 */
#include "dl.h"
using dl::Tensor;

void test_matmul() {
    std::vector<float> raw({1, 1, 1, 1});
    Tensor             x(raw, {4, 1});
    dl::DisplayTensor(&x);

    raw = {1, 1, 1, 1, 1};
    Tensor y(raw, {1, 5});
    dl::DisplayTensor(&y);

    Tensor res({4, 5});
    dl::Mat(x, y, &res);
    dl::DisplayTensor(&res);
    LOG_INFO("test matmul finish");
}

void test_conv2d() {
    Tensor x({2, 3, 5, 5});
    dl::Random(x.data(), x.size());
    dl::DisplayTensor(&x);
    Tensor w({2, 3, 5, 5});
    dl::Random(w.data(), w.size());
    dl::DisplayTensor(&w);

    Tensor out({2, 2, 1, 1});

    dl::Conv2D(x, w, &out, 1, 0, "zeros");
    dl::DisplayTensor(&out);
}

void test_tensor() {
    std::vector<float> d{1};

    Tensor t(d, {1});
    Tensor t2 = t;
    dl::DisplayTensor(&t);
    dl::DisplayTensor(&t2);

    Tensor r{1};
    dl::Add(t, t, &r);
    dl::DisplayTensor(&r);

    test_matmul();
    test_conv2d();
}

void test_node() {

    std::vector<float>   da{1, 2, 3, 4};
    Tensor               ta(da, dl::Shape({2, 2}));
    dl::DataProviderImpl a(0, dl::Shape({2, 2}));
    *(a.tensor()) = ta;
    dl::DisplayTensor(a.tensor());

    std::vector<float>   db{2, 3, 4, 5};
    Tensor               tb(db, dl::Shape({2, 2}));
    dl::DataProviderImpl b(0, dl::Shape({2, 2}));
    *(b.tensor()) = tb;
    dl::DisplayTensor(b.tensor());

    dl::DataProviderImpl c(0, dl::Shape({2, 2}));

    dl::AddImpl add(2);
    add.setInNodes(std::vector<dl::DataNode *>{&a, &b});
    add.setOutNodes(&c);

    add.applyForward();
    dl::DisplayTensor(c.tensor());
}

void test_layer() {
    dl::DataNode *image = dl::CreateNode({1, 3, 7, 7});
    dl::DataNode *x1    = dl::nn::Conv2D(3, 5, 7, 1, 0, false)(image);
    LOG_INFO("create x1");
    dl::DisplayTensor(x1->tensor());
    dl::DataNode *x2 = dl::nn::Conv2D(5, 3, 1, 1, 0, false)(x1);
    LOG_INFO("create x2");
    dl::DisplayTensor(x2->tensor());
    LOG_INFO("create reduce mean");
    dl::DataNode *loss = dl::CreateNode<dl::ReduceMeanImpl>({x2});
    auto          func = dl::Compile({image}, {loss});

    Tensor da(image->tensor()->shape());
    dl::Random(da.data(), da.size());
    func.forward({{image, da}});
    LOG_INFO("image value");
    dl::DisplayTensor(image->tensor());
    LOG_INFO("x2 value");
    dl::DisplayTensor(x2->tensor());
    LOG_INFO("loss value");
    dl::DisplayTensor(loss->tensor());
    func.backward({{loss, *(loss->tensor())}});
    LOG_INFO("x2 grad");
    dl::DisplayTensor(x2->grad());
    LOG_INFO("x1 grad");
    dl::DisplayTensor(x1->grad());
}

void test_conv2d_opr() {
    int n           = 2; 
    int c_in        = 3;
    int height      = 10;
    int width       = 10;
    int kernel_size = 5;
    int c_out       = 2; 
    int stride      = 2; 
    int padding     = 0;

    std::unique_ptr<Tensor> inp(new Tensor({n, c_in, height, width}));
    std::unique_ptr<Tensor> weight(
        new Tensor({c_out, c_in, kernel_size, kernel_size}));
    dl::Random(inp->data(), inp->size());
    // dl::Ones(weight->data(), weight->size());
    dl::Random(weight->data(), weight->size());

    dl::Conv2DImpl              opr(0, kernel_size, stride, padding);
    std::vector<const Tensor *> tensor_and_weight({inp.get(), weight.get()});
    std::unique_ptr<Tensor>     out(
        new Tensor(opr.inferenceShape(tensor_and_weight)));

    opr.forward(tensor_and_weight, out.get());

    for (int _n = 0; _n < out->shape()[0]; ++_n) {
        for (int _c = 0; _c < out->shape()[1]; ++_c) {
            for (int _h = 0; _h < out->shape()[2]; ++_h) {
                for (int _w = 0; _w < out->shape()[3]; ++_w) {
                    float value = 0.0f;

                    int offset =
                        dl::Expand(_w, out->shape()[3], _h, out->shape()[2], _c,
                                   out->shape()[1], _n);
                    for (int _cin = 0; _cin < c_in; ++_cin) {
                        for (int k1 = 0; k1 < kernel_size; ++k1) {
                            for (int k2 = 0; k2 < kernel_size; ++k2) {
                                value += inp->data()[dl::Expand(
                                             k1 + _w * stride, width,
                                             k2 + _h * stride, height, _cin,
                                             c_in, _n)] *
                                         weight->data()[dl::Expand(
                                             k1, kernel_size, k2, kernel_size,
                                             _cin, c_in, _c)];
                            }
                        }
                    }
                    if(value - out->data()[offset] > 1e-5){
                        LOG_ERROR("offset %d bf value:%f, conv value:%f", offset, value, out->data()[offset]);
                    }
                }
            }
        }
    }

    LOG_INFO("test conv opr finish");
}

int main() {
    test_node();
    test_tensor();
    // test_graph();
    test_layer();
    test_conv2d_opr();
}