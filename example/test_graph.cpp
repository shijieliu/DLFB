/*
 * @Author: your name
 * @Date: 2020-06-20 07:17:31
 * @LastEditTime: 2020-06-29 19:22:43
 * @LastEditors: liushijie
 * @Description: In User Settings Edit
 * @FilePath: /LightLR/example/test_graph.cpp
 */
#include "dl.h"

void test_matmul() {
    std::vector<float> raw({1, 1, 1, 1});
    dl::Tensor         x(raw, {4, 1});
    dl::DisplayTensor(&x);

    raw = {1, 1, 1, 1, 1};
    dl::Tensor y(raw, {1, 5});
    dl::DisplayTensor(&y);

    dl::Tensor res({4, 5});
    dl::Mat(x, y, &res);
    dl::DisplayTensor(&res);
    LOG_INFO("test matmul finish");
}

void test_conv2d() {
    dl::Tensor x({2, 3, 5, 5});
    dl::Random(x.data(), x.size());
    dl::DisplayTensor(&x);
    dl::Tensor w({2, 3, 5, 5});
    dl::Random(w.data(), w.size());
    dl::DisplayTensor(&w);

    dl::Tensor out({2, 2, 1, 1});

    dl::Conv2D(x, w, &out, 1, 0, "zeros");
    dl::DisplayTensor(&out);
}

void test_tensor() {
    std::vector<float> d{1};

    dl::Tensor t(d, {1});
    dl::Tensor t2 = t;
    dl::DisplayTensor(&t);
    dl::DisplayTensor(&t2);

    dl::Tensor r{1};
    dl::Add(t, t, &r);
    dl::DisplayTensor(&r);

    test_matmul();
    test_conv2d();
}

void test_node() {

    std::vector<float>   da{1, 2, 3, 4};
    dl::Tensor           ta(da, dl::Shape({2, 2}));
    dl::DataProviderImpl a(0, dl::Shape({2, 2}));
    *(a.tensor()) = ta;
    dl::DisplayTensor(a.tensor());

    std::vector<float>   db{2, 3, 4, 5};
    dl::Tensor           tb(db, dl::Shape({2, 2}));
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

void test_layer(){
    dl::DataNode *image = dl::CreateNode({1, 3, 7, 7});
    dl::DataNode* x1 = dl::nn::Conv2D(3, 5, 7, 1, 0, false)(image);
    LOG_INFO("create x1");
    dl::DisplayTensor(x1->tensor());
    dl::DataNode *x2 = dl::nn::Conv2D(5, 3, 1, 1, 0, false)(x1);
    LOG_INFO("create x2");
    dl::DisplayTensor(x2->tensor());
    LOG_INFO("create reduce mean");
    dl::DataNode *loss = dl::CreateNode<dl::ReduceMeanImpl>({x2});
    auto func = dl::Compile({loss});

    dl::Tensor da(image->tensor()->shape());
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
int main() {
    test_node();
    test_tensor();
    // test_graph();
    test_layer();
}