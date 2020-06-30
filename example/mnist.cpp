/*
 * @Author: liushijie
 * @Date: 2020-06-28 18:04:32
 * @LastEditTime: 2020-07-01 05:35:48
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/example/mnist.cpp
 */
#include "dl.h"
#include <chrono>
#include <functional>
using std::vector;
using std::string;

string train_img_path("../data/train-images-idx3-ubyte");
string train_label_path("../data/train-labels-idx1-ubyte");
int    batchsize    = 8;
int    height       = 28;
int    width        = 28;
int    n_epoch      = 1;
float  lr           = 1e-3;
float  momentum     = 0.9;
float  weight_decay = 1e-3;

dl::DataNode *
    build_network(dl::DataNode **ret_image, dl::DataNode **ret_label,
                  dl::DataNode **ret_loss,
                  std::unordered_map<string, dl::DataNode *> *inner_nodes) {
    dl::DataNode *image = dl::CreateNode({batchsize, 1, height, width});
    dl::DataNode *label = dl::CreateNode({batchsize});
    dl::DataNode *x1    = dl::nn::Conv2D(1, 6, 5)(image);
    inner_nodes->insert(std::make_pair("x1", x1));
    dl::DataNode *x2 = dl::nn::Maxpool2d(2, 2, 0)(x1);
    inner_nodes->insert(std::make_pair("x2", x2));
    dl::DataNode *x3 = dl::nn::Conv2D(6, 16, 5)(x2);
    dl::DataNode *x4 = dl::nn::Maxpool2d(2, 2, 0)(x3);
    dl::DataNode *x5 = dl::nn::Conv2D(16, 120, 2)(x4);
    inner_nodes->insert(std::make_pair("x5", x5));
    dl::DataNode *x6 = dl::nn::Maxpool2d(2, 2, 0)(x5);
    inner_nodes->insert(std::make_pair("x6", x6));
    dl::DataNode *reshape_x6 = dl::nn::Reshape({batchsize, 120})(x6);
    inner_nodes->insert(std::make_pair("reshape x6", reshape_x6));
    dl::DataNode *x7 = dl::nn::Linear(120, 84, false)(reshape_x6);
    dl::DataNode *x8 = dl::nn::ReLU()(x7);
    dl::DataNode *x9 = dl::nn::Linear(84, 10, false)(x8);
    inner_nodes->insert({std::make_pair("x9", x9)});
    dl::DataNode *logits = dl::nn::Softmax()(x9);
    inner_nodes->insert({std::make_pair("logits", logits)});
    dl::DataNode *loss = dl::nn::CrossEntropy()(logits, label);
    inner_nodes->insert({std::make_pair("loss", loss)});
    dl::DataNode *loss_mean = dl::CreateNode<dl::ReduceMeanImpl>({loss});
    inner_nodes->insert({std::make_pair("loss_mean", loss_mean)});
    *ret_image = image;
    *ret_label = label;
    *ret_loss  = loss_mean;
}

int main() {

    dl::Tensor train_data;
    dl::Tensor train_label;

    dl::ReadMnistImageData(train_img_path.c_str(), &train_data);
    dl::ReadMnistLabelData(train_label_path.c_str(), &train_label);

    dl::DataNode *image, *label, *loss;
    std::unordered_map<string, dl::DataNode *> inner_nodes;
    build_network(&image, &label, &loss, &inner_nodes);

    auto loss_func = dl::Compile({image, label}, {loss});

    std::unique_ptr<dl::Optimizer> optimizer(new dl::SGD(
        dl::Graph::GetInstance().params(), lr, momentum, weight_decay));
    vector<dl::Tensor> batch_imgs;
    vector<dl::Tensor> batch_labels;
    for (int epoch = 0; epoch < n_epoch; ++epoch) {
        dl::Shuffle(&train_data, &train_label);
        batch_imgs.clear();
        batch_labels.clear();
        dl::Split(train_data, train_label, batchsize, &batch_imgs,
                  &batch_labels);

        LOG_INFO("start epoch:%d", epoch);
        for (int n_step = 0; n_step < batch_imgs.size(); ++n_step) {
            optimizer->zeroGrad();
            dl::Time("forward", [&]() {
                loss_func.forward({{image, batch_imgs[n_step]},
                                   {label, batch_labels[n_step]}});
            });

            float loss_value = loss->tensor()->data()[0];
            LOG_INFO("\n\tepoch:%d\n\tstep:%d\n\tloss:%f", epoch, n_step,
                     loss_value);
            dl::Time("backward", [&]() {
                loss_func.backward({{loss, *(loss->tensor())}});
            });

            optimizer->step();
            for (auto inner_node_pack : inner_nodes) {
                dl::DisplayTensor(inner_node_pack.second->tensor(),
                                  inner_node_pack.first + " tensor ");
                dl::DisplayTensor(inner_node_pack.second->grad(),
                                  inner_node_pack.first + " grad ");
            }
            assert(0);
        }
    }
}