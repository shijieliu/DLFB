/*
 * @Author: liushijie
 * @Date: 2020-06-28 18:04:32
 * @LastEditTime: 2020-06-30 11:12:30
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/example/mnist.cpp
 */
#include "dl.h"
using std::vector;
using std::string;

string train_img_path("../data/train-images-idx3-ubyte");
string train_label_path("../data/train-labels-idx1-ubyte");
int64_t batchsize = 8;
int64_t height    = 28;
int64_t width     = 28;
int64_t n_epoch = 100;

dl::DataNode *build_network(dl::DataNode **ret_image, dl::DataNode **ret_label,
                            dl::DataNode **ret_loss, vector<dl::DataNode*>* inner_nodes) {
    dl::DataNode *image  = dl::CreateNode({batchsize, 1, height, width});
    dl::DataNode *label  = dl::CreateNode({batchsize});
    dl::DataNode *x1     = dl::nn::Conv2D(1, 6, 5)(image);
    inner_nodes->push_back(x1);
    dl::DataNode *x2     = dl::nn::Maxpool2d(2, 2, 0)(x1);
    dl::DataNode *x3     = dl::nn::Conv2D(6, 16, 5)(x2);
    dl::DataNode *x4     = dl::nn::Maxpool2d(2, 2, 0)(x3);
    dl::DataNode *x5     = dl::nn::Conv2D(16, 120, 2)(x4);
    dl::DataNode *x6     = dl::nn::Maxpool2d(2, 2, 0)(x5);
    dl::DataNode *reshape_x6 = dl::nn::Reshape({batchsize, 120})(x6);
    dl::DataNode *x7     = dl::nn::Linear(120, 84, false)(reshape_x6);
    dl::DataNode *x8     = dl::nn::ReLU()(x7);
    dl::DataNode *x9     = dl::nn::Linear(84, 10, false)(x8);
    dl::DataNode *logits = dl::nn::Softmax()(x9);
    dl::DataNode *loss   = dl::nn::CrossEntropy()(logits, label);
    *ret_image           = image;
    *ret_label           = label;
    *ret_loss            = loss;
}

int main() {

    dl::Tensor train_data;
    dl::Tensor train_label;

    dl::ReadMnistImageData(train_img_path.c_str(), &train_data);
    dl::ReadMnistLabelData(train_label_path.c_str(), &train_label);

    dl::DataNode* image, *label, *loss;
    vector<dl::DataNode*> inner_nodes;
    build_network(&image, &label, &loss, &inner_nodes);

    auto loss_func = dl::Compile({loss});
    
    for(int epoch = 0; epoch < n_epoch; ++epoch){
        dl::Shuffle(&train_data, &train_label);
        vector<dl::Tensor> batch_imgs;
        vector<dl::Tensor> batch_labels;
        dl::Split(train_data, train_label, batchsize, &batch_imgs, &batch_labels);
        
        for(int n_step = 0; n_step < batch_imgs.size(); ++n_step){
            LOG_INFO("label shape:%s", dl::FormatShape(batch_labels[n_step].shape()).c_str());
            loss_func.forward({{image, batch_imgs[n_step]}, {label, batch_labels[n_step]}});
            float loss_value = loss->tensor()->data()[0];
            LOG_INFO("\n\tepoch:%d\n\tstep:%d\n\tloss:%f", epoch, n_step, loss_value);
            // for(dl::DataNode* inner_node: inner_nodes){
            //     dl::DisplayTensor(inner_node->tensor());
            // }
        }
    }
}