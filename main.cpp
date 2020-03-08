#include <iostream>
#include "utils.h"
#include "tensor.h"
#include <algorithm>
#include <math.h>

const int EPOCH = 100;

int main() {
    lr::Tensor<float> X = lr::ReadTensorFromFile<float>("/home/liushijie/code/projects/LightLR/x.txt");
    lr::Tensor<float> label = lr::ReadTensorFromFile<float>("/home/liushijie/code/projects/LightLR/label.txt");
    lr::Tensor<float> weights = lr::RandomTensor<float>(X.Col(), label.Col());
    lr::L2Loss<float> lossFunc;
    int batchsize = 256;
    for (int epoch = 0; epoch < EPOCH; ++epoch) {
        auto idx = X.Shuffle();
        label = label[idx];
        int step_count = 0;
        for (int i = 0; i < idx.size(); i += batchsize) {
            int i_end = i + batchsize > idx.size() ? idx.size() : i + batchsize;
            auto train_step_x = X.Slice(i, i_end);
            auto train_step_label = label.Slice(i, i_end);
            printf("x shape:(%d, %d)\n", train_step_x.Row(), train_step_x.Col());
            printf("y shape:(%d, %d)\n", train_step_label.Row(), train_step_label.Col());
            auto predict = train_step_x * weights;
            float loss = lr::CalcLoss(&predict, train_step_label, &lossFunc);
            printf("epoch: %d, step: %d, loss value is %.2f\n", epoch, step_count, loss);
            weights.ClearGradient();
            //更新 loss gradient
            lossFunc.Backward(&predict, label);
            lr::CalcGradient(predict, train_step_x, &weights);
            weights.Step();
            step_count += 1;
        }
    }

    return 0;
}
