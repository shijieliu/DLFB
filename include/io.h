/*
 * @Author: your name
 * @Date: 2020-06-20 07:07:04
 * @LastEditTime: 2020-06-30 19:55:05
 * @LastEditors: liushijie
 * @Description: In User Settings Edit
 * @FilePath: /LightLR/include/io.h
 */
//
// Created by 刘仕杰 on 2020/2/2.
//
#pragma once
#include "utils.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string.h>
#include <vector>

namespace dl {
void ReadMnistImageData(const char *filename, Tensor *res) {
    LOG_INFO("start reading mnist image data");
    // read images
    FILE *stream = fopen(filename, "rb");
    if (stream == nullptr) {
        LOG_ERROR("no file to read %s", filename);
    }
    int32_t magic_number;
    int32_t n_images;
    int32_t n_rows;
    int32_t n_cols;
    fread(&magic_number, sizeof(magic_number), 1, stream);
    fread(&n_images, sizeof(n_images), 1, stream);
    fread(&n_rows, sizeof(n_rows), 1, stream);
    fread(&n_cols, sizeof(n_cols), 1, stream);
    magic_number = __builtin_bswap32(magic_number);
    n_images     = __builtin_bswap32(n_images);
    n_rows       = __builtin_bswap32(n_rows);
    n_cols       = __builtin_bswap32(n_cols);
    n_images = 100;
    LOG_INFO("\n\timage data "
             "meta\n\t\tmagic_number:%d\n\t\tn_images:%d\n\t\tn_rows:%d\n\t\tn_cols:%d",
             magic_number, n_images, n_rows, n_cols);
    res->reshape({n_images, n_rows, n_cols});
    std::vector<uint8_t> buffer(n_rows * n_cols, 0);
    LOG_DEBUG("res size:%lu, buffer size:%lu", res->size(), buffer.size());

    int offset = 0;
    while (int ret = fread(buffer.data(), sizeof(uint8_t), n_rows * n_cols,
                              stream) > 0) {
        for (int i = 0; i < n_rows * n_cols; ++i) {
            res->data()[offset + i] = static_cast<float>(buffer[i]);
        }
        offset += ret;
    }
    fclose(stream);
}

void ReadMnistLabelData(const char *filename, Tensor *res) {
    // read images
    FILE *stream = fopen(filename, "rb");
    if (stream == nullptr) {
        LOG_ERROR("no file to read %s", filename);
    }
    int32_t magic_number;
    int32_t n_images;
    fread(&magic_number, sizeof(magic_number), 1, stream);
    fread(&n_images, sizeof(n_images), 1, stream);
    magic_number = __builtin_bswap32(magic_number);
    n_images     = __builtin_bswap32(n_images);
    n_images = 100;

    LOG_INFO("\n\timage data meta\n\t\tmagic_number:%d\n\t\tn_images:%d",
             magic_number, n_images);
    res->reshape({n_images});
    std::vector<uint8_t> buffer(1024, 0);

    int offset = 0;
    while (int ret =
               fread(buffer.data(), sizeof(uint8_t), 1024, stream) > 0) {
        for (int i = 0; i < ret; ++i) {
            *(res->data() + offset + i) = static_cast<float>(buffer[i]);
        }
        offset += ret;
    }
    fclose(stream);
}

/**
 * @description: 
 * @param {type} shuffle the dataset: image (n, c, h, w), label(n)
 * @return: 
 */
void Shuffle(Tensor* image, Tensor* label){
    CHECK_EQ(image->shape()[0], label->shape()[0]);
    int n = image->shape()[0];
    int offset = image->size() / n;
    std::vector<int> idx;
    for(int i = 0; i < n; ++i){
        idx.push_back(i);
    }
    std::random_shuffle(idx.begin(), idx.end());
    
    Tensor image_copy = *image;
    for(int i = 0; i < n; ++i){
        memcpy(image->data() + offset * i, image_copy.data() + idx[i] * offset, sizeof(float) * offset);
    }

    Tensor label_copy = *label;
    for(int i = 0; i < n; ++i){
        label->data()[i] = label_copy.data()[idx[i]];
    }
}

void Split(const Tensor& image, const Tensor& label, int batchsize, std::vector<Tensor>* batch_image, std::vector<Tensor>* batch_label){
    CHECK_EQ(image.shape()[0], label.shape()[0]);
    int n = image.shape()[0];
    int offset = image.size() / n;

    for(int b = 0; (b + 1) * batchsize < n; ++b){
        Tensor curr_image({batchsize, 1,image.shape()[1], image.shape()[2]});
        curr_image.copyFrom(image.data() + b * batchsize * offset, image.data() + (b + 1) * batchsize * offset);
        
        Tensor curr_label({batchsize});
        curr_label.copyFrom(label.data() + b * batchsize, label.data() + (b + 1) * batchsize);
        
        batch_image->push_back(std::move(curr_image));
        batch_label->push_back(std::move(curr_label));
    }
}

}