//
// Created by 刘仕杰 on 2020/2/2.
//

#ifndef LIGHTLR_IO_H
#define LIGHTLR_IO_H

#include "utils.h"
#include <vector>
#include <fstream>
#include <sstream>

namespace lr {
    template<typename T>
    Tensor<T> ReadTensorFromFile(const string &filename) {
        std::ifstream fs(filename);
        string line;
        std::vector<std::vector<T>> res;
        int col = -1;
        int row = 0;
        while (std::getline(fs, line)) {
            auto member = split<T>(line);
            res.push_back(member);
            row += 1;
            col = (col == -1) ? member.size() : col;
        }
        printf("[io.h]datafile: %s, row: %d, col: %d\n", filename.c_str(), row, col);
        return Tensor<T>(res, col, row);
    }

    template<typename T>
    Tensor<T> RandomTensor(const int row, const int col) {
        std::default_random_engine generator;
        std::uniform_int_distribution<int> distribution(1, 6);
        std::vector<std::vector<T>> data;
        for (int r = 0; r < row; ++r) {
            std::vector<T> one_row;
            for (int c = 0; c < col; ++c) {
                one_row.push_back(static_cast<T>(distribution(generator)));
            }
            data.push_back(std::move(one_row));
        }
        return Tensor<T>(data, col, row);
    }
}
#endif //LIGHTLR_IO_H
