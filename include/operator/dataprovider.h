//
// Created by 刘仕杰 on 2020/4/6.
//

#ifndef LIGHTLR_DATAPROVIDER_H
#define LIGHTLR_DATAPROVIDER_H

#include "node.h"

namespace lr {
    class DataProvider : public Node {
    private:
        Tensor m_data;
    };
}
#endif //LIGHTLR_DATAPROVIDER_H
