//
// Created by 刘仕杰 on 2020/4/6.
//

#ifndef LIGHTLR_NODE_H
#define LIGHTLR_NODE_H

#include <vector>
#include "tensor.h"

namespace lr {
    class Node {
    public:
        Node(Tensor &tensor);

        void AddInNode(Node *innode);

        void AddOutNode(Node *outNode);

        virtual void Forward();

        virtual void Backward();

    private:
        std::vector<Node *> mInNodes;
        std::vector<Node *> mOutNodes;
        Tensor mTensor;
    };


}
#endif //LIGHTLR_NODE_H
