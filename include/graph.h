//
// Created by 刘仕杰 on 2020/4/6.
//

#ifndef LIGHTLR_GRGraphPH_H
#define LIGHTLR_GRGraphPH_H

#include <unordered_map>

namespace lr {
    class Node;

    class Graph {
    public:
        static Graph getInstance() {
            return m_instance;
        }

        void AddNode(Node *node);

        void Transpose();

        void TopologicalSort();

        ~Graph() {}

    private:
        Graph() {
            //do whatever
        }

        static Graph m_instance;
        std::vector<Node *> m_nodes;

    };
}
#endif //LIGHTLR_GRGraphPH_H
