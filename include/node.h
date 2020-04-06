//
// Created by 刘仕杰 on 2020/4/6.
//

#ifndef LIGHTLR_NODE_H
#define LIGHTLR_NODE_H

#include <vector>
#include "tensor.h"
#include "graph.h"

namespace lr {
    class Node {
    public:
        Node(Graph &graph = Graph::getInstance()) : m_graph(graph) {}

        void AddInNode(Node *innode);

        void AddOutNode(Node *outNode);

    private:
        std::vector<Node *> m_in_nodes;
        std::vector<Node *> m_out_nodes;
        Graph m_graph;
    };


}
#endif //LIGHTLR_NODE_H
