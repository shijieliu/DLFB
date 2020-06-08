//
// Created by 刘仕杰 on 2020/4/11.
//

#include "core/node.h"
#include "core/graph.h"

namespace dl
{
const std::string Node::Type = "node";

Node::~Node() {}

void Node::forward() {}
void Node::backward() {}

// void Parameter::Forward() {
//     std::vector<OperatorNode *> fwd_oprs = mGraph.ForwardSort(this);
//     for (OperatorNode *node: fwd_oprs) {
//         printf("[Parameter::Forward] fwd opr name: %s\n", node->mName.c_str());
//         node->Forward();
//     }
// }

// void Parameter::Backward() {
//     std::vector<OperatorNode *> bck_oprs = mGraph.BackwardSort(this);
//     for (Node *node: bck_oprs) {
//         node->Backward();
//     }
// }

} // namespace dl