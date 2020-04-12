//
// Created by 刘仕杰 on 2020/4/11.
//

#include "node.h"

namespace lr {
    void Parameter::Forward() {
        std::vector<OperatorNode *> fwd_oprs = m_graph.TopologicalSort(this);
        for (OperatorNode *node: fwd_oprs) {
            printf("[Parameter::Forward] fwd opr name: %s\n", node->m_name.c_str());

            node->Forward();
        }
    }

    void Parameter::Backward() {
        std::vector<OperatorNode *> bck_oprs = m_graph.Transpose(this);
        for (Node *node: bck_oprs) {
            node->Backward();
        }
    }

}