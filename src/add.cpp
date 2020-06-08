//
// Created by 刘仕杰 on 2020/4/11.
//

#include "operator/add.h"

namespace dl {
    void Add::Forward() {
        printf("[Add::Forward]\n");
        assert(m_out_parameters.size() == 1);
        for (Parameter* in_node: m_in_parameters) {
            AddOp(m_out_parameters[0]->Data(), in_node->Data(), m_out_parameters[0]->MutuableData());
        }
    }

    void Add::Backward() {
        printf("[Add::Backward]\n");
        for(Parameter* in_node: m_in_parameters){
            in_node->MutuableGradient()->ones();
        }
    }

    Shape Add::GetOutShape() {
        if(m_in_parameters.size() != 1){
            for(int i = 0; i < m_in_parameters.size() - 1; ++i){
                assert(m_in_parameters[i]->Data().shape() == m_in_parameters[i + 1]->Data().shape());
            }
        }
        return m_in_parameters[0]->Data().shape();
    }
}