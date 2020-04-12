//
// Created by 刘仕杰 on 2020/4/6.
//

#ifndef LIGHTLR_GRGraphPH_H
#define LIGHTLR_GRGraphPH_H

#include <unordered_map>
#include <string>
#include <memory>

namespace lr {
    class Parameter;

    class OperatorNode;

    class Graph {
    public:
        static Graph &getInstance() {
            static Graph m_instance;
            return m_instance;
        }

        Graph(const Graph &other) = delete;

        Graph &operator=(const Graph &other) = delete;

        void AddNode(OperatorNode *father, Parameter *child);

        void AddNode(Parameter *father, OperatorNode *child);

        std::vector<OperatorNode *> Transpose(Parameter *dst);

        std::vector<OperatorNode *> TopologicalSort(Parameter *dst);

    private:
        Graph() {}

        std::unordered_multimap<std::string, std::string> m_graph;
        std::unordered_map<std::string, Parameter *> m_parameter_map;
        std::unordered_map<std::string, OperatorNode *> m_operator_map;
    };

}
#endif //LIGHTLR_GRGraphPH_H
