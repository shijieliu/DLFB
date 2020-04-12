//
// Created by 刘仕杰 on 2020/4/6.
//

#ifndef LIGHTLR_NODE_H
#define LIGHTLR_NODE_H

#include <vector>
#include <string>
#include <memory>
#include "tensor.h"
#include "graph.h"

namespace lr {
    class Node {
    public:
        explicit Node(std::string &name) : m_name(name) {}

        Node(const Node &) = delete;

        Node &operator=(const Node &other) = delete;

        virtual void Forward() = 0;

        virtual void Backward() = 0;

        virtual ~Node() = default;

        std::string m_name;
    protected:
        Graph &m_graph = Graph::getInstance();
    };

    class Parameter : public Node {
    public:
        Parameter(std::string name, Shape shape) : Node(name), m_data(new Tensor(shape)),
                                                   m_gradient(new Tensor(shape)),
                                                   m_requires_grad(true) {}

        inline Tensor &Data() const {
            return *m_data;
        }

        inline Tensor &Gradient() const {
            return *m_gradient;
        }

        inline Tensor *MutuableData() {
            return m_data.get();
        }

        inline Tensor *MutuableGradient() {
            return m_gradient.get();
        }

        void Forward() override;

        void Backward() override;

    private:
        std::unique_ptr<Tensor> m_data;
        std::unique_ptr<Tensor> m_gradient;
        bool m_requires_grad;
    };

    class OperatorNode : public Node {
    public:
        explicit OperatorNode(std::string &name) : Node(name) {}

        template<typename ...Args>
        std::unique_ptr<Parameter> operator()(Parameter *param, Args &&...args) {
            printf("[OperatorNode::operator()]param name: %s\n", param->m_name.c_str());
            AddChildToGraph(param);
            return operator()(args...);
        }

        std::unique_ptr<Parameter> operator()(Parameter *param) {
            printf("[OperatorNode::operator()]param name: %s\n", param->m_name.c_str());
            AddChildToGraph(param);

            std::unique_ptr<Parameter> out_param(new Parameter(m_name + "_out", GetOutShape()));
            AddFatherToGraph(out_param.get());
            return out_param;
        }

        virtual Shape GetOutShape() = 0;

        virtual ~OperatorNode() = default;

    protected:
        std::vector<Parameter *> m_in_parameters;
        std::vector<Parameter *> m_out_parameters;

    private:
        void AddChildToGraph(Parameter *child) {
            m_graph.AddNode(this, child);
            m_in_parameters.push_back(child);
        }

        void AddFatherToGraph(Parameter *father) {
            m_graph.AddNode(father, this);
            m_out_parameters.push_back(father);
        }
    };
}
#endif //LIGHTLR_NODE_H
