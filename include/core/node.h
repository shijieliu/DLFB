//
// Created by 刘仕杰 on 2020/4/6.
//

#ifndef NODE_H
#define NODE_H

#include <vector>
#include <string>
#include <memory>
#include "core/tensor.h"
#include "macro.h"

namespace dl
{
class Graph;

class Node
{
public:
    explicit Node(std::string& name): mName(name) {}
    Node(std::string& name, Shape& shape): mName(name), mTensor(std::make_shared<Tensor>(shape)), mGradient(std::make_shared<Tensor>(shape)) {}
    
    virtual void forward();
    virtual void backward();
    void setTensor(Tensor *data);
    void setGrad(Tensor *data);
    virtual ~Node();
    std::string mName;
    static const std::string Type;

private:
    std::unique_ptr<Graph> mGraph;
    std::shared_ptr<Tensor> mTensor;
    std::shared_ptr<Tensor> mGradient;
};


// class Node {
// public:
//     explicit Node(std::string &name) : mName(name) {}

//     Node(const Node &) = delete;

//     Node &operator=(const Node &other) = delete;

//     virtual void Forward() = 0;

//     virtual void Backward() = 0;

//     virtual ~Node() = default;

//     std::string mName;
// protected:
//     Graph &mGraph = Graph::GetInstance();
// };

// class Parameter : public Node {
// public:
//     Parameter(std::string name, Shape shape) : Node(name), mData(new Tensor(shape)),
//                                                mGradient(new Tensor(shape)),
//                                                m_requires_grad(true) {}

//     inline Tensor &Data() const {
//         return *mData;
//     }

//     inline Tensor &Gradient() const {
//         return *mGradient;
//     }

//     inline Tensor *MutuableData() {
//         return mData.get();
//     }

//     inline Tensor *MutuableGradient() {
//         return mGradient.get();
//     }

//     void Forward() override;

//     void Backward() override;

// private:
//     std::unique_ptr<Tensor> mData;
//     std::unique_ptr<Tensor> mGradient;
//     bool m_requires_grad;
// };

// class OperatorNode : public Node {
// public:
//     explicit OperatorNode(std::string &name) : Node(name) {}

//     template<typename ...Args>
//     std::unique_ptr<Parameter> operator()(Parameter *param, Args &&...args) {
//         printf("[OperatorNode::operator()]param name: %s\n", param->mName.c_str());
//         AddChildToGraph(param);
//         return operator()(args...);
//     }

//     std::unique_ptr<Parameter> operator()(Parameter *param) {
//         printf("[OperatorNode::operator()]param name: %s\n", param->mName.c_str());
//         AddChildToGraph(param);

//         std::unique_ptr<Parameter> out_param(new Parameter(mName + "_out", GetOutShape()));
//         AddFatherToGraph(out_param.get());
//         return out_param;
//     }

//     virtual Shape GetOutShape() = 0;

//     virtual ~OperatorNode() = default;

// protected:
//     std::vector<Parameter *> m_in_parameters;
//     std::vector<Parameter *> m_out_parameters;

// private:
//     void AddChildToGraph(Parameter *child) {
//         mGraph.AddNode(this, child);
//         m_in_parameters.push_back(child);
//     }

//     void AddFatherToGraph(Parameter *father) {
//         mGraph.AddNode(father, this);
//         m_out_parameters.push_back(father);
//     }
// };
} // namespace dl
#endif //NODE_H
