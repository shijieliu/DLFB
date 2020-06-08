//
// Created by 刘仕杰 on 2020/4/6.
//

#ifndef GraphPH_H
#define GraphPH_H

#include <unordered_map>
#include <string>
#include <memory>
#include <vector>
#include <unordered_set>
#include "core/tensor.h"

namespace dl
{
class Node;

// run the graph
class GraphExecutor
{
public:
    GraphExecutor(const std::vector<Node *> &fwd, const std::vector<Node *> bck) : fwd_nodes(fwd), bck_nodes(bck) {}
    void forward(std::unordered_map<Node *, Tensor> &data);
    void backward();

private:
    std::vector<Node *> fwd_nodes;
    std::vector<Node *> bck_nodes;
};

class Graph
{
public:
    static Graph *GetInstance()
    {
        static Graph m_instance;
        return &m_instance;
    }
    Graph(const Graph &other) = delete;

    Graph &operator=(const Graph &other) = delete;
    ~Graph() = default;

    // store nodes and their relation
    template <typename Ops>
    Ops *add(Shape shape, std::initializer_list<Node *> &pre_nodes)
    {
        size_t idx = mCount[Ops::Type]++;
        std::string name = Ops::Type + std::to_string(idx);
        mNodeMap.insert(std::make_pair(name, std::unique_ptr<Ops>(new Ops(name, shape))));
        std::unique_ptr<Ops> &res = mNodeMap[name];
        if(pre_nodes.size() > 0){
            insert(name, pre_nodes.begin(), pre_nodes.end());
        }
        return mNodeMap[name].get();
    }

    template <typename ForwardIt>
    void insert(std::string &name, const ForwardIt& first, const ForwardIt& last){
        auto it = first;
        while(it != last){
            mGraph.insert(std::make_pair((*it)->mName, name));
        }
    }
    
    std::vector<Node *> getFather(Node *cur);
    std::vector<Node *> getChildren(Node *cur);

    GraphExecutor compile(std::initializer_list<Node *> &end_nodes);

private:
    using GraphRelationMap = std::unordered_multimap<std::string, std::string>;
    GraphRelationMap mGraph;
    GraphRelationMap mTransposeGraph;

    using GraphAllocMap = std::unordered_map<std::string, std::unique_ptr<Node>>;
    GraphAllocMap mNodeMap;
    
    using GraphIdxMap = std::unordered_map<std::string, size_t>;
    GraphIdxMap mCount;

    Graph() {}
    void transpose();
    std::vector<Node *> topologicalSort(const GraphRelationMap &graph, const std::unordered_set<std::string>& dst) const;
    
};

template <typename Ops>
Ops *CreateNode(std::initializer_list<size_t> shape, std::initializer_list<Node *> pre_nodes)
{
    Graph *graph = Graph::GetInstance();
    return graph->add<Ops>(shape, pre_nodes);
}

// class Graph {
// public:
//     static Graph &GetInstance() {
//         static Graph m_instance;
//         return m_instance;
//     }

//     Graph(const Graph &other) = delete;

//     Graph &operator=(const Graph &other) = delete;

//     void AddNode(OperatorNode *father, Parameter *child);

//     void AddNode(Parameter *father, OperatorNode *child);

//     std::vector<OperatorNode *> ForwardSort(Parameter *dst);

//     std::vector<OperatorNode *> BackwardSort(Parameter *dst);

// private:
//     void Transpose();

//     std::vector<OperatorNode *> TopologicalSort(const std::unordered_multimap<std::string, std::string>& graph, Parameter *dst);

//     Graph(): m_need_transpose(true), m_fwd_need_build(true), m_bck_need_build(true) {}

//     std::unordered_multimap<std::string, std::string> mGraph;
//     std::unordered_multimap<std::string, std::string> mTransposeGraph;

//     std::unordered_map<std::string, Parameter *> m_parameter_map;
//     std::unordered_map<std::string, OperatorNode *> m_operator_map;

//     std::vector<OperatorNode *> m_fwd_nodes;
//     std::vector<OperatorNode *> m_bck_nodes;

//     bool m_need_transpose;
//     bool m_fwd_need_build;
//     bool m_bck_need_build;
// };

} // namespace dl
#endif //GraphPH_H
