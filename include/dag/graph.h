//
// Created by 刘仕杰 on 2020/4/6.
//
#pragma once

#include "dag/node.h"
#include "thread_pool.h"
#include "utils.h"
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace dl {
class DataNode;

// run the graph
class GraphExecutor {
  public:
    GraphExecutor(const std::vector<OperatorNodeBase *> &fwd,
                  const std::vector<OperatorNodeBase *>  bck)
        : fwd_nodes(fwd)
        , bck_nodes(bck) {}
    void forward(
        std::initializer_list<std::pair<DataNode *, const Tensor&>> initial_list);
    void backward(std::initializer_list<std::pair<DataNode *, Tensor>> losses);

  private:
    std::vector<OperatorNodeBase *> fwd_nodes;
    std::vector<OperatorNodeBase *> bck_nodes;
    ThreadPool &                    pool = ThreadPool::Instance();
};

class Graph {
  public:
    static Graph &GetInstance() {
        static Graph m_instance;
        return m_instance;
    }

    Graph(const Graph &other) = delete;
    Graph &operator=(const Graph &other) = delete;
    ~Graph();

    // store nodes and their relation
    template <typename Opr, typename... Params>
    DataNode *add(std::vector<DataNode *> &pre_nodes, Params... params) {
        int64_t opr_node_uid = mOprNodes.size();
        mOprNodes.emplace_back(
            std::unique_ptr<Opr>(new Opr(opr_node_uid, params...)));
        auto &opr_node = mOprNodes.back();
        opr_node->setInNodes(pre_nodes.begin(), pre_nodes.end());
        for (auto pre_node : pre_nodes) {
            if (mDataNodes.find(pre_node->mUID) == mDataNodes.end()) {
                LOG_ERROR("pre_node %zu not in graph", pre_node->mUID);
                assert(0);
            }
            auto pre_opr_it = mDataOprRelationGraph.find(pre_node->mUID);
            if (pre_opr_it == mDataOprRelationGraph.end()) {
                continue;
            }
            int64_t pre_opr_node_uid   = pre_opr_it->second;
            auto   pre_opr_node_range = mGraph.equal_range(opr_node_uid);
            bool   has_inserted       = false;
            for (auto relation_it = pre_opr_node_range.first;
                 relation_it != pre_opr_node_range.second; ++relation_it) {
                if (relation_it->second == pre_opr_node_uid) {
                    has_inserted = true;
                    break;
                }
            }
            if (!has_inserted) {
                mGraph.insert(std::make_pair(opr_node_uid, pre_opr_node_uid));
            }
        }
        int64_t end_node_uid = mDataNodes.size();
        mDataNodes.insert(std::make_pair(
            end_node_uid,
            std::unique_ptr<DataNode>(new DataNode(
                end_node_uid, opr_node->applyInferenceShape(), true))));
        opr_node->setOutNodes(mDataNodes[end_node_uid].get());

        mDataOprRelationGraph.insert(
            std::make_pair(end_node_uid, opr_node_uid));
        return mDataNodes[end_node_uid].get();
    }

    DataNode *add(const Shape &shape, bool is_data_provider);

    GraphExecutor compile(std::initializer_list<DataNode *> end_nodes_);

    inline void train() {
        for (auto &opr_node_ptr : mOprNodes) {
            opr_node_ptr->train();
        }
    }

    inline void eval() {
        for (auto &opr_node_ptr : mOprNodes) {
            opr_node_ptr->eval();
        }
    }

  private:
    using GraphRelationMap = std::unordered_multimap<int64_t, int64_t>;
    GraphRelationMap mGraph;
    GraphRelationMap mTransposeGraph;
    std::unordered_map<int64_t, int64_t>                    mDataOprRelationGraph;
    std::unordered_map<int64_t, std::unique_ptr<DataNode>> mDataNodes;
    std::vector<std::unique_ptr<OperatorNodeBase>> mOprNodes;
    std::vector<int64_t>                            mDataProviders;
    Graph();

    void transpose();
    std::unordered_map<int64_t, int>
        prepareTopologicalSort(const GraphRelationMap &graph,
                               std::queue<int64_t> *    start_queue) const;
    std::vector<OperatorNodeBase *>
        topologicalSort(const GraphRelationMap &          graph,
                        const GraphRelationMap &          transpose_graph,
                        const std::unordered_set<int64_t> &dst) const;
};

template <typename Opr, typename... Params>
DataNode *CreateNode(std::initializer_list<DataNode *> pre_nodes_,
                     Params... params) {
    Graph &                 graph = Graph::GetInstance();
    std::vector<DataNode *> pre_nodes(pre_nodes_.begin(), pre_nodes_.end());
    return graph.add<Opr>(pre_nodes, params...);
}

DataNode *CreateNode(const Shape shape, bool requires_grad=false) {
    LOG_INFO("Create DataNode");
    return Graph::GetInstance().add(shape, requires_grad);
}

void GraphExecutor::forward(
    std::initializer_list<std::pair<DataNode *, const Tensor&>> initial_list) {
    for (auto &datapack : initial_list) {
        auto  datanode_ptr        = datapack.first;
        auto &t                   = datapack.second;
        if(datanode_ptr->tensor()->shape() != t.shape()){
            LOG_ERROR("\n\tdatanode shape not match\n\t\tdatanode shape:%s\n\t\ttensor shape:%s", FormatShape(datanode_ptr->tensor()->shape()).c_str(), FormatShape(t.shape()).c_str());
        }
        *(datanode_ptr->tensor()) = t;
    }
    for (OperatorNodeBase *opr : fwd_nodes) {
        opr->applyForward();
    }
    // pool.submit
}

void GraphExecutor::backward(
    std::initializer_list<std::pair<DataNode *, Tensor>> losses) {
    for (auto &losspack : losses) {
        DataNode *datanode_ptr  = losspack.first;
        auto &    t             = losspack.second;
        *(datanode_ptr->grad()) = t;
    }
    for (OperatorNodeBase *opr : bck_nodes) {
        opr->applyBackward();
    }
}

void Graph::transpose() {
    if (mTransposeGraph.size() > 0) {
        return;
    }
    for (auto &node_pair : mGraph) {
        mTransposeGraph.insert(
            std::make_pair(node_pair.second, node_pair.first));
    }
    LOG_INFO("Transpose graph node num %lu", mTransposeGraph.size());
}

Graph::Graph() {}
Graph::~Graph() = default;

GraphExecutor Graph::compile(std::initializer_list<DataNode *> end_nodes_) {
    std::unordered_set<int64_t> end_nodes;
    for (auto end_node : end_nodes_) {
        int64_t opr_node_uid = mDataOprRelationGraph[end_node->mUID];
        end_nodes.insert(opr_node_uid);
    }
    transpose();
    return GraphExecutor(topologicalSort(mGraph, mTransposeGraph, end_nodes),
                         topologicalSort(mTransposeGraph, mGraph, end_nodes));
}

DataNode *Graph::add(const Shape &shape, bool is_data_provider) {
    int64_t data_node_uid = mDataNodes.size();
    mDataNodes.insert(std::make_pair(
        data_node_uid,
        std::unique_ptr<DataNode>(new DataNode(data_node_uid, shape, true))));
    if (is_data_provider) {
        mDataProviders.push_back(data_node_uid);
    }
    return mDataNodes[data_node_uid].get();
}

std::unordered_map<int64_t, int>
    Graph::prepareTopologicalSort(const GraphRelationMap &graph,
                                  std::queue<int64_t> *    start_queue) const {
    std::unordered_map<int64_t, int> dist;
    for (int64_t uid = 0; uid < mOprNodes.size(); ++uid) {
        if (graph.find(uid) == graph.end()) {
            start_queue->push(uid);
        }
        dist[uid] = graph.count(uid);
    }
    return dist;
}

std::vector<OperatorNodeBase *>
    Graph::topologicalSort(const GraphRelationMap &          graph,
                           const GraphRelationMap &          transpose_graph,
                           const std::unordered_set<int64_t> &dst) const {
    std::queue<int64_t>              sort_queue;
    std::vector<OperatorNodeBase *> sort_res;
    sort_res.reserve(graph.size());

    // 1. calc in and out of nodes
    std::unordered_map<int64_t, int> dist =
        prepareTopologicalSort(graph, &sort_queue);

    // 2. topological sort

    while (!sort_queue.empty()) {
        int64_t current_uid     = sort_queue.front();
        auto * current_opr_ptr = mOprNodes[current_uid].get();
        sort_res.push_back(current_opr_ptr);
        sort_queue.pop();
        if (dst.find(current_uid) != dst.end()) {
            continue;
        }

        const auto next_opr_range = transpose_graph.equal_range(current_uid);
        for (auto next_opr_it = next_opr_range.first;
             next_opr_it != next_opr_range.second; ++next_opr_it) {
            int64_t next_uid = next_opr_it->second;
            --dist[next_uid];
            if (dist[next_uid] == 0) {
                sort_queue.push(next_uid);
            }
        }
    }
    LOG_INFO("topological sort done!");
    return sort_res;
}

} // namespace dl