//
// Created by 刘仕杰 on 2020/4/11.
//

#include <vector>
#include "core/graph.h"
#include "core/node.h"
#include <queue>
#include <deque>

namespace dl
{


void GraphExecutor::forward(std::unordered_map<Node*, Tensor>& input_data){
    for(auto& datapack: input_data){

    }
}

void GraphExecutor::backward(){

}

void Graph::transpose() {
    for (auto &node_pair: mGraph) {
        mTransposeGraph.insert(std::make_pair(node_pair.second, node_pair.first));
    }
}

GraphExecutor Graph::compile(std::initializer_list<Node*>& end_nodes_){
    std::unordered_set<std::string> end_nodes;
    for(auto end_node : end_nodes_){
        end_nodes.insert(end_node->mName);
    }
    transpose();
    return GraphExecutor(topologicalSort(mGraph, end_nodes), topologicalSort(mTransposeGraph, end_nodes));
}

std::vector<Node *> Graph::topologicalSort(const GraphRelationMap &graph, const std::unordered_set<std::string>& dst) const {
    LOG_INFO("Graph node num %lu\n", graph.size());
    std::unordered_map<std::string, int> dist;
    std::queue<std::string> sort_queue;
    std::vector<Node *> sort_res;
    sort_res.reserve(graph.size());

    auto get_prenode_name = [](GraphRelationMap::const_iterator& pack){
        return pack->first;
    };
    auto get_child_name = [](GraphRelationMap::const_iterator& pack){
        return pack->second;
    };

    auto get_node_name = [](const GraphAllocMap::value_type& pack) -> const std::string& {
        return pack.first;
    };

    for(const auto& node: mNodeMap){
        const std::string &node_name = get_node_name(node);
        if(graph.find(node_name) == graph.end()){
            dist[node_name] = 0;
            sort_queue.push(get_node_name(node));
        }else{
            const auto& pre_nodes_iter = graph.equal_range(node_name);
            while(pre_nodes_iter.first != pre_nodes_iter.second){
                dist[node_name] += 1;
            }
        }
    }

    while (!sort_queue.empty()) {
        std::string current_name = sort_queue.front();
        sort_queue.pop();
        auto prenodes_range = graph.equal_range(current_name);
        for (auto prenode_iterator = prenodes_range.first; prenode_iterator != prenodes_range.second; ++prenode_iterator) {
            const std::string& prenode_name = get_prenode_name(prenode_iterator);
            dist[prenode_name]--;
            if(dst.find(prenode_name) == dst.end()){
                continue;
            }
            if (dist[prenode_name] == 0) {
                sort_queue.push(prenode_name);
            }
        }
    }
    return sort_res;
}
} // namespace dl