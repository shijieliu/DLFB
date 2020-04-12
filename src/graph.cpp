//
// Created by 刘仕杰 on 2020/4/11.
//

#include <vector>
#include "graph.h"
#include "node.h"
#include <queue>
#include <deque>
namespace lr {
    std::vector<OperatorNode *> Graph::Transpose(Parameter *dst) {

        return std::vector<OperatorNode *>();
    }

    std::vector<OperatorNode *> Graph::TopologicalSort(Parameter *dst) {
        printf("[Graph::TopologicalSort]Graph node num %d\n", m_graph.size());
        std::unordered_map<std::string, int> dist;
        std::queue<std::string> sort_queue;
        std::vector<OperatorNode *> sort_res;
        sort_res.reserve(m_operator_map.size());

        for(auto iter_graph = m_graph.begin(); iter_graph != m_graph.end(); ++iter_graph){
            const std::string& child_name = iter_graph->first;
            if(dist.find(child_name) == dist.end()){
                dist[child_name] = 0;
            }
            const std::string& father_name = iter_graph->second;
            if(dist.find(father_name) == dist.end()){
                dist[father_name] = 1;
            }else{
                dist[father_name] += 1;
            }
        }

        for(auto iter_dist = dist.begin(); iter_dist != dist.end(); ++iter_dist){
            if(iter_dist->second == 0){
                sort_queue.push(iter_dist->first);
            }
        }

        while(!sort_queue.empty() && dist[dst->m_name] > 0){
            std::string current_name = sort_queue.front();
            sort_queue.pop();
            auto father_range = m_graph.equal_range(current_name);
            for(auto father = father_range.first; father != father_range.second; ++father){
                std::string father_name = father->second;
                dist[father_name]--;
                if(dist[father_name] == 0){
                    sort_queue.push(father_name);
                    if(m_operator_map.find(father_name) != m_operator_map.end()){
                        sort_res.push_back(m_operator_map[father_name]);
                    }
                }
            }
        }
        return sort_res;
    }

    void Graph::AddNode(OperatorNode *father, Parameter *child) {
        m_graph.insert(std::make_pair(child->m_name, father->m_name));
        printf("[Graph::AddNode]Graph father node: %s, child node: %s\n", father->m_name.c_str(), child->m_name.c_str());
        if(m_operator_map.find(father->m_name) == m_operator_map.end()){
            m_operator_map[father->m_name]= father;
        }
        if(m_parameter_map.find(child->m_name) == m_parameter_map.end()){
            m_parameter_map[child->m_name] = child;
        }
    }

    void Graph::AddNode(Parameter *father, OperatorNode *child) {
        m_graph.insert(std::make_pair(child->m_name, father->m_name));
        printf("[Graph::AddNode]Graph father node: %s, child node: %s\n", father->m_name.c_str(), child->m_name.c_str());

        if(m_operator_map.find(child->m_name) == m_operator_map.end()){
            m_operator_map[child->m_name] = child;
        }
        if(m_parameter_map.find(father->m_name) == m_parameter_map.end()){
            m_parameter_map[father->m_name] = father;
        }
    }

}