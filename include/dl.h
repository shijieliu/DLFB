#ifndef DL_H
#define DL_H
#include "core/graph.h"

namespace dl{
    inline void Init(){
        Graph::GetInstance();
    }

    inline GraphExecutor Compile(std::initializer_list<Node*> dst){
        Graph* graph = Graph::GetInstance();
        return graph->compile(dst);
    }
}
#endif