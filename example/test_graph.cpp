#include "dl.h"

int main(){
    dl::Node* src = dl::CreateNode<dl::Node>({1}, {});
    
    auto func = dl::Compile({src});
    // func.forward({"":Tensor()});
    // func.backward();
}