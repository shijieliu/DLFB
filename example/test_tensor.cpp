#include "core/tensor.h"
#include <vector>

void display_tensor(const dl::Tensor& t){
    std::string shape_info("(");
    for(size_t i = 0; i < t.shape().size(); ++i){
        shape_info.append(std::to_string(t.shape()[i]));
        shape_info.append(",");
    }
    shape_info.append(")");
    LOG_INFO("[shape]%s", shape_info.c_str());

    std::string value_info;
    for(size_t i = 0; i < t.size(); ++i){
        value_info.append(std::to_string(t[i]));
        value_info.append("\t");
    }
    LOG_INFO("[value]%s", value_info.c_str());

}
int main(){
    dl::Shape shape{1};
    std::vector<float> d{1};

    dl::Tensor t(d, shape);
    display_tensor(t);
    dl::Tensor r = dl::Add(t, t);
    display_tensor(r);
}