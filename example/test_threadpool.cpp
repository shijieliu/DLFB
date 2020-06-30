#include "thread_pool.h"
#include <vector>

int get_result(int x) { return x; }

void test() {
    dl::ThreadPool &              threadpool = dl::ThreadPool::Instance();
    std::vector<std::future<int>> result;
    for (int i = 0; i < 10; ++i) {
        result.push_back(std::move(threadpool.submit(get_result, i)));
    }
    for (auto &r : result) {
        printf("result is %d\n", r.get());
    }
}

int main() { test(); }