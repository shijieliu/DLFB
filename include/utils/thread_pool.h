//
// Created by 刘仕杰 on 2020/2/2.
//

#ifndef LIGHTLR_THREAD_POOL_H
#define LIGHTLR_THREAD_POOL_H

#include <thread>
#include <condition_variable>
#include <queue>

namespace lr {
    class ThreadPool {
        std::mutex _mu;
        std::condition_variable _condition;
        std::queue<std::function<void()>> _queue;
        std::vector<std::thread> _thread_pool;
        size_t _num;
        bool stop;

        void init();

    public:
        explicit ThreadPool(size_t num);

        ThreadPool() = delete;

        ThreadPool(const ThreadPool &) = delete;

        ThreadPool &operator=(const ThreadPool &) = delete;

        ThreadPool(const ThreadPool &&) = delete;

        ThreadPool &operator=(const ThreadPool &&) = delete;

        ~ThreadPool();

        template<typename F, typename... Args>
        auto submit(F &&func, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>;

        void wait();

        static ThreadPool &Instance();

    };
}
#endif //LIGHTLR_THREAD_POOL_H
