//
// Created by 刘仕杰 on 2020/2/2.
//

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <thread>
#include <condition_variable>
#include <future>
#include <queue>

namespace dl {
    class ThreadPool {


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

    private:
        std::mutex _mu;
        std::condition_variable _condition;
        std::queue<std::function<void()>> _queue;
        std::vector<std::thread> _thread_pool;
        size_t _num;
        bool stop;

        void init();
    };
}
#endif //THREAD_POOL_H
