//
// Created by 刘仕杰 on 2020/2/2.
//
#pragma once

#include "macro.h"
#include <condition_variable>
#include <future>
#include <queue>
#include <thread>

namespace dl {
class ThreadPool {

  public:
    explicit ThreadPool(int64_t num);

    ThreadPool() = delete;

    ThreadPool(const ThreadPool &) = delete;

    ThreadPool &operator=(const ThreadPool &) = delete;

    ThreadPool(const ThreadPool &&) = delete;

    ThreadPool &operator=(const ThreadPool &&) = delete;

    ~ThreadPool();

    template <typename F, typename... Args>
    std::future<typename std::result_of<F(Args...)>::type>
        submit(F &&func, Args &&... args);

    void wait();

    static ThreadPool &Instance();

  private:
    std::mutex                        _mu;
    std::condition_variable           _condition;
    std::queue<std::function<void()>> _queue;
    std::vector<std::thread>          _thread_pool;
    int64_t                            _num;
    bool                              stop;

    void init();
};

template <typename F, typename... Args>
std::future<typename std::result_of<F(Args...)>::type>
    ThreadPool::submit(F &&func, Args &&... args) {
    using ret_typ = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<ret_typ()>>(
        std::bind(std::forward<F>(func),
                  std::forward<Args>(args)...)); //使用 shared_ptr 来避免移动

    std::future<ret_typ> future = task->get_future();
    {
        std::unique_lock<std::mutex> lk(_mu);
        _queue.emplace([task]() { (*task)(); });
    }

    _condition.notify_one();
    return future;
}
ThreadPool::ThreadPool(int64_t num)
    : _num(num) {
    init();
}

ThreadPool::~ThreadPool() {
    wait();
}

void ThreadPool::wait() {
    {
        std::unique_lock<std::mutex> lk(_mu);
        stop = true;
    }
    _condition.notify_all();

    for (auto &t : _thread_pool) {
        t.join();
    }
    _thread_pool.clear();
}

void ThreadPool::init() {
    if (!_thread_pool.empty()) return;
    for (int64_t i = 0; i < _num; ++i) {
        _thread_pool.emplace_back([this, i]() {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lk(this->_mu);
                    this->_condition.wait(lk, [this]() {
                        return !this->_queue.empty() || this->stop;
                    });
                    if (this->stop && this->_queue.empty()) return;
                    task = std::move(this->_queue.front());
                    this->_queue.pop();
                }
                task();
            }
        });
    }
}

ThreadPool &ThreadPool::Instance() {
    static ThreadPool threadpool(std::thread::hardware_concurrency());
    return threadpool;
}
}