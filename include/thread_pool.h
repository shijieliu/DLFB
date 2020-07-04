//
// Created by 刘仕杰 on 2020/2/2.
//
#pragma once

#include "macro.h"
#include "queue.h"
#include <condition_variable>
#include <future>
#include <queue>
#include <thread>

namespace dl {
class ThreadPool {

  public:
    explicit ThreadPool(int num);

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
    ThreadsafeQueue<std::function<void()>> _queue;
    std::vector<std::thread>               _thread_pool;
    int                                    _num;
    void                                   init();
};

template <typename F, typename... Args>
std::future<typename std::result_of<F(Args...)>::type>
    ThreadPool::submit(F &&func, Args &&... args) {
    using ret_typ = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<ret_typ()>>(
        std::bind(std::forward<F>(func),
                  std::forward<Args>(args)...)); //使用 shared_ptr 来避免移动

    std::future<ret_typ> future = task->get_future();
    _queue.push([task]() { (*task)(); });

    return future;
}
ThreadPool::ThreadPool(int num)
    : _num(num) {
    init();
}

ThreadPool::~ThreadPool() { wait(); }

void ThreadPool::wait() {
    _queue.close();

    for (auto &t : _thread_pool) {
        t.join();
    }
    _thread_pool.clear();
}

void ThreadPool::init() {
    if (!_thread_pool.empty()) return;
    for (int i = 0; i < _num; ++i) {
        _thread_pool.emplace_back([this, i]() {
            for (;;) {
                std::function<void()> task;
                this->_queue.waitAndPop(&task);
                if (!task) return;
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