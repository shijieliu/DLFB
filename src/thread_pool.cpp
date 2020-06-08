//
// Created by 刘仕杰 on 2020/2/2.
//

#include "utils/thread_pool.h"
#include <future>

namespace dl {
    ThreadPool::ThreadPool(size_t num) : _num(num) {
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
        for (const auto &t: _thread_pool) {
            t.join();
        }
    }

    void ThreadPool::init() {
        if (!_thread_pool.empty())
            return;
        for (size_t i = 0; i < _num; ++i) {
            _thread_pool.emplace_back([this]() {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lk(this->_mu);
                        this->_condition.wait(lk, [this]() {
                            return !this->_queue.empty() || !this->stop;
                        });
                        if (this->stop && this->_queue.empty())
                            return;
                        task = std::move(this->_queue.front());
                        this->_queue.pop();
                    }
                    task();
                }
            });
        }
    }

    auto ThreadPool::submit(F &&func, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using ret_typ = std::future<typename std::result_of<F(Args...)>::type>;
        auto task = std::make_shared<std::packaged_task<F(Args...)> >(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );//使用 shared_ptr 来避免移动
        ret_typ future = task->get_future();
        {
            std::unique_lock<std::mutex> lk(_mu);
            _queue.emplace([task]() {
                (*task)();
            });
        }

        _condition.notify_one();
        return future;
//        using ret_typ = std::future<typename std::result_of<F(Args...)>::type>;
//        std::packaged_task<F(Args...)> task(
//                std::bind(std::forward<F>(f), std::forward<Args>(args)...)
//        );
//        ret_typ future = task.get_future();
//        {
//            std::unique_lock<std::mutex> lk(_mu);
//            _queue.emplace(std::move(task)); // 此时 queue 里面的对象不是 std::function
//        }
//
//        _condition.notify_one();
//        return future;

    }

    static ThreadPool &ThreadPool::Instance() {
        static ThreadPool threadpool(std::thread::hardware_concurrency());
        return threadpool;
    }

}