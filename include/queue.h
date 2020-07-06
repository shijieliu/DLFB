/*
 * @Author: liushijie
 * @Date: 2020-07-04 16:30:33
 * @LastEditTime: 2020-07-06 17:10:22
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/queue.h
 */

#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>

namespace dl {

template <typename T> class ThreadsafeQueue {
  public:
    ThreadsafeQueue()
        : stop_(false) {}
    ~ThreadsafeQueue() {}

    void push(T new_value) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            queue_.push(std::move(new_value));
        }
        cond_.notify_all();
    }

    void waitAndPop(T *value) {
        std::unique_lock<std::mutex> lk(mu_);
        cond_.wait(lk, [this] { return !queue_.empty() || stop_; });
        if (stop_ && queue_.empty()) return;
        *value = std::move(queue_.front());
        queue_.pop();
        lk.unlock();
    }

    void close() {
        {
            std::lock_guard<std::mutex> lk(mu_);    
            stop_ = true;
        }
        cond_.notify_all();
    }

  private:
    bool                    stop_;
    mutable std::mutex      mu_;
    std::queue<T>           queue_;
    std::condition_variable cond_;
};
}
