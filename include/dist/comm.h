/*
 * @Author: liushijie
 * @Date: 2020-06-18 14:13:15
 * @LastEditTime: 2020-07-11 15:24:00
 * @LastEditors: liushijie
 * @Description:
 * @FilePath: /LightLR/include/dist/comm.h
 */
#pragma once

#include "dag/graph.h"
#include "dist/socket.h"
#include "queue.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace dl {
namespace dist {

class Comm {
  public:
    Comm();

    ~Comm() {
        mProcessingThread->join();
        mReceiverThread->join();
        mHeartBeatThread->join();
    }

    void send(DataNode *node, int offset, Message::Control control);

    void barrier();

  protected:
    void scheduler();
    void receiving();
    void processing();
    void heartbeat();

    std::unordered_map<std::string, int> mConnectedNodes;
    bool                         mNextReady;
    std::unique_ptr<std::thread> mReceiverThread;
    std::unique_ptr<std::thread> mProcessingThread;
    std::unique_ptr<std::thread> mHeartBeatThread;
    std::vector<TcpSocket>       mNodes;
    std::condition_variable      mBarrierCond;
    std::mutex                   mBarrierMutex;
    int                          mRank;
    int                          mWorldSize;
    ThreadsafeQueue<Message>     mQueue;
    TcpSocket                    mNextNodeSocket;
    TcpSocket                    mRootNodeSocket;
};

class SocketComm : public Comm {};

Comm::Comm()
    : mRank(std::atoi(std::getenv("RANK")))
    , mWorldSize(std::atoi(std::getenv("WORLD_SIZE"))) {

    TcpSocket sock;
    if (mRank == 0) {
        mHeartBeatThread = std::unique_ptr<std::thread>(
            new std::thread(&Comm::scheduler, this));
    } else {
        mHeartBeatThread = std::unique_ptr<std::thread>(
            new std::thread(&Comm::heartbeat, this));
    }

    int next_rank = (mRank + 1) % mWorldSize;

    mReceiverThread = std::unique_ptr<std::thread>(new std::thread(
        &Comm::receiving, this)); // 每个节点上用来做 allreduce 的 thread
    mProcessingThread =
        std::unique_ptr<std::thread>(new std::thread(&Comm::processing, this));
}

void Comm::scheduler() {
    // rank 0
    // 1. 建立监听端口
    // 2. 定时发送 heart beat
    Addrinfo  addrinfo("0.0.0.0", std::atoi(std::getenv("PORT")));
    TcpSocket sock;
    sock.Create();
    sock.Bind(addrinfo);
    sock.Listen();
    for (int i = 0; i < mWorldSize; ++i) {
        TcpSocket other_node = sock.Accept();
        {
            std::lock_guard<std::mutex> lk(mBarrierMutex);
            mNodes.push_back(other_node);
        }
    }
    mBarrierCond.notify_all();
    while (true) {
        // barrier
        for (int i = 0; i < mNodes.size(); ++i) {
            Message msg;
            msg.control = Message::Control::Heartbeat;
            mNodes[i].SendMessage(msg);
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

/**
 * @description: start to recv allreduce data and put it into queue
 * @param {type}
 * @return:
 */
void Comm::receiving() {
    Addrinfo  current_rank_addrinfo("0.0.0.0", 20);
    TcpSocket allreduce_sock;
    allreduce_sock.Create();
    allreduce_sock.Bind(current_rank_addrinfo);
    allreduce_sock.Listen();

    TcpSocket next_sock = allreduce_sock.Accept();
    {
        std::lock_guard<std::mutex> lk(mBarrierMutex);
        mNextReady = true;
    }
    mBarrierCond.notify_all();
    while (true) {
        Message recv;
        next_sock.RecvMessage(&recv);
        mQueue.push(recv);
    }
}

void Comm::processing() {
    {
        std::unique_lock<std::mutex> lk(mBarrierMutex);
        mBarrierCond.wait(lk, [this]() {
            if (mRank == 0) {
                return mNodes.size() == mWorldSize - 1 && mNextReady;
            }
            return mNextReady;
        });
        lk.unlock();
    }
    while (true) {
        Message recv;
        mQueue.waitAndPop(&recv);
        assert(recv.dst_rank == mRank);
        switch (recv.control) {
        case Message::Control::Report: {
            if (mRank == 0) {
                LOG_ERROR("rank 0 should not recv [Message::Control::Report]");
            }
            Message msg;
            msg.control  = Message::Control::Report;
            msg.dst_rank = 0;
            msg.src_rank = mRank;
            // msg.data =
            mRootNodeSocket.SendMessage(msg);
        }

            continue;
        case Message::Control::Heartbeat: {
            if (mRank == 0) {
                LOG_ERROR(
                    "rank o should not recv [Message::Control::Heartbeat]");
            } else {
                if (mNextNodeSocket.isinvalid()) {
                    std::string next_node_ip;
                    int         next_node_port;
                    Addrinfo    next_node_addr(next_node_ip.c_str(),
                                            next_node_port);
                    mNextNodeSocket.Create();
                    mNextNodeSocket.Connect(next_node_addr);
                }
            }
        }

            continue;
        case Message::Control::ReduceSum: {
            // Graph& graph = Graph::GetInstance();
            // DataNode* dst_node = graph.getParam(msg.uid);

            // memcpy(dst_node->grad() + msg.offset,
            // reinterpret_cast<float*>(msg.data), msg.size);
        }

            continue;
        case Message::Control::Gather:

            continue;
        default:
            LOG_ERROR("should not reach");
        }
    }
}

void Comm::heartbeat() {
    Addrinfo addrinfo(std::getenv("ROOT_IP"),
                      std::atoi(std::getenv("ROOT_PORT")));

    mRootNodeSocket.Create();
    mRootNodeSocket.Connect(addrinfo);

    while (true) {
        Message recv;
        mRootNodeSocket.RecvMessage(&recv);
        mQueue.push(recv);
    }
}

void Comm::send(DataNode *node, int offset, Message::Control control) {
    assert(control == Message::Control::ReduceSum ||
           control == Message::Control::Gather);
    const Tensor *grad      = node->grad();
    int           step_size = std::max(64, (int) grad->size() / mWorldSize);

    int start = step_size * offset;
    if (start > grad->size()) {
        return;
    }
    int end = std::min((int) grad->size(), start + offset);
    while (start < end) {
        Message msg;
        msg.control  = control;
        msg.uid      = node->mUID;
        msg.dst_rank = (mRank + 1) % mWorldSize;
        msg.src_rank = mRank;
        msg.offset   = start;
        msg.size     = std::min(64, end - start);
        memcpy(msg.data, grad->data() + start, sizeof(float) * msg.size);
        mNextNodeSocket.SendMessage(msg);
        start += msg.size;
    }
}

void Comm::barrier() {
    if (mRank == 0) return;
    Message msg;
    msg.control = Message::Control::Barrier;
    mRootNodeSocket.SendMessage(msg);
    {
        std::unique_lock<std::mutex> lk(mBarrierMutex);
        mBarrierCond.wait(lk, []() {
            return true;
        });
        lk.unlock();
    }
}

void AllReduce(std::vector<DataNode *> datanodes, Comm *comm) {
    // for (int step = 0; step < worldsize; ++step) {
    //     for (DataNode *node : datanodes) {
    //         comm->sendAsync(node, step, mrank, control);
    //         // add fault tolanrence
    //     }
    //     int error_code = comm->barrier();
    // }

    // for (int step = 0; step < worldsize; ++step) {
    //     for (DataNode *node : datanodes) {
    //         comm->sendAsync(node, 0, mrank, control);
    //     }
    //     int error_code = comm->barrier();
    // }
}
}
}
