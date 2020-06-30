/*
 * @Author: your name
 * @Date: 2020-06-20 10:10:29
 * @LastEditTime: 2020-06-20 10:12:49
 * @LastEditors: your name
 * @Description: In User Settings Edit
 * @FilePath: /LightLR/include/dist/comm.h
 */
/*
 * @Author: liushijie
 * @Date: 2020-06-18 14:13:15
 * @Last Modified by: liushijie
 * @Last Modified time: 2020-06-18 14:23:47
 */
#pragma once

#include "dist/socket.h"
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

namespace dl {
namespace dist {
struct Message {};
class Comm {
  public:
    Comm() {
        mReceiverThread = std::unique_ptr<std::thread>(
            new std::thread(&Comm::receiving, this));
        mHeartBeatThread = std::unique_ptr<std::thread>(
            new std::thread(&Comm::heartbeat, this));
    }
    ~Comm() { mReceiverThread->join(); }

  private:
    void receiving() {
        while (!mStop) {
        }
    }

    void heartbeat() {
        while (!mStop) {
        }
    }

    std::unordered_map<std::string, int> mConnectedNodes;
    std::atomic<bool>            mStop{true};
    std::unique_ptr<std::thread> mReceiverThread;
    std::unique_ptr<std::thread> mHeartBeatThread;
};

class SocketComm : public Comm {};

/**
 * @description: init distributed mode
 * @param {type} root port
 * @return:
 */
void Init(const char *ip, int port, int rank, int worldsize) {}
void AllReduce();
}
}
