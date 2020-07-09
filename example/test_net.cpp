/*
 * @Author: liushijie
 * @Date: 2020-07-04 19:45:52
 * @LastEditTime: 2020-07-04 20:43:09
 * @LastEditors: liushijie
 * @Description: 
 * @FilePath: /LightLR/example/test_net.cpp
 */ 
#include "dist/message.h"
#include "dist/socket.h"
#include <chrono>
#include <cstring>
#include <thread>

void start_server(const char *ip, const char *port) {
    LOG_INFO("start server");
    dl::dist::Addrinfo  addrinfo(ip, atoi(port));
    dl::dist::TcpSocket socket;
    socket.Create();
    socket.Bind(addrinfo);
    socket.Listen();

    auto current_socket = socket.Accept();
    for (;;) {
        LOG_INFO("recving msg\n");

        dl::Message tmp;
        current_socket.RecvMessage(&tmp);
        LOG_INFO("recv: %d\n", tmp.control);
    }
}

void start_client(const char *ip, const char *port) {
    LOG_INFO("start client");
    dl::dist::Addrinfo  addrinfo = dl::dist::Addrinfo(ip, atoi(port));
    dl::dist::TcpSocket tcpSocket;
    tcpSocket.Create();
    tcpSocket.Connect(addrinfo);

    for (;;) {
        LOG_INFO("sending msg\n");
        dl::Message tmp;
        tcpSocket.SendMessage(tmp);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

int main(int argc, const char *argv[]) {
    assert(argc == 4); // type ip port
    const char *type = argv[1];
    const char *ip   = argv[2];
    const char *port = argv[3];
    LOG_INFO("args: %s, %s, %s", type, ip, port);
    if (std::strcmp(type, "server") == 0) {
        start_server(ip, port);
    } else {
        start_client(ip, port);
    }
}