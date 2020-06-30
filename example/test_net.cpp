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

        std::string tmp{};
        current_socket.RecvStr(&tmp);
        LOG_INFO("recv: %s\n", tmp.c_str());
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
        tcpSocket.SendStr("hello world");
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