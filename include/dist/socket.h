/*
 * @Author: liushijie
 * @Date: 2020-06-15 21:08:28
 * @Last Modified by: liushijie
 * @Last Modified time: 2020-06-15 21:12:16
 */
#pragma once
#include "dist/message.h"
#include "macro.h"
#include <arpa/inet.h>
#include <assert.h>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace dl {
namespace dist {

using SOCKET                = int;
const SOCKET INVALID_SOCKET = -1;

struct AddrinfoDetail {
    char *host = nullptr;
    int   port = -1;

    AddrinfoDetail(const char *host, int port)
        : host(const_cast<char *>(host))
        , port(port) {}
};

struct Addrinfo {
    sockaddr_in    m_addr;
    AddrinfoDetail m_addrinfoDetail;

    Addrinfo() = delete;
    explicit Addrinfo(const char *host, const int port)
        : m_addrinfoDetail(host, port) {
        addrinfo hints;
        memset(&hints, 0, sizeof(hints));
        hints.ai_family     = AF_UNSPEC;
        hints.ai_socktype   = SOCK_STREAM; // TCP stream sockets
        hints.ai_flags      = AI_PASSIVE;  // fill in my IP for me
        addrinfo *serveinfo = nullptr;

        int status = getaddrinfo(host, NULL, &hints, &serveinfo);
        if (status == 0 && serveinfo != nullptr) {
            memcpy(&m_addr, serveinfo->ai_addr, serveinfo->ai_addrlen);
            m_addr.sin_port = htons(port);
            freeaddrinfo(serveinfo);
            return;
        }
        LOG_ERROR("can not obtain addr %s\n", host);
    }

    ~Addrinfo() {}
};


class Socket {
  public:
    Socket()
        : m_socket(INVALID_SOCKET) {}
    Socket(SOCKET _socket)
        : m_socket(_socket) {}

    inline int Close() { return close(m_socket); }

    inline int  Shutdown() { return shutdown(m_socket, SHUT_RDWR); }
    inline bool isinvalid() { return m_socket == INVALID_SOCKET; }

    inline bool SetSocketBlockingEnabled(bool blocking) {
        int flags = fcntl(m_socket, F_GETFL, 0);
        if (flags == -1) return false;
        flags = blocking ? (flags & ~O_NONBLOCK) : (flags | O_NONBLOCK);
        return (fcntl(m_socket, F_SETFL, flags) == 0) ? true : false;
    }

    inline void Bind(const Addrinfo &addr) {
        LOG_INFO("bind to %s:%d\n", addr.m_addrinfoDetail.host,
                 addr.m_addrinfoDetail.port);
        if (-1 == bind(m_socket,
                       reinterpret_cast<const sockaddr *>(&addr.m_addr),
                       sizeof(addr.m_addr))) {
            LOG_ERROR("bind to %s:%d error\n", addr.m_addrinfoDetail.host,
                      addr.m_addrinfoDetail.port);
        }
    }

    inline void Listen(int backlog = 16) {
        LOG_INFO("listen\n");
        if (-1 == listen(m_socket, backlog)) {
            LOG_ERROR("listen error!\n");
        }
    }

    inline void Connect(const Addrinfo &addr) {
        LOG_INFO("connct to %s:%d\n", addr.m_addrinfoDetail.host,
                 addr.m_addrinfoDetail.port);
        if (-1 == connect(m_socket,
                          reinterpret_cast<const sockaddr *>(&addr.m_addr),
                          sizeof(addr.m_addr))) {
            Close();
            LOG_ERROR("connect to %s:%d error\n", addr.m_addrinfoDetail.host,
                      addr.m_addrinfoDetail.port);
        }
    }

    inline int Send(const void *data, int size, int flag = 0) {
        const char *buf = reinterpret_cast<const char *>(data);
        return send(m_socket, buf, size, flag);
    }

    inline int SendAll(const void *data, int len) {
        const char *buf   = reinterpret_cast<const char *>(data);
        int         ndown = 0;
        while (ndown < len) {
            int ret = Send(buf, len - ndown);

            if (ret == -1) {
                LOG_ERROR("Send msg wrong\n");
            }
            if (ret == 0) {
                return ndown;
            }
            buf += ret;
            ndown += ret;
        }
        return ndown;
    }

    inline void SendMessage(const Message &msg) {
        int len = sizeof(msg);
        assert(SendAll(&len, sizeof(len)) == sizeof(len));
        assert(SendAll(&msg, len) == len);
    }

    inline int Recv(void *data, int size, int flag = 0) {
        char *buf = reinterpret_cast<char *>(data);
        return recv(m_socket, buf, size, flag);
    }

    inline int RecvAll(void *data, int len) {
        char *buf   = reinterpret_cast<char *>(data);
        int   ndown = 0;

        while (ndown < len) {
            int ret = Recv(buf, len - ndown);
            if (ret == -1) {
                LOG_ERROR("Recv msg wrong\n");
            }
            if (ret == 0) {
                return ndown;
            }
            buf += ret;
            ndown += ret;
        }
        return ndown;
    }

    inline void RecvMessage(Message *msg) {
        int len;
        assert(RecvAll(&len, sizeof(len)) == sizeof(len));
        memset(msg, 0, len);
        assert(RecvAll(msg, len) == len);
    }

    SOCKET m_socket;
};

class TcpSocket : public Socket {
  public:
    TcpSocket()
        : Socket() {}
    TcpSocket(SOCKET _socket)
        : Socket(_socket) {}

    inline void Create() {
        if (m_socket != INVALID_SOCKET) {
            // relese before create
            Close();
        }
        m_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
        int on   = 1;
        if (0 != setsockopt(m_socket, SOL_SOCKET, SO_REUSEADDR, (char *) &on,
                            sizeof(on))) {
            Close();            
            LOG_ERROR("setsocketopt reuseaddr wrong");
        }
    }

    inline TcpSocket Accept() {
        LOG_INFO("tcp socket accepting\n");
        SOCKET comming_sockt = accept(m_socket, nullptr, nullptr);
        LOG_INFO("tcp socket accepting done\n");

        return TcpSocket(comming_sockt);
    }
};

class PollHelper {
  public:
    PollHelper()
        : pollfd_map{} {}
    inline void AddRead(SOCKET socketfd) {
        pollfd _pollfd;
        _pollfd.fd           = socketfd;
        _pollfd.events       = POLLIN;
        pollfd_map[socketfd] = _pollfd;
    }

    inline void AddWrite(SOCKET socketfd) {
        pollfd _pollfd;
        _pollfd.fd           = socketfd;
        _pollfd.events       = POLLOUT;
        pollfd_map[socketfd] = _pollfd;
    }

    inline Socket GetRead() {
        SOCKET res = INVALID_SOCKET;
        for (auto &pair : pollfd_map) {
            if (pair.second.revents & POLLIN) {
                res = pair.first;
                break;
            }
        }
        pollfd_map[res].events &= pollfd_map[res].revents;
        return res;
    }

    inline Socket GetWrite() {
        SOCKET res = INVALID_SOCKET;
        for (auto &pair : pollfd_map) {
            if (pair.second.revents & POLLOUT) {
                res = pair.first;
                break;
            }
        }
        pollfd_map[res].events &= pollfd_map[res].revents;
        return res;
    }

    inline void Poll(int timeout = -1) {
        std::vector<pollfd> fdset;
        fdset.reserve(pollfd_map.size());
        for (auto kv : pollfd_map) {
            fdset.push_back(kv.second);
        }
        int ret = poll(fdset.data(), fdset.size(), timeout);
        if (ret == -1) {
            LOG_ERROR("poll error");
        } else {
            for (auto &pfd : fdset) {
                auto revents = pfd.revents & pfd.events;
                if (!revents) {
                    pollfd_map.erase(pfd.fd);
                } else {
                    pollfd_map[pfd.fd].events = revents;
                }
            }
        }
    }

  private:
    std::unordered_map<SOCKET, pollfd> pollfd_map;
};
}

} // namespace dl