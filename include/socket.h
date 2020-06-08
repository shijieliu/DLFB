#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <poll.h>
#include <vector>

namespace dl
{
using SOCKET = int;
const SOCKET INVALID_SOCKET = -1;

struct AddrinfoDetail
{
    char *host = nullptr;
    int port = -1;

    AddrinfoDetail(const char *host, int port) : host(const_cast<char *>(host)),
                                                 port(port) {}
};
struct Addrinfo
{
    sockaddr_in m_addr;
    AddrinfoDetail m_addrinfoDetail;

    Addrinfo() = delete;
    explicit Addrinfo(const char *host, const int port) : m_addrinfoDetail(host, port)
    {
        addrinfo hints;
        memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM; // TCP stream sockets
        hints.ai_flags = AI_PASSIVE;     // fill in my IP for me
        addrinfo *serveinfo = nullptr;

        int status = getaddrinfo(host, NULL, &hints, &serveinfo);
        if (status == 0 && serveinfo != nullptr)
        {
            memcpy(&m_addr, serveinfo->ai_addr, serveinfo->ai_addrlen);
            m_addr.sin_port = htons(port);
            freeaddrinfo(serveinfo);
            return;
        }
        fprintf(stderr, "[socket.h]can not obtain addr %s\n", host);
        assert(0);
    }

    ~Addrinfo() {}
};

class Socket
{
public:
    Socket() : m_socket(INVALID_SOCKET) {}
    Socket(SOCKET _socket) : m_socket(_socket) {}

    inline int Close()
    {
        return close(m_socket);
    }

    inline int Shutdown()
    {
        return shutdown(m_socket, SHUT_RDWR);
    }
    inline bool Isvalid()
    {
        return m_socket == INVALID_SOCKET;
    }

    inline void Bind(const Addrinfo &addr)
    {
        fprintf(stdout, "[socket.h]bind to %s:%d\n", addr.m_addrinfoDetail.host, addr.m_addrinfoDetail.port);
        if (-1 == bind(m_socket, reinterpret_cast<const sockaddr *>(&addr.m_addr), sizeof(addr.m_addr)))
        {
            fprintf(stderr, "[socket.h]can not bind to %s:%d\n", addr.m_addrinfoDetail.host, addr.m_addrinfoDetail.port);
            assert(0);
        }
    }

    inline void Listen(int backlog = 16)
    {
        fprintf(stdout, "[socket.h]listen\n");
        if (-1 == listen(m_socket, backlog))
        {
            fprintf(stderr, "[socket.h]can not listen\n");
            assert(0);
        }
    }

    inline void Connect(const Addrinfo &addr)
    {
        fprintf(stdout, "[socket.h]connect to %s:%d\n", addr.m_addrinfoDetail.host, addr.m_addrinfoDetail.port);
        if (-1 == connect(m_socket, reinterpret_cast<const sockaddr *>(&addr.m_addr), sizeof(addr.m_addr)))
        {
            Close();
            fprintf(stderr, "[socket.h]can not connect to %s:%d\n", addr.m_addrinfoDetail.host, addr.m_addrinfoDetail.port);
            assert(0);
        }
        fprintf(stdout, "[socket.h]connect to %s:%d finish\n", addr.m_addrinfoDetail.host, addr.m_addrinfoDetail.port);
    }

    inline size_t Send(const void *data, size_t size, int flag = 0)
    {
        const char *buf = reinterpret_cast<const char *>(data);
        return send(m_socket, buf, size, flag);
    }

    inline size_t SendAll(const void *data, size_t len)
    {
        const char *buf = reinterpret_cast<const char *>(data);
        size_t ndown = 0;

        fprintf(stdout, "[socket.h]sendall msg %s, len: %zu\n", buf, len);
        while (ndown < len)
        {
            size_t ret = Send(buf, len - ndown);
            fprintf(stdout, "[socket.h]sendall msg down size %zu\n", ret);
            // assert(0);
            if (ret == -1)
            {
                fprintf(stderr, "[socket.h]Send msg wrong\n");
                assert(0);
            }
            if (ret == 0)
            {
                return ndown;
            }
            buf += ret;
            ndown += ret;
        }
        fprintf(stdout, "[socket.h]sendall msg finish\n");
        return ndown;
    }

    inline void SendStr(const std::string &data)
    {
        int len = static_cast<int>(data.size());
        assert(SendAll(&len, sizeof(len)) == sizeof(len));
        assert(SendAll(data.c_str(), data.size()) == len);
    }

    inline size_t Recv(void *data, size_t size, int flag = 0)
    {
        char *buf = reinterpret_cast<char *>(data);
        return recv(m_socket, buf, size, flag);
    }

    inline size_t RecvAll(void *data, size_t len)
    {
        char *buf = reinterpret_cast<char *>(data);
        size_t ndown = 0;

        fprintf(stdout, "[socket.h]recvall msg\n");
        while (ndown < len)
        {
            size_t ret = Recv(buf, len - ndown);
            fprintf(stdout, "[socket.h]recv msg down size %zu\n", ret);
            if (ret == -1)
            {
                fprintf(stderr, "[socket.h]Recv msg wrong\n");
                assert(0);
            }
            if (ret == 0)
            {
                return ndown;
            }
            buf += ret;
            ndown += ret;
        }
        fprintf(stdout, "[socket.h]recvall msg finish\n");
        return ndown;
    }

    inline void RecvStr(std::string *data)
    {
        int len;
        assert(RecvAll(&len, sizeof(len)) == sizeof(len));
        fprintf(stdout, "[socket.h]recvstr reserve\n");
        data->reserve(len);
        fprintf(stdout, "[socket.h]recvstr reserve done\n");
        assert(RecvAll(&(*data)[0], len) == len);
    }

    SOCKET m_socket;
};

class TcpSocket : public Socket
{
public:
    TcpSocket() : Socket() {}
    TcpSocket(SOCKET _socket) : Socket(_socket) {}

    inline void Create()
    {
        if (m_socket != INVALID_SOCKET)
        {
            // relese before create
            Close();
        }
        m_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
        int on = 1;
        if (0 != setsockopt(m_socket, SOL_SOCKET, SO_REUSEADDR, (char *)&on, sizeof(on)))
        {
            fprintf(stderr, " setsocketopt reuseaddr wrong");
            Close();
            assert(0);
        }
    }

    inline TcpSocket Accept()
    {
        fprintf(stdout, "[socket.h]tcp socket accepting\n");
        SOCKET comming_sockt = accept(m_socket, nullptr, nullptr);
        fprintf(stdout, "[socket.h]tcp socket accepting done\n");

        return TcpSocket(comming_sockt);
    }
};

class Poller
{
private:
    std::unordered_map<SOCKET, pollfd> pollfd_map;

public:
    Poller() : pollfd_map{} {}
    inline void AddRead(SOCKET socketfd)
    {
        pollfd _pollfd;
        _pollfd.fd = socketfd;
        _pollfd.events = POLLIN;
        pollfd_map[socketfd] = _pollfd;
    }

    inline void AddWrite(SOCKET socketfd)
    {
        pollfd _pollfd;
        _pollfd.fd = socketfd;
        _pollfd.events = POLLOUT;
        pollfd_map[socketfd] = _pollfd;
    }

    inline Socket GetRead()
    {
        SOCKET res = INVALID_SOCKET;
        for (auto &pair : pollfd_map)
        {
            if (pair.second.revents & POLLIN)
            {
                res = pair.first;
                break;
            }
        }
        pollfd_map[res].events &= pollfd_map[res].revents;
        return res;
    }

    inline Socket GetWrite()
    {
        SOCKET res = INVALID_SOCKET;
        for (auto &pair : pollfd_map)
        {
            if (pair.second.revents & POLLOUT)
            {
                res = pair.first;
                break;
            }
        }
        pollfd_map[res].events &= pollfd_map[res].revents;
        return res;
    }

    inline void Poll(int timeout = -1)
    {
        std::vector<pollfd> fdset;
        fdset.reserve(pollfd_map.size());
        for (auto kv : pollfd_map)
        {
            fdset.push_back(kv.second);
        }
        int ret = poll(fdset.data(), fdset.size(), timeout);
        if (ret == -1)
        {
            fprintf(stderr, "[socket.h] poll ret -1");
        }
        else
        {
            for (auto &pfd : fdset)
            {
                auto revents = pfd.revents & pfd.events;
                if (!revents)
                {
                    pollfd_map.erase(pfd.fd);
                }
                else
                {
                    pollfd_map[pfd.fd].events = revents;
                }
            }
        }
    }
};
} // namespace dl