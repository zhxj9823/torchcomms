#ifndef NCCL_MOCK_SYSCALLS_H_
#define NCCL_MOCK_SYSCALLS_H_

#ifndef NCCL_MOCK_TOPOLOGY
#define NCCL_MOCK_TOPOLOGY
#endif

#ifdef NCCL_MOCK_TOPOLOGY

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/socket.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <netinet/in.h>

// Mock topology configuration structure
struct MockTopology {
  // Network interfaces configuration
  int num_interfaces;
  struct {
    char name[16];
    int family;  // AF_INET or AF_INET6
    char ip[INET6_ADDRSTRLEN];
    char netmask[INET6_ADDRSTRLEN];
    int flags;
  } interfaces[16];
  
  // Socket behavior configuration
  int socket_success;
  int bind_success;
  int listen_success;
  int accept_success;
  int connect_success;
  
  // Communication simulation
  int simulate_network_delay;
  int simulate_failures;
};

// Global mock topology instance
extern struct MockTopology g_mock_topology;

// Initialize mock topology from environment variables or config file
void ncclInitMockTopology();

// Mock function declarations
int mock_getifaddrs(struct ifaddrs **ifap);
void mock_freeifaddrs(struct ifaddrs *ifa);
int mock_getnameinfo(const struct sockaddr *sa, socklen_t salen,
                     char *host, size_t hostlen,
                     char *serv, size_t servlen, int flags);
int mock_getaddrinfo(const char *node, const char *service,
                     const struct addrinfo *hints,
                     struct addrinfo **res);
void mock_freeaddrinfo(struct addrinfo *res);
int mock_socket(int domain, int type, int protocol);
int mock_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
int mock_listen(int sockfd, int backlog);
int mock_accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen);
int mock_connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen);
ssize_t mock_send(int sockfd, const void *buf, size_t len, int flags);
ssize_t mock_recv(int sockfd, void *buf, size_t len, int flags);
int mock_setsockopt(int sockfd, int level, int optname,
                    const void *optval, socklen_t optlen);
int mock_getsockopt(int sockfd, int level, int optname,
                    void *optval, socklen_t *optlen);
int mock_getsockname(int sockfd, struct sockaddr *addr, socklen_t *addrlen);
int mock_fcntl(int fd, int cmd, ...);
int mock_close(int fd);
int mock_shutdown(int sockfd, int how);
int mock_dup2(int oldfd, int newfd);

// Function pointer redirection macros
// #define getifaddrs mock_getifaddrs
// #define freeifaddrs mock_freeifaddrs
// #define getnameinfo mock_getnameinfo
// #define getaddrinfo mock_getaddrinfo
// #define freeaddrinfo mock_freeaddrinfo
// #define socket mock_socket
// #define bind mock_bind
// #define listen mock_listen
// #define accept mock_accept
// #define connect mock_connect
// #define send mock_send
// #define recv mock_recv
// #define setsockopt mock_setsockopt
// #define getsockopt mock_getsockopt
// #define getsockname mock_getsockname
// #define fcntl mock_fcntl
// #define close mock_close
// #define shutdown mock_shutdown
// #define dup2 mock_dup2

// Utility functions for shared memory mock communication
void ncclMockSetNodeInfo(int node_id, const char* hostname, const char* ip);
void ncclMockPrintNodeInfo();
void ncclMockResetSharedMemory();

#ifdef __cplusplus
}
#endif

#endif // NCCL_MOCK_TOPOLOGY

#endif // NCCL_MOCK_SYSCALLS_H_
