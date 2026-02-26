/*************************************************************************
 * Copyright (c) 2025, Mock Topology Implementation for NCCL
 *
 * Mock syscalls implementation for NCCL topology emulation
 ************************************************************************/

#define NCCL_MOCK_TOPOLOGY
#ifdef NCCL_MOCK_TOPOLOGY

#include "../include/mock_syscalls.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdarg.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>

#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <semaphore.h>

// C linkage for function declarations
extern "C" {

// Forward declarations
void mock_freeifaddrs(struct ifaddrs *ifa);

// Global mock topology configuration
struct MockTopology g_mock_topology = {0};

// Shared memory communication structures
#define MOCK_SHM_NAME "/nccl_mock_comm"
#define MAX_NODES 16
#define MAX_CONNECTIONS_PER_NODE 32
#define RING_BUFFER_SIZE 65536

struct MockRingBuffer {
  volatile int head;
  volatile int tail;
  volatile int size;
  char data[RING_BUFFER_SIZE];
  pthread_mutex_t mutex;
  sem_t data_available;
  sem_t space_available;
};

struct MockNodeConnection {
  int active;
  int node_id;
  int peer_node_id;
  int port;
  struct MockRingBuffer send_ring;
  struct MockRingBuffer recv_ring;
};

struct MockSharedMemory {
  int initialized;
  int num_nodes;
  pthread_mutex_t global_mutex;
  struct MockNodeConnection connections[MAX_NODES][MAX_CONNECTIONS_PER_NODE];
  char node_hostnames[MAX_NODES][64];
  char node_ips[MAX_NODES][INET_ADDRSTRLEN];
};

static struct MockSharedMemory* g_mock_shm = NULL;
static int g_node_id = -1;
static char g_node_hostname[64] = {0};

// Mock socket communication simulation
#define MAX_MOCK_SOCKETS 256
struct MockSocket {
  int fd;
  int peer_fd;
  char send_buffer[8192];
  int send_head, send_tail;
  char recv_buffer[8192];
  int recv_head, recv_tail;
  int connected;
  int listening;
  struct sockaddr_storage local_addr;
  struct sockaddr_storage peer_addr;
  // Shared memory communication fields
  int node_id;
  int peer_node_id;
  int connection_id;
  struct MockNodeConnection* connection;
};

static struct MockSocket mock_sockets[MAX_MOCK_SOCKETS];
static int mock_socket_count = 0;

// Initialize shared memory for mock communication
static int init_shared_memory() {
  if (g_mock_shm != NULL) {
    return 0; // Already initialized
  }
  
  int shm_fd = shm_open(MOCK_SHM_NAME, O_CREAT | O_RDWR, 0666);
  if (shm_fd == -1) {
    perror("[MOCK] shm_open failed");
    return -1;
  }
  
  // Set size of shared memory
  if (ftruncate(shm_fd, sizeof(struct MockSharedMemory)) == -1) {
    perror("[MOCK] ftruncate failed");
    close(shm_fd);
    return -1;
  }
  
  // Map shared memory
  g_mock_shm = (struct MockSharedMemory*)mmap(NULL, sizeof(struct MockSharedMemory),
                                              PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
  if (g_mock_shm == MAP_FAILED) {
    perror("[MOCK] mmap failed");
    close(shm_fd);
    return -1;
  }
  
  close(shm_fd);
  
  // Initialize shared memory structure if not already done
  if (!g_mock_shm->initialized) {
    memset(g_mock_shm, 0, sizeof(struct MockSharedMemory));
    pthread_mutexattr_t mutex_attr;
    pthread_mutexattr_init(&mutex_attr);
    pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&g_mock_shm->global_mutex, &mutex_attr);
    pthread_mutexattr_destroy(&mutex_attr);
    
    g_mock_shm->num_nodes = 0;
    g_mock_shm->initialized = 1;
    
    printf("[MOCK] Shared memory initialized\n");
    fflush(stdout);
  } else {
    // If shared memory already exists, reset it for clean start
    printf("[MOCK] Shared memory already exists, resetting for clean start\n");
    fflush(stdout);
    
    pthread_mutex_lock(&g_mock_shm->global_mutex);
    
    // Clean up existing connections
    for (int i = 0; i < MAX_NODES; i++) {
      for (int j = 0; j < MAX_CONNECTIONS_PER_NODE; j++) {
        struct MockNodeConnection* conn = &g_mock_shm->connections[i][j];
        if (conn->active) {
          conn->active = 0;
          sem_destroy(&conn->send_ring.data_available);
          sem_destroy(&conn->send_ring.space_available);
          sem_destroy(&conn->recv_ring.data_available);
          sem_destroy(&conn->recv_ring.space_available);
          pthread_mutex_destroy(&conn->send_ring.mutex);
          pthread_mutex_destroy(&conn->recv_ring.mutex);
        }
      }
    }
    
    // Reset node count and hostnames
    g_mock_shm->num_nodes = 0;
    memset(g_mock_shm->node_hostnames, 0, sizeof(g_mock_shm->node_hostnames));
    memset(g_mock_shm->node_ips, 0, sizeof(g_mock_shm->node_ips));
    
    pthread_mutex_unlock(&g_mock_shm->global_mutex);
    
    printf("[MOCK] Shared memory reset completed\n");
    fflush(stdout);
  }
  
  return 0;
}

// Get or assign node ID based on hostname and interface
static int get_node_id() {
  if (g_node_id >= 0) {
    return g_node_id;
  }
  
  if (init_shared_memory() != 0) {
    return -1;
  }
  
  // Get hostname
  if (gethostname(g_node_hostname, sizeof(g_node_hostname)) != 0) {
    strcpy(g_node_hostname, "localhost");
  }
  
  // Make hostname unique by including process ID and interface name
  const char* socket_ifname = getenv("NCCL_SOCKET_IFNAME");
  char unique_hostname[64];
  if (socket_ifname) {
    snprintf(unique_hostname, sizeof(unique_hostname), "%s-%s-%d", 
             g_node_hostname, socket_ifname, getpid());
  } else {
    snprintf(unique_hostname, sizeof(unique_hostname), "%s-%d", g_node_hostname, getpid());
  }
  strcpy(g_node_hostname, unique_hostname);
  
  pthread_mutex_lock(&g_mock_shm->global_mutex);
  
  // Check if this hostname already has a node ID
  for (int i = 0; i < g_mock_shm->num_nodes; i++) {
    if (strcmp(g_mock_shm->node_hostnames[i], g_node_hostname) == 0) {
      g_node_id = i;
      pthread_mutex_unlock(&g_mock_shm->global_mutex);
      printf("[MOCK] Found existing node ID %d for hostname %s\n", g_node_id, g_node_hostname);
      fflush(stdout);
      return g_node_id;
    }
  }
  
  // Assign new node ID - for MPI testing, try to assign based on interface name
  int preferred_node_id = -1;
  if (socket_ifname) {
    if (strcmp(socket_ifname, "tap-nccl-0") == 0) {
      preferred_node_id = 0;
    } else if (strcmp(socket_ifname, "tap-nccl-1") == 0) {
      preferred_node_id = 1;
    }
  }
  
  if (preferred_node_id >= 0 && preferred_node_id < MAX_NODES) {
    // Check if preferred node ID is available
    int id_available = 1;
    for (int i = 0; i < g_mock_shm->num_nodes; i++) {
      if (i == preferred_node_id && strlen(g_mock_shm->node_hostnames[i]) > 0) {
        id_available = 0;
        break;
      }
    }
    
    if (id_available) {
      g_node_id = preferred_node_id;
      if (g_mock_shm->num_nodes <= g_node_id) {
        g_mock_shm->num_nodes = g_node_id + 1;
      }
    }
  }
  
  // If no preferred ID or it's taken, assign next available
  if (g_node_id < 0) {
    if (g_mock_shm->num_nodes >= MAX_NODES) {
      pthread_mutex_unlock(&g_mock_shm->global_mutex);
      printf("[MOCK] Too many nodes, maximum is %d\n", MAX_NODES);
      fflush(stdout);
      return -1;
    }
    
    g_node_id = g_mock_shm->num_nodes++;
  }
  
  strcpy(g_mock_shm->node_hostnames[g_node_id], g_node_hostname);
  
  // Set IP based on node ID and interface
  if (socket_ifname && strstr(socket_ifname, "tap-nccl-")) {
    const char* num_str = socket_ifname + strlen("tap-nccl-");
    int interface_num = atoi(num_str);
    // Ensure interface_num is in valid range for IP address (1-254)
    int ip_octet = (interface_num >= 0 && interface_num < 254) ? interface_num + 1 : 1;
    snprintf(g_mock_shm->node_ips[g_node_id], INET_ADDRSTRLEN, "10.1.2.%d", ip_octet);
  } else {
    // Ensure node_id is in valid range for IP address (1-254)
    int ip_octet = (g_node_id >= 0 && g_node_id < 254) ? g_node_id + 1 : 1;
    snprintf(g_mock_shm->node_ips[g_node_id], INET_ADDRSTRLEN, "10.1.2.%d", ip_octet);
  }
  
  pthread_mutex_unlock(&g_mock_shm->global_mutex);
  
  printf("[MOCK] Assigned node ID %d for hostname %s (interface: %s)\n", 
         g_node_id, g_node_hostname, socket_ifname ? socket_ifname : "none");
  fflush(stdout);
  return g_node_id;
}

// Initialize ring buffer
static void init_ring_buffer(struct MockRingBuffer* ring) {
  ring->head = 0;
  ring->tail = 0;
  ring->size = 0;
  
  pthread_mutexattr_t mutex_attr;
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutexattr_setpshared(&mutex_attr, PTHREAD_PROCESS_SHARED);
  pthread_mutex_init(&ring->mutex, &mutex_attr);
  pthread_mutexattr_destroy(&mutex_attr);
  
  sem_init(&ring->data_available, 1, 0);
  sem_init(&ring->space_available, 1, RING_BUFFER_SIZE);
}

// Write to ring buffer
static int ring_buffer_write(struct MockRingBuffer* ring, const void* data, size_t len) {
  if (len > RING_BUFFER_SIZE) {
    return -1; // Data too large
  }
  
  // Wait for space
  struct timespec timeout;
  clock_gettime(CLOCK_REALTIME, &timeout);
  timeout.tv_sec += 1; // 1 second timeout
  
  for (size_t i = 0; i < len; i++) {
    if (sem_timedwait(&ring->space_available, &timeout) != 0) {
      return i; // Partial write
    }
  }
  
  pthread_mutex_lock(&ring->mutex);
  
  const char* src = (const char*)data;
  for (size_t i = 0; i < len; i++) {
    ring->data[ring->tail] = src[i];
    ring->tail = (ring->tail + 1) % RING_BUFFER_SIZE;
    ring->size++;
  }
  
  pthread_mutex_unlock(&ring->mutex);
  
  // Signal data available
  for (size_t i = 0; i < len; i++) {
    sem_post(&ring->data_available);
  }
  
  return len;
}

// Read from ring buffer
static int ring_buffer_read(struct MockRingBuffer* ring, void* data, size_t len, int blocking) {
  char* dst = (char*)data;
  size_t bytes_read = 0;
  
  for (size_t i = 0; i < len; i++) {
    if (blocking) {
      if (sem_wait(&ring->data_available) != 0) {
        break;
      }
    } else {
      if (sem_trywait(&ring->data_available) != 0) {
        break;
      }
    }
    
    pthread_mutex_lock(&ring->mutex);
    
    if (ring->size > 0) {
      dst[bytes_read] = ring->data[ring->head];
      ring->head = (ring->head + 1) % RING_BUFFER_SIZE;
      ring->size--;
      bytes_read++;
    }
    
    pthread_mutex_unlock(&ring->mutex);
    
    sem_post(&ring->space_available);
  }
  
  return bytes_read;
}

// Find or create connection between nodes
static struct MockNodeConnection* get_connection(int local_node_id, int peer_node_id, int port) {
  if (init_shared_memory() != 0) {
    return NULL;
  }
  
  pthread_mutex_lock(&g_mock_shm->global_mutex);
  
  // Look for existing connection
  for (int i = 0; i < MAX_CONNECTIONS_PER_NODE; i++) {
    struct MockNodeConnection* conn = &g_mock_shm->connections[local_node_id][i];
    if (conn->active && conn->peer_node_id == peer_node_id && conn->port == port) {
      pthread_mutex_unlock(&g_mock_shm->global_mutex);
      return conn;
    }
  }
  
  // Create new connection
  for (int i = 0; i < MAX_CONNECTIONS_PER_NODE; i++) {
    struct MockNodeConnection* conn = &g_mock_shm->connections[local_node_id][i];
    if (!conn->active) {
      conn->active = 1;
      conn->node_id = local_node_id;
      conn->peer_node_id = peer_node_id;
      conn->port = port;
      
      init_ring_buffer(&conn->send_ring);
      init_ring_buffer(&conn->recv_ring);
      
      printf("[MOCK] Created connection: node %d -> node %d on port %d\n", 
             local_node_id, peer_node_id, port);
      fflush(stdout);
      
      pthread_mutex_unlock(&g_mock_shm->global_mutex);
      return conn;
    }
  }
  
  pthread_mutex_unlock(&g_mock_shm->global_mutex);
  printf("[MOCK] Failed to create connection: no slots available\n");
  fflush(stdout);
  return NULL;
}

// Find mock socket by fd
static struct MockSocket* find_mock_socket(int fd) {
  for (int i = 0; i < mock_socket_count; i++) {
    if (mock_sockets[i].fd == fd && fd >= 0) {
      return &mock_sockets[i];
    }
  }
  return NULL;
}

// Create a new mock socket entry
static struct MockSocket* create_mock_socket(int fd) {
  if (mock_socket_count >= MAX_MOCK_SOCKETS) {
    return NULL;
  }
  
  struct MockSocket* sock = &mock_sockets[mock_socket_count++];
  memset(sock, 0, sizeof(struct MockSocket));
  sock->fd = fd;
  sock->peer_fd = -1;
  sock->node_id = get_node_id();
  sock->peer_node_id = -1;
  sock->connection_id = -1;
  sock->connection = NULL;
  return sock;
}

// Function pointers to original system calls
static int (*orig_getifaddrs)(struct ifaddrs **ifap) = NULL;
static void (*orig_freeifaddrs)(struct ifaddrs *ifa) = NULL;
static int (*orig_socket)(int domain, int type, int protocol) = NULL;
static int (*orig_bind)(int sockfd, const struct sockaddr *addr, socklen_t addrlen) = NULL;
static int (*orig_listen)(int sockfd, int backlog) = NULL;
static int (*orig_accept)(int sockfd, struct sockaddr *addr, socklen_t *addrlen) = NULL;
static int (*orig_connect)(int sockfd, const struct sockaddr *addr, socklen_t addrlen) = NULL;
static ssize_t (*orig_send)(int sockfd, const void *buf, size_t len, int flags) = NULL;
static ssize_t (*orig_recv)(int sockfd, void *buf, size_t len, int flags) = NULL;
static int (*orig_close)(int fd) = NULL;
static int (*orig_setsockopt)(int sockfd, int level, int optname, const void *optval, socklen_t optlen) = NULL;
static int (*orig_getsockopt)(int sockfd, int level, int optname, void *optval, socklen_t *optlen) = NULL;
static int (*orig_getsockname)(int sockfd, struct sockaddr *addr, socklen_t *addrlen) = NULL;
static int (*orig_shutdown)(int sockfd, int how) = NULL;
static int (*orig_getnameinfo)(const struct sockaddr *sa, socklen_t salen, char *host, size_t hostlen, char *serv, size_t servlen, int flags) = NULL;
static int (*orig_getaddrinfo)(const char *node, const char *service, const struct addrinfo *hints, struct addrinfo **res) = NULL;
static void (*orig_freeaddrinfo)(struct addrinfo *res) = NULL;
static int (*orig_fcntl)(int fd, int cmd, ...) = NULL;
static int (*orig_dup2)(int oldfd, int newfd) = NULL;

// Initialize original function pointers
static void init_orig_functions() {
  if (!orig_getifaddrs) {
    orig_getifaddrs = (int(*)(struct ifaddrs**))dlsym(RTLD_NEXT, "getifaddrs");
  }
  if (!orig_freeifaddrs) {
    orig_freeifaddrs = (void(*)(struct ifaddrs*))dlsym(RTLD_NEXT, "freeifaddrs");
  }
  if (!orig_socket) {
    orig_socket = (int(*)(int, int, int))dlsym(RTLD_NEXT, "socket");
  }
  if (!orig_bind) {
    orig_bind = (int(*)(int, const struct sockaddr*, socklen_t))dlsym(RTLD_NEXT, "bind");
  }
  if (!orig_listen) {
    orig_listen = (int(*)(int, int))dlsym(RTLD_NEXT, "listen");
  }
  if (!orig_accept) {
    orig_accept = (int(*)(int, struct sockaddr*, socklen_t*))dlsym(RTLD_NEXT, "accept");
  }
  if (!orig_connect) {
    orig_connect = (int(*)(int, const struct sockaddr*, socklen_t))dlsym(RTLD_NEXT, "connect");
  }
  if (!orig_send) {
    orig_send = (ssize_t(*)(int, const void*, size_t, int))dlsym(RTLD_NEXT, "send");
  }
  if (!orig_recv) {
    orig_recv = (ssize_t(*)(int, void*, size_t, int))dlsym(RTLD_NEXT, "recv");
  }
  if (!orig_close) {
    orig_close = (int(*)(int))dlsym(RTLD_NEXT, "close");
  }
  if (!orig_setsockopt) {
    orig_setsockopt = (int(*)(int, int, int, const void*, socklen_t))dlsym(RTLD_NEXT, "setsockopt");
  }
  if (!orig_getsockopt) {
    orig_getsockopt = (int(*)(int, int, int, void*, socklen_t*))dlsym(RTLD_NEXT, "getsockopt");
  }
  if (!orig_getsockname) {
    orig_getsockname = (int(*)(int, struct sockaddr*, socklen_t*))dlsym(RTLD_NEXT, "getsockname");
  }
  if (!orig_shutdown) {
    orig_shutdown = (int(*)(int, int))dlsym(RTLD_NEXT, "shutdown");
  }
  if (!orig_getnameinfo) {
    orig_getnameinfo = (int(*)(const struct sockaddr*, socklen_t, char*, size_t, char*, size_t, int))dlsym(RTLD_NEXT, "getnameinfo");
  }
  if (!orig_getaddrinfo) {
    orig_getaddrinfo = (int(*)(const char*, const char*, const struct addrinfo*, struct addrinfo**))dlsym(RTLD_NEXT, "getaddrinfo");
  }
  if (!orig_freeaddrinfo) {
    orig_freeaddrinfo = (void(*)(struct addrinfo*))dlsym(RTLD_NEXT, "freeaddrinfo");
  }
  if (!orig_fcntl) {
    orig_fcntl = (int(*)(int, int, ...))dlsym(RTLD_NEXT, "fcntl");
  }
  if (!orig_dup2) {
    orig_dup2 = (int(*)(int, int))dlsym(RTLD_NEXT, "dup2");
  }
}

// Constructor to initialize mock topology on library load
// __attribute__((constructor))
static void init_mock_topology() {
  printf("[MOCK] Library loaded, initializing mock topology\n");
  fflush(stdout);
  ncclInitMockTopology();
}

void ncclInitMockTopology() {
  // Initialize default topology if not already configured
  if (g_mock_topology.num_interfaces == 0) {
    printf("[MOCK] Initializing mock topology\n");
    fflush(stdout);
    
    // Check for specific interface names from environment
    const char* socket_ifname = getenv("NCCL_SOCKET_IFNAME");
    printf("[MOCK] NCCL_SOCKET_IFNAME = %s\n", socket_ifname ? socket_ifname : "NULL");
    fflush(stdout);
    
    // Default: Create TAP interfaces that match the actual network setup
    if (socket_ifname) {
      g_mock_topology.num_interfaces = 1;
      strcpy(g_mock_topology.interfaces[0].name, socket_ifname);
      g_mock_topology.interfaces[0].family = AF_INET;
      strcpy(g_mock_topology.interfaces[0].netmask, "255.255.255.0");
      g_mock_topology.interfaces[0].flags = IFF_UP | IFF_BROADCAST | IFF_RUNNING | IFF_MULTICAST;
      
      // Match IP address based on interface name
      if (strcmp(socket_ifname, "tap-nccl-0") == 0) {
        strcpy(g_mock_topology.interfaces[0].ip, "10.1.2.1");
      } else if (strcmp(socket_ifname, "tap-nccl-1") == 0) {
        strcpy(g_mock_topology.interfaces[0].ip, "10.1.2.2");
      } else {
        // Default IP for other interface names
        strcpy(g_mock_topology.interfaces[0].ip, "10.1.2.1");
      }
    } else {
      // Create both tap-nccl-0 and tap-nccl-1 interfaces
      g_mock_topology.num_interfaces = 2;
      
      // tap-nccl-0
      strcpy(g_mock_topology.interfaces[0].name, "tap-nccl-0");
      g_mock_topology.interfaces[0].family = AF_INET;
      strcpy(g_mock_topology.interfaces[0].ip, "10.1.2.1");
      strcpy(g_mock_topology.interfaces[0].netmask, "255.255.255.0");
      g_mock_topology.interfaces[0].flags = IFF_UP | IFF_BROADCAST | IFF_RUNNING | IFF_MULTICAST;
      
      // tap-nccl-1
      strcpy(g_mock_topology.interfaces[1].name, "tap-nccl-1");
      g_mock_topology.interfaces[1].family = AF_INET;
      strcpy(g_mock_topology.interfaces[1].ip, "10.1.2.2");
      strcpy(g_mock_topology.interfaces[1].netmask, "255.255.255.0");
      g_mock_topology.interfaces[1].flags = IFF_UP | IFF_BROADCAST | IFF_RUNNING | IFF_MULTICAST;
    }
       
    g_mock_topology.socket_success = 1;
    g_mock_topology.bind_success = 1;
    g_mock_topology.listen_success = 1;
    g_mock_topology.accept_success = 1;
    g_mock_topology.connect_success = 1;
    g_mock_topology.simulate_network_delay = 0;
    g_mock_topology.simulate_failures = 0;
    
    printf("[MOCK] Mock topology initialized with %d interfaces\n", g_mock_topology.num_interfaces);
    for (int i = 0; i < g_mock_topology.num_interfaces; i++) {
      printf("[MOCK] Interface %d: %s (%s) flags=0x%x\n", i, 
             g_mock_topology.interfaces[i].name,
             g_mock_topology.interfaces[i].ip,
             g_mock_topology.interfaces[i].flags);
    }
    fflush(stdout);
  }
}

int mock_getifaddrs(struct ifaddrs **ifap) {
  printf("[MOCK] mock_getifaddrs called\n");
  fflush(stdout);
  
  init_orig_functions();
  
  const char* socket_ifname = getenv("NCCL_SOCKET_IFNAME");
  printf("[MOCK] NCCL_SOCKET_IFNAME = %s\n", socket_ifname ? socket_ifname : "NULL");
  fflush(stdout);
  
  if (socket_ifname) {
    // If NCCL_SOCKET_IFNAME is set, use the real getifaddrs but filter for the specified interface
    struct ifaddrs *real_ifap = NULL;
    int ret = orig_getifaddrs(&real_ifap);
    if (ret != 0) {
      return ret;
    }
    
    // Filter for the specified interface
    struct ifaddrs *filtered_head = NULL, *filtered_current = NULL;
    for (struct ifaddrs *ifa = real_ifap; ifa != NULL; ifa = ifa->ifa_next) {
      if (strcmp(ifa->ifa_name, socket_ifname) == 0) {
        // Copy this interface to our filtered list
        struct ifaddrs *new_ifa = (struct ifaddrs*)malloc(sizeof(struct ifaddrs));
        memcpy(new_ifa, ifa, sizeof(struct ifaddrs));
        
        // Copy strings
        new_ifa->ifa_name = strdup(ifa->ifa_name);
        if (ifa->ifa_addr) {
          new_ifa->ifa_addr = (struct sockaddr*)malloc(sizeof(struct sockaddr_storage));
          memcpy(new_ifa->ifa_addr, ifa->ifa_addr, sizeof(struct sockaddr_storage));
        }
        if (ifa->ifa_netmask) {
          new_ifa->ifa_netmask = (struct sockaddr*)malloc(sizeof(struct sockaddr_storage));
          memcpy(new_ifa->ifa_netmask, ifa->ifa_netmask, sizeof(struct sockaddr_storage));
        }
        if (ifa->ifa_dstaddr) {
          new_ifa->ifa_dstaddr = (struct sockaddr*)malloc(sizeof(struct sockaddr_storage));
          memcpy(new_ifa->ifa_dstaddr, ifa->ifa_dstaddr, sizeof(struct sockaddr_storage));
        }
        
        new_ifa->ifa_next = NULL;
        
        if (filtered_head == NULL) {
          filtered_head = new_ifa;
          filtered_current = new_ifa;
        } else {
          filtered_current->ifa_next = new_ifa;
          filtered_current = new_ifa;
        }
      }
    }
    
    orig_freeifaddrs(real_ifap);
    *ifap = filtered_head;
    
    printf("[MOCK] mock_getifaddrs returning filtered interface: %s\n", socket_ifname);
    fflush(stdout);
    return 0;
  } else {
    // If no specific interface requested, use original mock behavior
    ncclInitMockTopology();
  
    *ifap = NULL;
    struct ifaddrs *head = NULL, *current = NULL;
    
    for (int i = 0; i < g_mock_topology.num_interfaces; i++) {
      struct ifaddrs *ifa = (struct ifaddrs*)calloc(1, sizeof(struct ifaddrs));
      if (!ifa) {
        mock_freeifaddrs(head);
        errno = ENOMEM;
        return -1;
      }
      
      // Interface name
      ifa->ifa_name = strdup(g_mock_topology.interfaces[i].name);
      ifa->ifa_flags = g_mock_topology.interfaces[i].flags;
      
      // Address
      if (g_mock_topology.interfaces[i].family == AF_INET) {
        struct sockaddr_in *addr = (struct sockaddr_in*)calloc(1, sizeof(struct sockaddr_in));
        addr->sin_family = AF_INET;
        inet_pton(AF_INET, g_mock_topology.interfaces[i].ip, &addr->sin_addr);
        ifa->ifa_addr = (struct sockaddr*)addr;
        
        // Netmask
        struct sockaddr_in *netmask = (struct sockaddr_in*)calloc(1, sizeof(struct sockaddr_in));
        netmask->sin_family = AF_INET;
        inet_pton(AF_INET, g_mock_topology.interfaces[i].netmask, &netmask->sin_addr);
        ifa->ifa_netmask = (struct sockaddr*)netmask;
      }
      // Add IPv6 support if needed
      
      // Link to list
      if (head == NULL) {
        head = ifa;
        current = ifa;
      } else {
        current->ifa_next = ifa;
        current = ifa;
      }
    }
    
    *ifap = head;
    printf("[MOCK] mock_getifaddrs returning %d interfaces\n", g_mock_topology.num_interfaces);
    fflush(stdout);
    return 0;
  }
}

void mock_freeifaddrs(struct ifaddrs *ifa) {
  printf("[MOCK] mock_freeifaddrs called\n");
  fflush(stdout);
  
  while (ifa) {
    struct ifaddrs *next = ifa->ifa_next;
    free(ifa->ifa_name);
    free(ifa->ifa_addr);
    free(ifa->ifa_netmask);
    free(ifa);
    ifa = next;
  }
}

int mock_setsockopt(int sockfd, int level, int optname,
                    const void *optval, socklen_t optlen) {
  printf("[MOCK] mock_setsockopt called: sockfd=%d, level=%d, optname=%d\n", sockfd, level, optname);
  fflush(stdout);
  
  init_orig_functions();
  
  // Check if this is a pure mock socket
  struct MockSocket* sock = find_mock_socket(sockfd);
  if (sock && sockfd >= 10000) {
    printf("[MOCK] mock_setsockopt: pure mock socket %d, simulating success\n", sockfd);
    fflush(stdout);
    return 0; // Simulate success for pure mock sockets
  }
  
  // Use real setsockopt for actual socket configuration
  int result = orig_setsockopt(sockfd, level, optname, optval, optlen);
  printf("[MOCK] mock_setsockopt result=%d for fd=%d\n", result, sockfd);
  fflush(stdout);
  return result;
}

int mock_getsockopt(int sockfd, int level, int optname,
                    void *optval, socklen_t *optlen) {
  printf("[MOCK] mock_getsockopt called: sockfd=%d, level=%d, optname=%d\n", sockfd, level, optname);
  fflush(stdout);
  
  init_orig_functions();
  
  // Check if this is a pure mock socket
  struct MockSocket* sock = find_mock_socket(sockfd);
  if (sock && sockfd >= 10000) {
    printf("[MOCK] mock_getsockopt: pure mock socket %d, simulating success\n", sockfd);
    fflush(stdout);
    // Return some reasonable default values for common socket options
    if (level == SOL_SOCKET && optname == SO_ERROR) {
      *(int*)optval = 0; // No error
      *optlen = sizeof(int);
    }
    return 0; // Simulate success for pure mock sockets
  }
  
  // Use real getsockopt for actual socket information
  int result = orig_getsockopt(sockfd, level, optname, optval, optlen);
  printf("[MOCK] mock_getsockopt result=%d for fd=%d\n", result, sockfd);
  fflush(stdout);
  return result;
}

int mock_getsockname(int sockfd, struct sockaddr *addr, socklen_t *addrlen) {
  printf("[MOCK] mock_getsockname called: sockfd=%d\n", sockfd);
  fflush(stdout);
  
  init_orig_functions();
  
  // Check if this is a pure mock socket
  struct MockSocket* sock = find_mock_socket(sockfd);
  if (sock && sockfd >= 10000) {
    printf("[MOCK] mock_getsockname: pure mock socket %d, returning mock address\n", sockfd);
    fflush(stdout);
    
    // Return a mock local address
    if (*addrlen >= sizeof(struct sockaddr_in)) {
      struct sockaddr_in* sin = (struct sockaddr_in*)addr;
      sin->sin_family = AF_INET;
      sin->sin_port = htons(29500 + sock->node_id); // Use node ID as port offset
      char mock_ip[32];
      // Ensure node_id is in valid range for IP address (1-254)
      int ip_octet = (sock->node_id >= 0 && sock->node_id < 254) ? sock->node_id + 1 : 1;
      snprintf(mock_ip, sizeof(mock_ip), "10.1.2.%d", ip_octet); // node 0 -> IP .1, node 1 -> IP .2
      inet_pton(AF_INET, mock_ip, &sin->sin_addr);
      *addrlen = sizeof(struct sockaddr_in);
      printf("[MOCK] mock_getsockname returning IP %s for node %d\n", mock_ip, sock->node_id);
      fflush(stdout);
      return 0;
    }
    errno = EINVAL;
    return -1;
  }
  
  // Use real getsockname for actual socket information
  int result = orig_getsockname(sockfd, addr, addrlen);
  printf("[MOCK] mock_getsockname result=%d for fd=%d\n", result, sockfd);
  fflush(stdout);
  return result;
}

int mock_fcntl(int fd, int cmd, ...) {
  printf("[MOCK] mock_fcntl called: fd=%d, cmd=%d\n", fd, cmd);
  fflush(stdout);
  
  init_orig_functions();
  
  // Check if this is a pure mock socket
  struct MockSocket* sock = find_mock_socket(fd);
  if (sock && fd >= 10000) {
    printf("[MOCK] mock_fcntl: pure mock socket %d, simulating result for cmd=%d\n", fd, cmd);
    fflush(stdout);
    
    // Return reasonable defaults for common fcntl commands
    switch (cmd) {
      case F_GETFL:
        return O_RDWR; // Simulate read/write access
      case F_SETFL:
        return 0; // Simulate success
      default:
        return 0; // Simulate success for other commands
    }
  }
  
  va_list args;
  va_start(args, cmd);
  
  // For fcntl, we need to handle variable arguments and use original function
  int result;
  switch (cmd) {
    case F_GETFL:
      result = ((int(*)(int, int))orig_fcntl)(fd, cmd);
      break;
    case F_SETFL: {
      int flags = va_arg(args, int);
      result = ((int(*)(int, int, int))orig_fcntl)(fd, cmd, flags);
      break;
    }
    default: {
      long arg = va_arg(args, long);
      result = ((int(*)(int, int, long))orig_fcntl)(fd, cmd, arg);
      break;
    }
  }
  
  va_end(args);
  printf("[MOCK] mock_fcntl result=%d for fd=%d\n", result, fd);
  fflush(stdout);
  return result;
}

int mock_shutdown(int sockfd, int how) {
  printf("[MOCK] mock_shutdown called: sockfd=%d, how=%d\n", sockfd, how);
  fflush(stdout);
  
  init_orig_functions();
  
  // Use real shutdown for actual socket shutdown
  int result = orig_shutdown(sockfd, how);
  printf("[MOCK] mock_shutdown result=%d for fd=%d\n", result, sockfd);
  fflush(stdout);
  return result;
}

int mock_getnameinfo(const struct sockaddr *sa, socklen_t salen,
                     char *host, size_t hostlen,
                     char *serv, size_t servlen, int flags) {
  printf("[MOCK] mock_getnameinfo called\n");
  fflush(stdout);
  
  init_orig_functions();
  
  // Use real getnameinfo for actual name resolution
  int result = orig_getnameinfo(sa, salen, host, hostlen, serv, servlen, flags);
  printf("[MOCK] mock_getnameinfo result=%d\n", result);
  fflush(stdout);
  return result;
}

int mock_getaddrinfo(const char *node, const char *service,
                     const struct addrinfo *hints,
                     struct addrinfo **res) {
  printf("[MOCK] mock_getaddrinfo called: node=%s, service=%s\n", 
         node ? node : "NULL", service ? service : "NULL");
  fflush(stdout);
  
  init_orig_functions();
  
  // Use real getaddrinfo for actual address resolution
  int result = orig_getaddrinfo(node, service, hints, res);
  printf("[MOCK] mock_getaddrinfo result=%d\n", result);
  fflush(stdout);
  return result;
}

void mock_freeaddrinfo(struct addrinfo *res) {
  printf("[MOCK] mock_freeaddrinfo called\n");
  fflush(stdout);
  
  init_orig_functions();
  
  // Use real freeaddrinfo for actual cleanup
  orig_freeaddrinfo(res);
  printf("[MOCK] mock_freeaddrinfo completed\n");
  fflush(stdout);
}

// Mock implementation of socket functions
int mock_socket(int domain, int type, int protocol) {
  printf("[MOCK] mock_socket called: domain=%d, type=%d, protocol=%d\n", domain, type, protocol);
  fflush(stdout);
  
  init_orig_functions();
  
  if (!g_mock_topology.socket_success) {
    errno = EAFNOSUPPORT;
    return -1;
  }
  
  // For network sockets (TCP), create a pure mock socket without using real file descriptors
  if (domain == AF_INET && type == SOCK_STREAM) {
    // Always use pure mock sockets for TCP connections when mocking is enabled
    // Create a virtual file descriptor starting from a high number to avoid conflicts
    static int mock_fd_counter = 10000;
    int mock_fd = mock_fd_counter++;
    
    // Register this as a pure mock socket
    struct MockSocket* sock = create_mock_socket(mock_fd);
    if (sock) {
      sock->fd = mock_fd;
      printf("[MOCK] mock_socket returning pure mock fd=%d\n", mock_fd);
      fflush(stdout);
      return mock_fd;
    }
  }
  
  // Use real socket for actual communication
  int fd = orig_socket(domain, type, protocol);
  if (fd >= 0) {
    // Register this socket for potential mocking
    create_mock_socket(fd);
  }
  printf("[MOCK] mock_socket returning real fd=%d\n", fd);
  fflush(stdout);
  return fd;
}

int mock_bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
  printf("[MOCK] mock_bind called: sockfd=%d, addrlen=%d\n", sockfd, addrlen);
  fflush(stdout);
  
  init_orig_functions();
  
  if (!g_mock_topology.bind_success) {
    errno = EADDRINUSE;
    return -1;
  }
  
  // Check if this is a pure mock socket
  struct MockSocket* sock = find_mock_socket(sockfd);
  if (sock && sockfd >= 10000) {
    printf("[MOCK] mock_bind: pure mock socket %d, simulating bind success\n", sockfd);
    fflush(stdout);
    
    // Store the local address
    if (addrlen <= sizeof(sock->local_addr)) {
      memcpy(&sock->local_addr, addr, addrlen);
    }
    return 0; // Simulate success for pure mock sockets
  }
  
  // Use real bind for actual network communication
  int result = orig_bind(sockfd, addr, addrlen);
  printf("[MOCK] mock_bind result=%d for fd=%d\n", result, sockfd);
  fflush(stdout);
  return result;
}

int mock_listen(int sockfd, int backlog) {
  printf("[MOCK] mock_listen called: sockfd=%d, backlog=%d\n", sockfd, backlog);
  fflush(stdout);
  
  init_orig_functions();
  
  if (!g_mock_topology.listen_success) {
    errno = EOPNOTSUPP;
    return -1;
  }
  
  // Check if this is a pure mock socket
  struct MockSocket* sock = find_mock_socket(sockfd);
  if (sock && sockfd >= 10000) {
    printf("[MOCK] mock_listen: pure mock socket %d, simulating listen success\n", sockfd);
    fflush(stdout);
    sock->listening = 1;
    return 0; // Simulate success for pure mock sockets
  }
  
  // Use real listen for actual network communication
  int result = orig_listen(sockfd, backlog);
  printf("[MOCK] mock_listen result=%d for fd=%d\n", result, sockfd);
  fflush(stdout);
  return result;
}

int mock_accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen) {
  printf("[MOCK] mock_accept called: sockfd=%d\n", sockfd);
  fflush(stdout);
  
  init_orig_functions();
  
  if (!g_mock_topology.accept_success) {
    errno = ECONNABORTED;
    return -1;
  }
  
  // Check if this is a pure mock socket
  struct MockSocket* sock = find_mock_socket(sockfd);
  if (sock && sockfd >= 10000) {
    if (sock->fd < 0) {
      printf("[MOCK] mock_accept: socket %d was already closed\n", sockfd);
      fflush(stdout);
      errno = EBADF;
      return -1;
    }
    
    if (sock->listening) {
      printf("[MOCK] mock_accept: pure mock socket %d, simulating accept\n", sockfd);
      fflush(stdout);
      
      // Create a new mock socket for the accepted connection
      static int mock_fd_counter = 10000;
      int new_fd = mock_fd_counter++;
      struct MockSocket* new_sock = create_mock_socket(new_fd);
      
      if (new_sock) {
        new_sock->fd = new_fd;
        new_sock->connected = 1;
        new_sock->node_id = sock->node_id; // Same node as listening socket
        new_sock->peer_node_id = -1; // Will be set when peer connects
        
        // Set up shared memory connection
        if (init_shared_memory() == 0) {
          // For now, assume peer is connecting from node 0 or 1 (opposite of current)
          int assumed_peer_node = (sock->node_id == 0) ? 1 : 0;
          new_sock->connection = get_connection(sock->node_id, assumed_peer_node, 29500);
          if (new_sock->connection) {
            new_sock->peer_node_id = assumed_peer_node;
            printf("[MOCK] mock_accept: established shared memory connection node %d <- %d\n", 
                   sock->node_id, assumed_peer_node);
            fflush(stdout);
          }
        }
        
        // Fill in a mock peer address if requested
        if (addr && addrlen && *addrlen >= sizeof(struct sockaddr_in)) {
          struct sockaddr_in* sin = (struct sockaddr_in*)addr;
          sin->sin_family = AF_INET;
          sin->sin_port = htons(29500);
          // Set peer IP based on assumed peer node
          char peer_ip[32];
          int ip_octet = (new_sock->peer_node_id >= 0 && new_sock->peer_node_id < 254) ? 
                         new_sock->peer_node_id + 1 : 1;
          snprintf(peer_ip, sizeof(peer_ip), "10.1.2.%d", ip_octet);
          inet_pton(AF_INET, peer_ip, &sin->sin_addr);
          *addrlen = sizeof(struct sockaddr_in);
        }
        
        printf("[MOCK] mock_accept: returning new mock fd=%d\n", new_fd);
        fflush(stdout);
        return new_fd;
      }
      
      errno = ENOMEM;
      return -1;
    }
  }
  
  // Use real accept for actual network communication
  int result = orig_accept(sockfd, addr, addrlen);
  printf("[MOCK] mock_accept result=%d for fd=%d\n", result, sockfd);
  fflush(stdout);
  return result;
}

int mock_connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen) {
  printf("[MOCK] mock_connect called: sockfd=%d\n", sockfd);
  fflush(stdout);
  
  init_orig_functions();
  
  if (!g_mock_topology.connect_success) {
    errno = ECONNREFUSED;
    return -1;
  }
  
  // Check if this is a mock IP address (10.1.2.x or localhost)
  if (addr->sa_family == AF_INET) {
    struct sockaddr_in *sin = (struct sockaddr_in *)addr;
    uint32_t ip = ntohl(sin->sin_addr.s_addr);
    int port = ntohs(sin->sin_port);
    
    // Check if this is a mock IP (10.1.2.x) or localhost - always use shared memory for these
    if ((ip & 0xFFFFFF00) == 0x0A010100 || ip == 0x7F000001) {
      printf("[MOCK] Connecting to mock IP: %s:%d (using shared memory)\n", inet_ntoa(sin->sin_addr), port);
      fflush(stdout);
      
      // Find or create mock socket
      struct MockSocket* sock = find_mock_socket(sockfd);
      if (!sock) {
        sock = create_mock_socket(sockfd);
      }
      
      if (sock) {
        // Determine peer node ID from IP - handle any IP in 10.1.2.x range
        int peer_node_id = -1;
        if (ip == 0x7F000001) { // localhost -> node 0
          peer_node_id = 0;
        } else {
          int last_octet = ip & 0xFF;
          if (last_octet >= 100) {
            peer_node_id = last_octet - 100; // 10.1.2.100 -> node 0, 10.1.2.101 -> node 1
          } else if (last_octet >= 1 && last_octet <= 10) {
            peer_node_id = last_octet - 1; // 10.1.2.1 -> node 0, 10.1.2.2 -> node 1, etc.
          }
          // Note: 10.1.2.0 is network address, not valid host - leave peer_node_id as -1
        }
        
        if (peer_node_id < 0 || peer_node_id >= MAX_NODES) {
          printf("[MOCK] Invalid peer node ID: %d (from IP %s)\n", peer_node_id, inet_ntoa(sin->sin_addr));
          fflush(stdout);
          errno = ECONNREFUSED;
          return -1;
        }
        
        sock->peer_node_id = peer_node_id;
        
        // Force shared memory initialization
        if (init_shared_memory() != 0) {
          printf("[MOCK] Failed to initialize shared memory\n");
          fflush(stdout);
          errno = ECONNREFUSED;
          return -1;
        }
        
        sock->connection = get_connection(sock->node_id, peer_node_id, port);
        
        if (sock->connection) {
          sock->connected = 1;
          memcpy(&sock->peer_addr, addr, addrlen);
          printf("[MOCK] Mock shared memory connection established: node %d -> node %d on port %d\n", 
                 sock->node_id, peer_node_id, port);
          fflush(stdout);
          return 0;
        } else {
          printf("[MOCK] Failed to establish shared memory connection\n");
          fflush(stdout);
          errno = ECONNREFUSED;
          return -1;
        }
      } else {
        printf("[MOCK] Failed to create mock socket entry\n");
        fflush(stdout);
        errno = ENOMEM;
        return -1;
      }
    }
  }
  
  // Only use real connect for non-mock IPs
  struct MockSocket* sock = find_mock_socket(sockfd);
  if (sock && sockfd >= 10000) {
    // This is a pure mock socket, don't call real connect
    printf("[MOCK] Pure mock socket %d cannot connect to real address\n", sockfd);
    fflush(stdout);
    errno = ECONNREFUSED;
    return -1;
  }
  
  // Use real connect for actual network communication
  int result = orig_connect(sockfd, addr, addrlen);
  printf("[MOCK] mock_connect result=%d for fd=%d\n", result, sockfd);
  fflush(stdout);
  return result;
}

ssize_t mock_send(int sockfd, const void *buf, size_t len, int flags) {
  init_orig_functions();
  
  // Check if this is a mock socket with shared memory connection
  struct MockSocket* sock = find_mock_socket(sockfd);
  if (sock && sock->connected && sock->connection) {
    printf("[MOCK] mock_send: shared memory send on fd=%d, len=%zu, node %d -> %d\n", 
           sockfd, len, sock->node_id, sock->peer_node_id);
    fflush(stdout);
    
    // Send data through shared memory ring buffer
    int bytes_sent = ring_buffer_write(&sock->connection->send_ring, buf, len);
    
    if (bytes_sent == len) {
      printf("[MOCK] mock_send: successfully sent %d bytes via shared memory\n", bytes_sent);
      fflush(stdout);
      return bytes_sent;
    } else if (bytes_sent > 0) {
      printf("[MOCK] mock_send: partial send %d/%zu bytes via shared memory\n", bytes_sent, len);
      fflush(stdout);
      return bytes_sent;
    } else {
      printf("[MOCK] mock_send: send failed via shared memory, setting EAGAIN\n");
      fflush(stdout);
      errno = EAGAIN;
      return -1;
    }
  } else if (sock && sock->connected) {
    printf("[MOCK] mock_send: basic mock send on fd=%d, len=%zu (no shared memory)\n", sockfd, len);
    fflush(stdout);
    // Simple mock without shared memory - just return success
    return len;
  }
  
  // Use real send for actual network communication
  ssize_t result = orig_send(sockfd, buf, len, flags);
  printf("[MOCK] mock_send: real send fd=%d, len=%zu, result=%zd\n", sockfd, len, result);
  return result;
}

ssize_t mock_recv(int sockfd, void *buf, size_t len, int flags) {
  init_orig_functions();
  
  // Check if this is a mock socket with shared memory connection
  struct MockSocket* sock = find_mock_socket(sockfd);
  if (sock && sock->connected && sock->connection) {
    printf("[MOCK] mock_recv: shared memory recv on fd=%d, len=%zu, node %d <- %d\n", 
           sockfd, len, sock->node_id, sock->peer_node_id);
    fflush(stdout);
    
    // First check if peer has sent us data (peer's send_ring is our recv_ring)
    struct MockNodeConnection* peer_conn = NULL;
    pthread_mutex_lock(&g_mock_shm->global_mutex);
    for (int i = 0; i < MAX_CONNECTIONS_PER_NODE; i++) {
      struct MockNodeConnection* conn = &g_mock_shm->connections[sock->peer_node_id][i];
      if (conn->active && conn->peer_node_id == sock->node_id && 
          conn->port == sock->connection->port) {
        peer_conn = conn;
        break;
      }
    }
    pthread_mutex_unlock(&g_mock_shm->global_mutex);
    
    if (peer_conn) {
      // Try to read from peer's send buffer (which is our receive buffer)
      int blocking = !(flags & MSG_DONTWAIT);
      int bytes_received = ring_buffer_read(&peer_conn->send_ring, buf, len, blocking);
      
      if (bytes_received > 0) {
        printf("[MOCK] mock_recv: received %d bytes via shared memory\n", bytes_received);
        fflush(stdout);
        return bytes_received;
      } else if (!blocking) {
        printf("[MOCK] mock_recv: no data available (non-blocking)\n");
        fflush(stdout);
        errno = EAGAIN;
        return -1;
      }
    }
    
    // If no peer connection or no data available, return EAGAIN to let NCCL retry
    printf("[MOCK] mock_recv: no peer connection or data available, setting EAGAIN\n");
    fflush(stdout);
    errno = EAGAIN;
    return -1;
  } else if (sock && sockfd >= 10000) {
    // This is a pure mock socket but not connected or no shared memory connection
    printf("[MOCK] mock_recv: pure mock socket %d, not connected or no shared memory\n", sockfd);
    fflush(stdout);
    
    // For pure mock sockets that aren't connected, we can't receive real data
    // Return EAGAIN to let NCCL retry or handle appropriately
    printf("[MOCK] mock_recv: pure mock - no connection, setting EAGAIN\n");
    fflush(stdout);
    errno = EAGAIN;
    return -1;
  } else if (sock && sock->connected) {
    printf("[MOCK] mock_recv: basic mock recv on fd=%d, len=%zu (no shared memory)\n", sockfd, len);
    fflush(stdout);
    
    // For basic mock without shared memory, we can't pass real data between processes
    // Return EAGAIN to indicate no data available
    errno = EAGAIN;
    printf("[MOCK] mock_recv: basic mock - no shared memory, setting EAGAIN\n");
    fflush(stdout);
    return -1;
  }
  
  // Check if this is a pure mock socket - don't call real recv on these
  if (sockfd >= 10000) {
    printf("[MOCK] mock_recv: pure mock socket %d, but no mock handler matched, setting EBADF\n", sockfd);
    fflush(stdout);
    errno = EBADF;
    return -1;
  }
  
  // Use real recv for actual network communication
  ssize_t result = orig_recv(sockfd, buf, len, flags);
  if (result == -1) {
    printf("[MOCK] mock_recv: real recv fd=%d, len=%zu, result=%zd, errno=%d (%s)\n", 
           sockfd, len, result, errno, strerror(errno));
  } else {
    printf("[MOCK] mock_recv: real recv fd=%d, len=%zu, result=%zd\n", sockfd, len, result);
  }
  return result;
}

int mock_close(int fd) {
  printf("[MOCK] mock_close called: fd=%d\n", fd);
  fflush(stdout);
  
  init_orig_functions();
  
  // Clean up mock socket if it exists
  struct MockSocket* sock = find_mock_socket(fd);
  if (sock) {
    printf("[MOCK] Cleaning up mock socket for fd=%d\n", fd);
    fflush(stdout);
    
    // Clean up shared memory connection
    if (sock->connection) {
      pthread_mutex_lock(&g_mock_shm->global_mutex);
      
      // Mark connection as inactive
      sock->connection->active = 0;
      
      // Clean up ring buffer semaphores
      sem_destroy(&sock->connection->send_ring.data_available);
      sem_destroy(&sock->connection->send_ring.space_available);
      sem_destroy(&sock->connection->recv_ring.data_available);
      sem_destroy(&sock->connection->recv_ring.space_available);
      
      // Clean up mutexes
      pthread_mutex_destroy(&sock->connection->send_ring.mutex);
      pthread_mutex_destroy(&sock->connection->recv_ring.mutex);
      
      printf("[MOCK] Cleaned up shared memory connection: node %d -> node %d\n",
             sock->node_id, sock->peer_node_id);
      fflush(stdout);
      
      pthread_mutex_unlock(&g_mock_shm->global_mutex);
    }
    
    sock->fd = -1;
    sock->connected = 0;
    sock->connection = NULL;
    sock->peer_node_id = -1;
    
    // If this is a pure mock socket (fd >= 10000), don't call real close
    if (fd >= 10000) {
      printf("[MOCK] Pure mock socket %d closed (no real fd to close)\n", fd);
      fflush(stdout);
      return 0;
    }
  }
  
  // Use real close for real file descriptors
  int result = orig_close(fd);
  printf("[MOCK] mock_close result=%d for fd=%d\n", result, fd);
  fflush(stdout);
  return result;
}

// Utility function to cleanup shared memory (called on exit)
static void cleanup_shared_memory() {
  if (g_mock_shm) {
    printf("[MOCK] Cleaning up shared memory for node %d\n", g_node_id);
    fflush(stdout);
    
    // Clean up all connections for this node
    if (g_node_id >= 0 && g_node_id < MAX_NODES) {
      pthread_mutex_lock(&g_mock_shm->global_mutex);
      
      for (int i = 0; i < MAX_CONNECTIONS_PER_NODE; i++) {
        struct MockNodeConnection* conn = &g_mock_shm->connections[g_node_id][i];
        if (conn->active) {
          conn->active = 0;
          
          // Clean up ring buffer semaphores
          sem_destroy(&conn->send_ring.data_available);
          sem_destroy(&conn->send_ring.space_available);
          sem_destroy(&conn->recv_ring.data_available);
          sem_destroy(&conn->recv_ring.space_available);
          
          // Clean up mutexes
          pthread_mutex_destroy(&conn->send_ring.mutex);
          pthread_mutex_destroy(&conn->recv_ring.mutex);
          
          printf("[MOCK] Cleaned up connection %d for node %d\n", i, g_node_id);
          fflush(stdout);
        }
      }
      
      pthread_mutex_unlock(&g_mock_shm->global_mutex);
    }
    
    // Unmap shared memory
    munmap(g_mock_shm, sizeof(struct MockSharedMemory));
    g_mock_shm = NULL;
    
    printf("[MOCK] Shared memory cleanup completed\n");
    fflush(stdout);
  }
}

// Set node hostname and IP for testing
void ncclMockSetNodeInfo(int node_id, const char* hostname, const char* ip) {
  if (init_shared_memory() != 0) {
    return;
  }
  
  if (node_id >= 0 && node_id < MAX_NODES) {
    pthread_mutex_lock(&g_mock_shm->global_mutex);
    
    if (hostname) {
      strncpy(g_mock_shm->node_hostnames[node_id], hostname, sizeof(g_mock_shm->node_hostnames[node_id]) - 1);
    }
    if (ip) {
      strncpy(g_mock_shm->node_ips[node_id], ip, sizeof(g_mock_shm->node_ips[node_id]) - 1);
    }
    
    if (g_mock_shm->num_nodes <= node_id) {
      g_mock_shm->num_nodes = node_id + 1;
    }
    
    pthread_mutex_unlock(&g_mock_shm->global_mutex);
    
    printf("[MOCK] Set node %d info: hostname=%s, ip=%s\n", 
           node_id, hostname ? hostname : "unchanged", ip ? ip : "unchanged");
    fflush(stdout);
  }
}

// Get node information for debugging
void ncclMockPrintNodeInfo() {
  if (g_mock_shm && g_mock_shm->initialized) {
    printf("[MOCK] Node information:\n");
    printf("[MOCK] Current node ID: %d (%s)\n", g_node_id, g_node_hostname);
    printf("[MOCK] Total nodes: %d\n", g_mock_shm->num_nodes);
    
    for (int i = 0; i < g_mock_shm->num_nodes; i++) {
      printf("[MOCK] Node %d: hostname=%s, ip=%s\n", 
             i, g_mock_shm->node_hostnames[i], g_mock_shm->node_ips[i]);
    }
    fflush(stdout);
  }
}

// Function to reset shared memory (useful for testing)
void ncclMockResetSharedMemory() {
  if (g_mock_shm && g_mock_shm->initialized) {
    printf("[MOCK] Resetting shared memory\n");
    fflush(stdout);
    
    pthread_mutex_lock(&g_mock_shm->global_mutex);
    
    // Clean up existing connections
    for (int i = 0; i < MAX_NODES; i++) {
      for (int j = 0; j < MAX_CONNECTIONS_PER_NODE; j++) {
        struct MockNodeConnection* conn = &g_mock_shm->connections[i][j];
        if (conn->active) {
          conn->active = 0;
          sem_destroy(&conn->send_ring.data_available);
          sem_destroy(&conn->send_ring.space_available);
          sem_destroy(&conn->recv_ring.data_available);
          sem_destroy(&conn->recv_ring.space_available);
          pthread_mutex_destroy(&conn->send_ring.mutex);
          pthread_mutex_destroy(&conn->recv_ring.mutex);
        }
      }
    }
    
    // Reset node count and hostnames
    g_mock_shm->num_nodes = 0;
    memset(g_mock_shm->node_hostnames, 0, sizeof(g_mock_shm->node_hostnames));
    memset(g_mock_shm->node_ips, 0, sizeof(g_mock_shm->node_ips));
    
    pthread_mutex_unlock(&g_mock_shm->global_mutex);
    
    printf("[MOCK] Shared memory reset completed\n");
    fflush(stdout);
  }
  
  // Also reset local node ID
  g_node_id = -1;
  memset(g_node_hostname, 0, sizeof(g_node_hostname));
}

// Destructor to cleanup shared memory on library unload
__attribute__((destructor))
static void cleanup_mock_topology() {
  cleanup_shared_memory();
}

int mock_dup2(int oldfd, int newfd) {
  printf("[MOCK] mock_dup2 called: oldfd=%d, newfd=%d\n", oldfd, newfd);
  fflush(stdout);
  
  init_orig_functions();
  
  // Check if oldfd is a mock socket
  struct MockSocket* old_sock = find_mock_socket(oldfd);
  if (old_sock) {
    // For mock sockets, we can't really duplicate the file descriptor in a meaningful way
    // since they don't correspond to real kernel file descriptors.
    // Just return success without doing anything - this is a mock after all.
    printf("[MOCK] mock_dup2: oldfd=%d is a mock socket, returning success without actual duplication\n", oldfd);
    fflush(stdout);
    return newfd;
  }
  
  // For real file descriptors, use the real dup2
  int result = orig_dup2(oldfd, newfd);
  printf("[MOCK] mock_dup2 result=%d for oldfd=%d, newfd=%d\n", result, oldfd, newfd);
  fflush(stdout);
  return result;
}
} // extern "C"

#endif