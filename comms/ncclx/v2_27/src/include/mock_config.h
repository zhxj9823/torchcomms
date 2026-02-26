/*************************************************************************
 * Copyright (c) 2025, Mock Topology Configuration for NCCL
 *
 * Configuration and control for NCCL topology mocking
 ************************************************************************/

#ifndef NCCL_MOCK_CONFIG_H_
#define NCCL_MOCK_CONFIG_H_

// Uncomment to enable topology mocking
#define NCCL_MOCK_TOPOLOGY

#ifdef NCCL_MOCK_TOPOLOGY

// Mock topology configuration structure
struct NcclMockConfig {
  // Network topology configuration
  struct {
    int enabled;
    char interfaces[256];    // Format: "eth0:192.168.1.100/24,ib0:10.0.0.100/16"
    char hostname[64];
    int simulate_ib;         // Simulate InfiniBand interfaces
    int simulate_nvlink;     // Simulate NVLink topology
  } network;
  
  // System topology configuration
  struct {
    int num_gpus;            // Number of mock GPUs
    int numa_nodes;          // Number of NUMA nodes
    char pci_topology[512];  // PCI topology description
    int nvlink_matrix[8][8]; // NVLink connectivity matrix
  } hardware;
  
  // Communication behavior
  struct {
    int latency_us;          // Simulated network latency in microseconds
    int bandwidth_gbps;      // Simulated bandwidth in Gbps
    int error_rate;          // Error injection rate (0-100%)
    int async_progress;      // Enable async progress simulation
  } performance;
  
  // Debug and logging
  struct {
    int log_syscalls;        // Log all syscall activity
    int trace_topology;      // Trace topology discovery
    char log_file[256];      // Log file path
  } debug;
};

// Global configuration instance
extern struct NcclMockConfig g_nccl_mock_config;

// Configuration functions
void ncclMockLoadConfig();
void ncclMockSaveConfig();
void ncclMockSetDefaults();
int ncclMockParseEnv();

// Mock topology description for different scenarios
void ncclMockSetupSingleNode();           // Single node with multiple GPUs
void ncclMockSetupMultiNode();            // Multi-node cluster
void ncclMockSetupDGX();                  // DGX-like topology
void ncclMockSetupCustom(const char* config); // Custom topology from string

#endif // NCCL_MOCK_TOPOLOGY

#endif // NCCL_MOCK_CONFIG_H_
