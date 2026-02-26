#ifndef CUDA_EMULATOR_HPP
#define CUDA_EMULATOR_HPP

#include <vector>
#include <functional>
#include <memory>
#include <iostream>
#include <boost/fiber/all.hpp>
#include <unordered_map>
#include <atomic>
#include <vector_types.h>
// #include <light/log.h>

// Thread context for each CUDA thread
struct ThreadContext {
    dim3 threadIdx;
    dim3 blockIdx; 
    dim3 blockDim;
    dim3 gridDim;
    
    ThreadContext(dim3 tid, dim3 bid, dim3 bdim, dim3 gdim)
        : threadIdx(tid), blockIdx(bid), blockDim(bdim), gridDim(gdim) {}
};

class SharedMemory;
class NamedBarrierManager;

// Thread-local context for fiber-local storage (per pthread kernel)
extern thread_local boost::fibers::fiber_specific_ptr<ThreadContext> current_thread_ctx;
extern thread_local SharedMemory* current_shared_mem;
extern thread_local NamedBarrierManager* current_named_barrier_manager;


struct CudaThread {
    boost::fibers::fiber fiber;
    ThreadContext context;
    bool completed;
    int thread_id;
    
    CudaThread(int id, dim3 tid, dim3 bid, dim3 bdim, dim3 gdim)
        : context(tid, bid, bdim, gdim), completed(false), thread_id(id) {}
};


// all barrier/sync functions
void barrier_sync(int name);
void barrier_sync(int name, int nThreads);
void barrier_sync_aligned(int name);
void barrier_sync_aligned(int name, int nThreads);
bool barrier_red_or(bool vote, int name);
bool barrier_red_or(bool vote, int name, int nThreads);
void __syncthreads();
int __any_sync(unsigned mask, int predicate);
int __all_sync(unsigned mask, int predicate);
unsigned __ballot_sync(unsigned mask, int predicate);
void __syncwarp();
void __threadfence_block();
void __threadfence();
void __threadfence_system();

// dim functions
dim3 threadIdx();
dim3 blockIdx();
dim3 blockDim();
dim3 gridDim();

using CudaKernel = std::function<void()>;

#define WARP_SIZE 32


// Shared memory layout manager for tracking allocations (per-pthread)
class SharedMemoryLayout {
private:
    static thread_local size_t static_memory_used_;
    
public:
    // Register a static shared memory allocation
    static size_t allocate_static(size_t size, size_t alignment = 1) {
        // Each pthread maintains its own allocation state
        size_t aligned_offset = (static_memory_used_ + alignment - 1) & ~(alignment - 1);
        size_t old_offset = aligned_offset;
        static_memory_used_ = aligned_offset + size;
        
        return old_offset;
    }
    
    static size_t get_dynamic_offset() {
        return static_memory_used_;
    }
    
    static void reset() {
        static_memory_used_ = 0;
    }

    static size_t get_static_memory_used() { return static_memory_used_; }
};

// shared memory accessor
template<typename T>
class shared_memory {
private:
    size_t offset_;
    size_t size_;
    bool is_dynamic_;

public:

    explicit shared_memory(size_t byte_size) 
        : size_(byte_size), is_dynamic_(false) {
        offset_ = SharedMemoryLayout::allocate_static(byte_size, alignof(T));
    }
    
    explicit shared_memory() : offset_(0), size_(0), is_dynamic_(true) {}
    
    T* data() const;
    
    T& operator[](size_t index) const {
        return data()[index];
    }
    
    operator T*() const {
        return data();
    }
    
    T& operator*() const {
        return *data();
    }
    
    T* operator->() const {
        return data();
    }
    
    size_t offset() const { 
        if (is_dynamic_) {
            return SharedMemoryLayout::get_dynamic_offset();
        } else {
            return offset_;
        }
    }
    size_t size() const { return size_; }
    size_t element_count() const { return size_ / sizeof(T); }
    bool is_dynamic() const { return is_dynamic_; }
};

// Shared memory manager (per-block)
class SharedMemory {
private:
    std::vector<std::vector<char>> block_memories;  // One per block
    size_t total_size_;
    size_t static_size_;
    dim3 grid_dim_;

public:
    SharedMemory(size_t total_size, size_t static_size, dim3 grid_dim) 
        : total_size_(total_size), static_size_(static_size), grid_dim_(grid_dim) {
        size_t num_blocks = grid_dim.x * grid_dim.y * grid_dim.z;
        block_memories.resize(num_blocks);
        for (auto& memory : block_memories) {
            memory.resize(total_size, 0);
        }
    }
    
    template<typename T>
    T* get(size_t offset = 0) {
        // Get block ID from current thread context
        if (!current_thread_ctx.get()) {
            throw std::runtime_error("No active thread context");
        }
        
        dim3 bid = current_thread_ctx->blockIdx;
        size_t block_id = bid.z * grid_dim_.x * grid_dim_.y + bid.y * grid_dim_.x + bid.x;
        
        if (block_id >= block_memories.size()) {
            throw std::runtime_error("Block ID out of range");
        }
        
        if (offset >= total_size_) {
            throw std::runtime_error("Shared memory access out of bounds");
        }
        
        return reinterpret_cast<T*>(block_memories[block_id].data() + offset);
    }
    
    // Get dynamic shared memory (starts after static allocations)
    template<typename T>
    T* get_dynamic() {
        size_t dynamic_offset = SharedMemoryLayout::get_dynamic_offset();
        return get<T>(dynamic_offset);
    }
    
    void* raw_ptr(size_t offset = 0) {
        if (!current_thread_ctx.get()) {
            throw std::runtime_error("No active thread context");
        }
        
        dim3 bid = current_thread_ctx->blockIdx;
        size_t block_id = bid.z * grid_dim_.x * grid_dim_.y + bid.y * grid_dim_.x + bid.x;
        
        if (block_id >= block_memories.size() || offset >= total_size_) {
            throw std::runtime_error("Shared memory access out of bounds");
        }
        
        return block_memories[block_id].data() + offset;
    }
    
    size_t total_size() const { return total_size_; }
    size_t static_size() const { return static_size_; }
    size_t dynamic_size() const { return total_size_ - static_size_; }
    
    void clear() { 
        for (auto& memory : block_memories) {
            std::fill(memory.begin(), memory.end(), 0);
        }
    }
};


template<typename T>
inline T* shared_memory<T>::data() const {
    if(!current_shared_mem) {
        throw std::runtime_error("No shared memory allocated for this kernel");
    }
    if (is_dynamic_) {
        return current_shared_mem->get_dynamic<T>();
    } else {
        return current_shared_mem->get<T>(offset_);
    }
}


// Named barrier manager for supporting CUDA barriers
class NamedBarrierManager {
public:
    struct BarrierData {
        std::unique_ptr<boost::fibers::barrier> barrier;
        std::vector<bool> votes;
        std::atomic<int> vote_count{0};
        std::atomic<bool> result{false}; // For vote results
        std::atomic<unsigned> barrier_result_uint{0}; // For __ballot_sync results
        std::mutex mutex;
        int expected_threads;
        
        BarrierData(int nThreads) : expected_threads(nThreads) {
            barrier = std::make_unique<boost::fibers::barrier>(nThreads);
            // potentially buggy, if the scope is larger than a warp, but for votes, the scope is normally <= a warp
            votes.resize(WARP_SIZE, false);
        }
    };
    
private:
    std::unordered_map<int, std::unique_ptr<BarrierData>> barriers_;
    std::mutex manager_mutex_;
    int default_thread_count_;
    
public:
    NamedBarrierManager(int default_threads) : default_thread_count_(default_threads) {}
    
    BarrierData* get_barrier(int name, int nThreads = 0) {
        std::lock_guard<std::mutex> lock(manager_mutex_);
        if (barriers_.find(name) == barriers_.end()) {
            int thread_count = nThreads > 0 ? nThreads : default_thread_count_;
            barriers_[name] = std::make_unique<BarrierData>(thread_count);
        }
        return barriers_[name].get();
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(manager_mutex_);
        barriers_.clear();
    }
};

// CUDA emulator with Boost.Fiber
class CudaEmulator {
private:
    std::vector<CudaThread> cuda_threads;
    std::unique_ptr<SharedMemory> shared_mem;
    std::unique_ptr<NamedBarrierManager> named_barrier_manager;
    
    dim3 grid_dim;
    dim3 block_dim;
    size_t dyn_shared_mem_size;
    static size_t static_max_shared_mem_size;

public:
    CudaEmulator(dim3 grid_dim, dim3 block_dim, size_t shared_mem_size = 0)
        : grid_dim(grid_dim), block_dim(block_dim), dyn_shared_mem_size(shared_mem_size) {
        
        // size_t total_threads = grid_dim.x * grid_dim.y * grid_dim.z * 
        //                       block_dim.x * block_dim.y * block_dim.z;
        size_t block_in_threads = block_dim.x * block_dim.y * block_dim.z;
        
        // Create named barrier manager with default block level scope
        named_barrier_manager = std::make_unique<NamedBarrierManager>(block_in_threads);
        
        // Create all thread contexts
        int thread_id = 0;
        for (unsigned int bz = 0; bz < grid_dim.z; bz++) {
            for (unsigned int by = 0; by < grid_dim.y; by++) {
                for (unsigned int bx = 0; bx < grid_dim.x; bx++) {
                    for (unsigned int tz = 0; tz < block_dim.z; tz++) {
                        for (unsigned int ty = 0; ty < block_dim.y; ty++) {
                            for (unsigned int tx = 0; tx < block_dim.x; tx++) {
                                cuda_threads.emplace_back(
                                    thread_id++,
                                    dim3(tx, ty, tz),      // threadIdx
                                    dim3(bx, by, bz),      // blockIdx
                                    block_dim,             // blockDim
                                    grid_dim               // gridDim
                                );
                            }
                        }
                    }
                }
            }
        }
        
        printf("Created %ld CUDA threads\n", cuda_threads.size());
    }
    
    ~CudaEmulator() {
        printf("Destroying CudaEmulator...\n");

        // Make sure all fibers are joined before cleanup
        for (auto& cuda_thread : cuda_threads) {
            if (cuda_thread.fiber.joinable()) {
                try {
                    cuda_thread.fiber.join();
                } catch (const std::exception& e) {
                    std::cerr << "Error joining fiber " << cuda_thread.thread_id 
                              << ": " << e.what() << std::endl;
                }
            }
        }
        
        // Barrier and shared memory will be destroyed by unique_ptr
        printf("CudaEmulator destroyed\n");
    }
    
    // Launch a kernel
    void launch_kernel(CudaKernel kernel) {
        printf("Launching kernel with %ld threads\n", cuda_threads.size());
        
        // Reset shared memory layout for this kernel launch
        SharedMemoryLayout::reset();
      
        shared_mem = std::make_unique<SharedMemory>(static_max_shared_mem_size, 0, grid_dim);

        current_shared_mem = shared_mem.get();

        current_named_barrier_manager = named_barrier_manager.get();
        
        // Launch all fibers
        for (auto& cuda_thread : cuda_threads) {

            ThreadContext thread_context = cuda_thread.context;

            int thread_id = cuda_thread.thread_id;
            
            cuda_thread.fiber = boost::fibers::fiber([thread_context, thread_id, kernel, this]() {

                current_thread_ctx.reset(new ThreadContext(thread_context));
                
                try {
                    // Execute kernel
                    kernel();
                } catch (const std::exception& e) {
                    std::cerr << "Error in fiber " << thread_id 
                              << ": " << e.what() << std::endl;
                }

            });
        }
        
        // Wait for all fibers to complete
        for (auto& cuda_thread : cuda_threads) {
            if (cuda_thread.fiber.joinable()) {
                cuda_thread.fiber.join();
            }
            cuda_thread.completed = true;
        }
        
        current_shared_mem = nullptr;
        current_named_barrier_manager = nullptr;

        printf("Kernel execution completed\n");
    }
    
    void print_thread_info() {
        printf("Grid dimensions: (%d,%d,%d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
        printf("Block dimensions: (%d,%d,%d)\n", block_dim.x, block_dim.y, block_dim.z);
        printf("Total threads: %ld\n", cuda_threads.size());
        printf("Dynamic shared memory size: %ld bytes\n", dyn_shared_mem_size);



        int active = 0, completed = 0;
        for (const auto& t : cuda_threads) {
            if (t.completed) completed++;
            else active++;
        }
        printf("Thread states - Active: %d, Completed: %d\n", active, completed);
    }
};

#endif // CUDA_EMULATOR_HPP
