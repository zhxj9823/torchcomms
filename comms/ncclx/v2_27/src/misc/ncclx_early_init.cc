/*************************************************************************
 * NEX CPU Emulation: Early initialization for ncclx
 *
 * Problem: ncclx links Folly/Thrift which have C++ static constructors that
 * call std::chrono::steady_clock::now() → clock_gettime(). When running under
 * NEX, accvm.so interposes clock_gettime() with a wrapper that calls through
 * orig_clock_gettime, a function pointer that is initially NULL and only
 * resolved during accvm_init(). If libnccl.so's .init_array runs before
 * accvm_init(), the NULL dereference causes SIGSEGV.
 *
 * Fix: This constructor runs at priority 101 (before default-priority C++
 * static constructors like Folly's) and patches accvm's orig_clock_gettime
 * via dlsym so that subsequent clock_gettime() calls work correctly.
 ************************************************************************/

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <time.h>
#include <stddef.h>

typedef int (*clock_gettime_fn_t)(clockid_t, struct timespec *);

__attribute__((constructor(101)))
static void ncclx_ensure_accvm_clock_init(void) {
    // Try to find accvm's orig_clock_gettime global variable.
    // If accvm.so is not loaded (not running under NEX), dlsym returns NULL
    // and we do nothing.
    clock_gettime_fn_t *orig_ptr =
        (clock_gettime_fn_t *)dlsym(RTLD_DEFAULT, "orig_clock_gettime");

    if (orig_ptr != NULL && *orig_ptr == NULL) {
        // accvm.so is loaded but hasn't initialized yet.
        // Resolve the real clock_gettime (after our library in the link order,
        // which skips accvm's interposer and finds libc's implementation).
        clock_gettime_fn_t real_fn =
            (clock_gettime_fn_t)dlsym(RTLD_NEXT, "clock_gettime");
        if (real_fn != NULL) {
            *orig_ptr = real_fn;
        }
    }
}
