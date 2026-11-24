// ascend/src/engine/missing_linux.cpp - Linux missing-entry handler and crash backtrace for Strata Ascend.
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <execinfo.h>
#include <exception>
#include <mutex>
#include <set>
#include <string>
#include <unistd.h>

#include "strata/kernels/cvec.hpp"

namespace strata::metal_engine {

void missing(const char* name) {
    static const bool log_only = [] {
        const char* v = std::getenv("STRATA_ASCEND_STUBS");
        return v != nullptr && std::string(v) == "log";
    }();
    if (!log_only) {
        std::fprintf(stderr, "strata ascend: %s has no Ascend implementation yet\n", name);
        std::fflush(stderr);
        std::_Exit(86);
    }
    static std::mutex mu;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lk(mu);
    if (seen.insert(name).second) std::fprintf(stderr, "strata ascend: STUB %s\n", name);
}

}  // namespace strata::metal_engine

namespace strata::kernels {
const Cvec& cvec() {
    static const Cvec none{};
    return none;
}
}  // namespace strata::kernels

namespace {
[[noreturn]] void ascend_terminate() {
    void* fr[64];
    const int n = backtrace(fr, 64);
    const char msg[] = "strata ascend: terminate - call stack:\n";
    (void) !write(2, msg, sizeof msg - 1);
    backtrace_symbols_fd(fr, n, 2);
    std::abort();
}
const bool g_term = (std::set_terminate(ascend_terminate), true);

void ascend_crash(int sig, siginfo_t* si, void*) {
    void* fr[64];
    const int n = backtrace(fr, 64);
    char msg[160];
    const int m = std::snprintf(msg, sizeof msg, "strata ascend: crashed (signal %d at %p) - call stack:\n",
                                sig, si ? si->si_addr : nullptr);
    (void) !write(2, msg, (size_t) m);
    backtrace_symbols_fd(fr, n, 2);
    signal(sig, SIG_DFL);
    raise(sig);
}
bool set_crash() {
    struct sigaction sa {};
    sa.sa_sigaction = ascend_crash;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
    return true;
}
const bool g_crash = set_crash();
}  // namespace
