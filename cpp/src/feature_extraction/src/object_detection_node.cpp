//
// Created by ckelton on 6/2/26.
//
#include <signal.h>
#include <execinfo.h>

#include "LidarFeatureExtractionNode.h"


static void segfault_handler(int sig) {
    void* array[30];
    size_t size = backtrace(array, 30);
    fprintf(stderr, "\n[Velodyne Error] Caught signal %d (SIGSEGV). Stack trace:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    _exit(1);
}

int main(int argc, char** argv) {
    signal(SIGSEGV, segfault_handler);
    signal(SIGABRT, segfault_handler);
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LidarFeatureExtractionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
}
