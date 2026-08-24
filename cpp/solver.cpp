#include <iostream>
#include <numeric>
#include <queue>
#include <mutex>
#include <condition_variable> 
#include <atomic>
#include <thread>
#include <chrono>
#include <CLI11.hpp>
#include <nvml.h>

#define MONITOR_INTERVAL_MILIS 200

struct ParsedArgs {
    int grid_size = 0;
    std::string framework;
    std::string workload;
};

struct Summary {
    unsigned int    gpu_max     = 0;
    double          gpu_avg     = 0.0;
    unsigned int    mem_util_max    = 0;
    double          mem_util_avg    = 0.0;
    double          power_max   = 0.0;
    double          power_avg   = 0.0;
    int             n_samples   = 0;
};

struct GpuSample {
    unsigned int gpu_util;
    unsigned int mem_util;
    unsigned long long mem_used;
    unsigned long long mem_total;
    double power_w;
};

class GpuMonitor {
public:
    GpuMonitor() {
        nvmlInit();
        nvmlDeviceGetHandleByIndex(0, &handle_);

        char name[NVML_DEVICE_NAME_BUFFER_SIZE];
        nvmlDeviceGetName(handle_, name, sizeof(name));
        device_name_ = name;

        char driver[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
        nvmlSystemGetDriverVersion(driver, sizeof(driver));
        driver_version_ = driver;
    }

    GpuMonitor(const GpuMonitor&) = delete;
    GpuMonitor& operator=(const GpuMonitor&) = delete;

    ~GpuMonitor() {
        if (thread_.joinable()) {
            stop_flag_.store(true);
            thread_.join();
        }

        nvmlShutdown();
    }

    const std::string& device_name() const { return device_name_; }
    const std::string& driver_version() const { return driver_version_; }

    void start() {
        nvmlUtilization_t warm{};
        nvmlDeviceGetUtilizationRates(handle_, &warm);
        ready_flag_.store(true);
        thread_ = std::thread(&GpuMonitor::run, this);
    }

    bool wait_ready(std::chrono::milliseconds timeout) const {
        auto start = std::chrono::steady_clock::now();

        while (!ready_flag_.load()) {
            if (std::chrono::steady_clock::now() - start > timeout) return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        return true;
    }

    void stop() {
        stop_flag_.store(true);
        if (thread_.joinable()) thread_.join();
        summary_ = compute_summary();
    }

    GpuSample latest() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return current_;
    }

    Summary summary() const { return summary_; }

private:
    void run() {
        while (!stop_flag_.load()) {
            nvmlUtilization_t util{};
            nvmlMemory_t mem{};
            unsigned int power_mw = 0;

            nvmlDeviceGetUtilizationRates(handle_, &util);
            nvmlDeviceGetMemoryInfo(handle_, &mem);
            nvmlDeviceGetPowerUsage(handle_, &power_mw);

            GpuSample sample{
                util.gpu, util.memory, mem.used, mem.total, power_mw / 1000.0
            };

            {
                std::lock_guard<std::mutex> lock(mutex_);
                current_ = sample;
            }

            gpu_samples_.push_back(util.gpu);
            mem_util_samples_.push_back(util.memory);
            power_samples_.push_back(power_mw / 1000.0);

            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    Summary compute_summary() const {
        Summary s{};
        s.n_samples = gpu_samples_.size();
        if (s.n_samples == 0) return s;

        s.gpu_max = *std::max_element(gpu_samples_.begin(), gpu_samples_.end());
        s.gpu_avg = std::accumulate(mem_util_samples_.begin(), mem_util_samples_.end(), 0.0)
                    / static_cast<double>(s.n_samples);
        
        s.power_max = *std::max_element(power_samples_.begin(), power_samples_.end());
        s.power_avg = std::accumulate(power_samples_.begin(), power_samples_.end(), 0.0)
                    / static_cast<double>(s.n_samples);
        
        return s;
    }

    nvmlDevice_t handle_{};
    std::string device_name_;
    std::string driver_version_;
    
    std::atomic<bool> ready_flag_{false};
    std::atomic<bool> stop_flag_{false};
    std::thread thread_;

    mutable std::mutex mutex_;
    GpuSample current_{};

    std::vector<unsigned int> gpu_samples_;
    std::vector<unsigned int> mem_util_samples_;
    std::vector<double> power_samples_;

    Summary summary_{};
};

ParsedArgs parse_args(int argc, char** argv) {
    CLI::App app{""};
    ParsedArgs args;

    app.add_option("--grid-size", args.grid_size, "Number of grid points along each spatial dimension.")
        ->required();

    app.add_option("--framework", args.framework, "Computational framework to use.")
        ->required()
        ->check(CLI::IsMember({"cupy", "numba"}));

    app.add_option("--workload", args.workload,
                    "Numerical method to calculate: "
                    "Monte Carlo, Conjugate Gradient, Direct Coulomb Simulation, "
                    "Spectral, Sparse Matrix Vector, Stencil.")
        ->required()
        ->check(CLI::IsMember({"mc", "cg", "dcs", "spectral", "spmv", "stencil"}));

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        std::exit(app.exit(e));
    }

    return args;
}

int main(int argc, char** argv) {
    ParsedArgs args = parse_args(argc, argv);

    std::printf("Framework:      %s\n", args.framework);
    std::printf("Workload:       %s\n", args.workload);
    std::printf("Grid size:     [%d, %d]\n", args.grid_size, args.grid_size);

    //TODO

    return 0;
}
