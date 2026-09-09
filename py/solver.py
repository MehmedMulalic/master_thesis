from argparse import ArgumentParser
from queue import Empty
import multiprocessing as mp
import pynvml
import time

MONITOR_INTERVAL = 0.01

FRAMEWORKS = {
    "cupy": "CuPy",
    "numba": "Numba"
}

NUM_METHOD = {
    "mc": "Monte Carlo",
    "cg": "Conjugate Gradient",
    "dcs": "Direct Coulomb Summation",
    "spectral": "Spectral",
    "spmv": "Sparse Matrix-Vector Multiplication",
    "stencil": "Stencil",
}

def parse_args():
    parser = ArgumentParser(
        description="Run the 3D heat equation benchmark.",
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        required=True,
        help="Number of grid points along each spatial dimension.",
    )

    parser.add_argument(
        "--framework",
        choices=["cupy", "numba"],
        required=True,
        help="Computational framework to use.",
    )

    parser.add_argument(
        "--workload",
        choices=["mc", "cg", "dcs", "spectral", "spmv", "stencil"],
        required=True,
        help="Numerical method to calculate: " \
            "Monte Carlo, Conjugate Gradient, Direct Coulomb Simulation, " \
            "Spectral, Sparse Matrix Vector, Stencil.",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run a CPU reference calculation and compare the result.",
    )

    ### Workload specific arguments

    # Monte Carlo
    parser.add_argument(
        "--N",
        type=int,
        help="Problem size N.",
    )

    parser.add_argument(
        "--d",
        type=int,
        help="Dimension d.",
    )

    parser.add_argument(
        "--bound",
        type=float,
        default=6.0,
        help="Domain bound used for volume for Monte Carlo. Default value is 6.0, " \
            "accurate enough for double-point floating values.",
    )

    # Conjugate Gradient
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="Maximum number of iterations. Default is 1000.",
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="Convergence tolerance. Default is 1e-10.",
    )

    args = parser.parse_args()

    if args.workload == "mc":
        if args.N is None or args.d is None or args.bound is None:
            parser.error("--workload mc requires --N, --d, and --bound")

    elif args.workload == "cg":
        if args.N is None:
            parser.error("--workload cg requires --N")

    return args

def monitor(queue, stop_event, ready_event, kernel_start_event):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    queue.put({
        "type": "gpu_stats",
        "name": pynvml.nvmlDeviceGetName(handle),
        "driver": pynvml.nvmlSystemGetDriverVersion(),
    })

    gpu_samples = []
    mem_util_samples = []
    power_samples = []

    # Warm-up run
    _ = pynvml.nvmlDeviceGetUtilizationRates(handle)
    ready_event.set()

    # Wait for kernel to start, then start monitoring
    while not kernel_start_event.is_set():
        if stop_event.is_set():
            pynvml.nvmlShutdown()
            return

        kernel_start_event.wait(timeout=0.1)

    # Gather information
    try:
        while not stop_event.is_set():
            util     = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem      = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)

            gpu_samples.append(util.gpu)
            mem_util_samples.append(util.memory)
            power_samples.append(power_mw / 1000)
            
            queue.put({
                "type": "gpu",
                "gpu": util.gpu,
                "mem_used": mem.used,
                "mem_total": mem.total
            })

            time.sleep(MONITOR_INTERVAL)
    finally:
        if gpu_samples:
            queue.put({
                "type": "gpu_summary",
                "gpu_max": max(gpu_samples),
                "gpu_avg": sum(gpu_samples) / len(gpu_samples),
                "mem_util_max": max(mem_util_samples),
                "mem_util_avg": sum(mem_util_samples) / len(mem_util_samples),
                "power_max": max(power_samples),
                "power_avg": sum(power_samples) / len(power_samples),
                "num_samples": len(gpu_samples),
            })
        pynvml.nvmlShutdown()

def workload(queue, args, kernel_start_event):
    if args.framework == "cupy":
        from cupy_methods import (
            cupy_monte_carlo, cupy_conjugate_gradient, cupy_dcs,
            cupy_spectral, cupy_spmv, cupy_stencil
        )

        workload_map = {
            "mc": cupy_monte_carlo,
            "cg": cupy_conjugate_gradient,
            "dcs": cupy_dcs,
            "spectral": cupy_spectral,
            "spmv": cupy_spmv,
            "stencil": cupy_stencil,
        }
    else:
        from numba_methods import (
            numba_monte_carlo, numba_conjugate_gradient, numba_dcs,
            numba_spectral, numba_spmv, numba_stencil
        )

        workload_map = {
            "mc": numba_monte_carlo,
            "cg": numba_conjugate_gradient,
            "dcs": numba_dcs,
            "spectral": numba_spectral,
            "spmv": numba_spmv,
            "stencil": numba_stencil,
        }

    implementation = workload_map[args.workload]
    runtime, result = implementation(args, kernel_start_event)

    if args.validate:
        from numpy_methods import numpy_theoretical_monte_carlo
        expected = numpy_theoretical_monte_carlo(args.d)

        queue.put({
            "type": "validate",
            "cpu_result": expected,
            "gpu_result": result,
        })

    queue.put({
        "type": "workload",
        "runtime_ms": runtime
    })

def print_workload_(message):
    print("\r" + " " * 100 + "\r", end="")
    print("\nGPU Summary:")
    print(f"  - Runtime:    {message['runtime_ms']:2.4f} ms\n")

def print_gpu_(message):
    gpu = message["gpu"]
    mem_used = message["mem_used"]
    mem_total = message["mem_total"]

    mem_used_gb = mem_used / (1024 ** 3)
    mem_total_gb = mem_total / (1024 ** 3)
    mem_percent = mem_used / mem_total * 100

    print(
        f"\rGPU: {gpu:3.1f}% | "
        f"VRAM: {mem_used_gb:.2f}/{mem_total_gb:.2f} GB "
        f"({mem_percent:.1f}%)",
        end="",
        flush=True
    )

def print_validate_(message):
    cpu_result = message['cpu_result']
    gpu_result = message['gpu_result']
    difference = abs(cpu_result - gpu_result)
    
    print("\n\nResult Validity:")
    print(f" - CPU result: {cpu_result:2.3f}")
    print(f" - GPU result: {gpu_result:2.3f}")
    print(f" - Difference: {difference:2.3f}")


if __name__ == "__main__":
    args = parse_args()

    print(f"Framework:      {FRAMEWORKS[args.framework]}")
    print(f"Workload:       {NUM_METHOD[args.workload]}")
    print(f"Grid size:      [{args.grid_size}, {args.grid_size}, {args.grid_size}]")
    print()

    queue = mp.Queue()
    stop_event = mp.Event()
    ready_event = mp.Event()
    kernel_start_event = mp.Event()

    monitor_process = mp.Process(target=monitor, args=(queue, stop_event, ready_event, kernel_start_event))
    workload_process = mp.Process(target=workload, args=(queue, args, kernel_start_event))

    monitor_process.start()
    ready_event.wait()
    workload_process.start()

    gpu = 0
    mem_used = 0
    mem_total = 0

    while workload_process.is_alive():
        try:
            message = queue.get(timeout=0.1)
        except Empty:
            message = None

        while message is not None:
            if message["type"] == "gpu_stats":
                print(
                    f"GPU name:       {message['name']}\n"
                    f"Driver version: {message['driver']}"
                )
            
            elif message["type"] == "validate":
                print_validate_(message)

            elif message["type"] == "gpu":
                print_gpu_(message)

            elif message["type"] == "workload":
                print_workload_(message)

            try:
                message = queue.get_nowait()
            except Empty:
                message = None

    # Fetch last message
    while True:
        try:
            message = queue.get_nowait()
        except Empty:
            break

        if message["type"] == "workload":
            print_workload_(message)

    stop_event.set()

    workload_process.join()
    monitor_process.join(timeout=30)

    # Fetch summary
    while True:
        try:
            message = queue.get_nowait()
        except Empty:
            break

        if message["type"] == "gpu_summary":
            print()
            print(f"Avg GPU Utilization:    {message['gpu_avg']:.2f} %")
            print(f"Max GPU Utilization:    {message['gpu_max']:.2f} %")
            print(f"Max Memory Utilization: {message['mem_util_max']:.2f} %")
            print(f"Avg Memory Utilization: {message['mem_util_avg']:.2f} %")
            print(f"Max Power:              {message['power_max']:.2f} W")
            print(f"Avg Power:              {message['power_avg']:.2f} W")
            print(f"Samples:                {message['num_samples']}")

    print()
