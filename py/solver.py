from py.cupy_methods import *
from py.numba_methods import *
from argparse import ArgumentParser
from queue import Empty
import multiprocessing as mp
import pynvml
import time

MONITOR_INTERVAL = 0.2

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

WORKLOADS = {
    "cupy": {
        "mc": cupy_monte_carlo,
        "cg": cupy_conjugate_gradient,
        "dcs": cupy_dcs,
        "spectral": cupy_spectral,
        "spmv": cupy_spmv,
        "stencil": cupy_stencil,
    },
    "numba": {
        "mc": numba_monte_carlo,
        "cg": numba_conjugate_gradient,
        "dcs": numba_dcs,
        "spectral": numba_spectral,
        "spmv": numba_spmv,
        "stencil": numba_stencil,
    },
}

def parse_args():
    parser = ArgumentParser(
        description="Run the 3D heat equation benchmark."
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
            "Spectral, Sparse Matrix Vector, Stencil."
    )

    return parser.parse_args()

def monitor(queue, stop_event, ready_event):
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

def workload(queue, args):
    implementation = WORKLOADS[args.framework][args.workload]

    t0 = time.perf_counter()
    status = implementation(args)
    elapsed = time.perf_counter() - t0

    queue.put({
        "type": "workload",
        "status": status,
        "elapsed": elapsed
    })

if __name__ == "__main__":
    args = parse_args()

    print(f"Framework:      {FRAMEWORKS[args.framework]}")
    print(f"Workload:       {NUM_METHOD[args.workload]}")
    print(f"Grid size:      [{args.grid_size}, {args.grid_size}, {args.grid_size}]")
    print()

    queue = mp.Queue()
    stop_event = mp.Event()
    ready_event = mp.Event()

    monitor_process = mp.Process(target=monitor, args=(queue, stop_event, ready_event))
    monitor_process.start()

    ready_event.wait(timeout=5)

    workload_process = mp.Process(target=workload, args=(queue, args, ))
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
                    "GPU name:       " + message["name"] + "\n" +
                    "Driver version: " + message["driver"]
                )

            if message["type"] == "gpu":
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

            elif message["type"] == "workload":
                print("\r" + " " * 100 + "\r", end="")
                print(f"Status:  {message['status']}")
                print(f"Runtime: {message['elapsed']:.3f} s")

            try:
                message = queue.get_nowait()
            except Empty:
                message = None

    while True:
        try:
            message = queue.get_nowait()
        except Empty:
            break

        if message["type"] == "workload":
            print("\r" + " " * 100 + "\r", end="")
            print(f"Status:  {message['status']}")
            print(f"Runtime: {message['elapsed']:.3f} s")

    stop_event.set()

    workload_process.join()
    monitor_process.join()

    while True:
        try:
            message = queue.get_nowait()
        except Empty:
            break

        if message["type"] == "gpu_summary":
            print(f"Max GPU: {message['gpu_max']:.2f}%")
            print(f"Avg GPU: {message['gpu_avg']:.2f}%")

    print()
