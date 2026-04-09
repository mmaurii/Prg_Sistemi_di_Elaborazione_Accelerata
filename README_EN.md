# Accelerated Computing Monte Carlo Project

[Italian version](README.md)

University project for Monte Carlo simulation of financial price dynamics with multiple implementations:

- naive CPU (float and double)
- SIMD CPU
- naive GPU (CUDA)
- optimized GPU (CUDA)
- path-dependent Monte Carlo variants
- custom GPU random-number generators (xorshift, xoroshiro, splitmix64, pcg32)

The goal is to estimate future portfolio values and compare performance and design trade-offs across implementations.

## Project Structure

### Core implementations
- `main_naif_cpu_0.cpp`: naive CPU Monte Carlo (double)
- `main_naif_cpu_1.cpp`: naive CPU Monte Carlo (float)
- `main_naif_cpu_path_dependent_1.cpp`: path-dependent CPU Monte Carlo
- `main_naif_cpu_multiple_dataset.cpp`: CPU Monte Carlo over multiple market datasets
- `main_simd.cpp`: SIMD CPU Monte Carlo with SSE intrinsics

- `main_naif_gpu_0.cu`: naive GPU Monte Carlo (double)
- `main_naif_gpu_1.cu`: naive GPU Monte Carlo (float)
- `main_naif_gpu_path_dependent_1.cu`: naive GPU path-dependent version

- `main_gpu_3.cu` to `main_gpu_10.cu`: progressively optimized CUDA implementations
- `main_gpu_path_dependent_2.cu`, `main_gpu_path_dependent_3.cu`: optimized path-dependent CUDA versions
- `main_gpu_multiple_dataset.cu`: multi-dataset CUDA simulation (MSCI, S&P500, GDAXI, N225)

### RNG-focused CUDA variants
- `main_gpu_xorshift_64_0.cu`
- `main_gpu_xorshift_64_1.cu`
- `main_gpu_xoroshiro_128_0.cu`
- `main_gpu_xoroshiro_128_1.cu`
- `main_gpu_splitmix64.cu`
- `main_gpu_pcg32.cu`

### Data and helper scripts
- `DATASET/`: historical price CSV files
- `DATASET/downloadDataSet.py`: yfinance download script
- `collectData.py`: automated benchmark/profiling pipeline (nsys + ncu)
- `collectData1.py`: native runtime benchmark pipeline

### Learning resources
- `risorse/cheatsheet_CUDA.cu`
- `risorse/simdCheatsheet.cpp`

## Model and Output

Most implementations follow a Geometric Brownian Motion setup:

$$
S_T = S_0 \cdot e^{\left(\mu - \frac{1}{2}\sigma^2\right)T + \sigma\sqrt{T}Z}
$$

with parameters estimated from historical log-returns. Outputs usually include:

- simulation time
- sorted simulation distribution
- 1st, 50th, and 99th percentiles (worst / median / best)
- equivalent projected portfolio value from an initial capital

Path-dependent versions simulate day-by-day trajectories over trading days.

## Requirements

### Hardware
- NVIDIA GPU (for CUDA files)

### Software
- CUDA Toolkit (`nvcc`, `curand`, Thrust)
- C++ compiler with SSE support (for SIMD), such as `g++`
- Python 3 for automation scripts
- Optional profiling tools:
	- Nsight Systems (`nsys`)
	- Nsight Compute (`ncu`)

### Python packages (for dataset download)
- `yfinance`
- `pandas`

Install with:

```bash
pip install yfinance pandas
```

## Build

### Build a CUDA file

```bash
nvcc main_gpu_10.cu -O3 -o main_gpu_10
```

### Build a CPU file

```bash
g++ main_naif_cpu_1.cpp -O3 -o main_naif_cpu_1
```

### Build SIMD file

```bash
g++ main_simd.cpp -O3 -msse4.2 -o main_simd
```

Notes:
- Use `-Xcompiler -fopenmp` only if a specific file requires OpenMP in your environment.
- On Windows, executable names may become `.exe`.

## Run

Most binaries accept an optional argument for number of simulations:

```bash
./main_gpu_10 10000000
./main_naif_cpu_1 10000000
./main_simd 10000000
```

If no argument is provided, each program uses its own default (commonly `10,000,000`, and some CPU variants use larger defaults).

## Profiling and Benchmark Automation

### Nsight-based benchmark script
`collectData.py` automates runs with:
- `nsys profile`
- `ncu --set detailed`
- timeout handling
- per-run logs in `benchmark_results/`

Run:

```bash
python collectData.py
```

### Native runtime benchmark script
`collectData1.py` executes binaries directly (without Nsight tools) and stores logs in `benchmark_results_native/`.

Run:

```bash
python collectData1.py
```

Important: these scripts include executable names that may not exist in the current repository (for example `main_gpu_11` or `main_naif_cpu_2`). Remove or update missing entries before large benchmark batches.

## Dataset

The repository includes historical CSV series in `DATASET/`, including:

- MSCI World
- S&P 500
- GDAXI
- N225

To download/update data, edit ticker/file options in `DATASET/downloadDataSet.py` and run:

```bash
python DATASET/downloadDataSet.py
```

## Suggested Workflow

1. Build one CPU baseline and one GPU variant.
2. Validate numerical consistency (distribution quantiles and portfolio outputs).
3. Scale simulation count (`10^5` to `10^8+`) and compare runtime.
4. Profile kernels with Nsight tools.
5. Compare RNG variants and path-dependent versions.

## Repository Purpose

This repository is designed as an experimentation lab for accelerated Monte Carlo methods, where algorithmic choices, data layout, RNG strategy, and hardware mapping can be compared directly under the same financial workload.
