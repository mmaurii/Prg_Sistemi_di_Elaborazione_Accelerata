#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <chrono>

// CUDA
#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>

// ---------------- CONFIG ----------------
const std::string CSV_FILENAME = "DATASET/msci_world_prezzi.csv";
const int N_SIMULATIONS = 10'000'000;
const double T_YEARS = 10.0;
const double CONFIDENCE_LEVEL = 0.99;

// ---------------- CSV ----------------
std::vector<double> readPrices(const std::string& filename) {
    std::vector<double> prices;
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line); // header
    while (std::getline(file, line)) {
        size_t lastComma = line.find_last_of(',');
        if (lastComma != std::string::npos) {
            try { prices.push_back(std::stod(line.substr(lastComma + 1))); }
            catch (...) {}
        }
    }
    return prices;
}

// ---------------- CUDA KERNELS ----------------

// Log-returns
__global__ void logReturnsKernel(
    const double* prices,
    double* logReturns,
    int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > 0 && idx < n)
        logReturns[idx - 1] = log(prices[idx] / prices[idx - 1]);
}

// Riduzione sum + sumsq
__global__ void reduceSumSqKernel(
    const double* data,
    double* blockSum,
    double* blockSq,
    int n) {

    extern __shared__ double s[];
    double* s_sum = s;
    double* s_sq  = s + blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double sum = 0.0;
    double sq  = 0.0;

    while (idx < n) {
        double v = data[idx];
        sum += v;
        sq  += v * v;
        idx += blockDim.x * gridDim.x;
    }

    s_sum[tid] = sum;
    s_sq[tid]  = sq;
    __syncthreads();

    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) {
            s_sum[tid] += s_sum[tid + s2];
            s_sq[tid]  += s_sq[tid + s2];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockSum[blockIdx.x] = s_sum[0];
        blockSq[blockIdx.x]  = s_sq[0];
    }
}

// Monte Carlo GBM
__global__ void monteCarloKernel(
    double* out,
    int n,
    double S0,
    double driftTerm,
    double volTerm,
    unsigned long seed) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    double Z = curand_normal_double(&state);
    out[idx] = S0 * exp(driftTerm + volTerm * Z);
}

// ---------------- MAIN ----------------
int main() {

    // ---------------- Load prices ----------------
    auto h_prices = readPrices(CSV_FILENAME);
    int Np = h_prices.size();

    double S0 = h_prices.back();

    double* d_prices;
    cudaMalloc(&d_prices, Np * sizeof(double));
    cudaMemcpy(d_prices, h_prices.data(),
               Np * sizeof(double),
               cudaMemcpyHostToDevice);

    // ---------------- Log-returns ----------------
    int threads = 256;
    int blocks  = (Np + threads - 1) / threads;

    double* d_logReturns;
    cudaMalloc(&d_logReturns, (Np - 1) * sizeof(double));

    logReturnsKernel<<<blocks, threads>>>(d_prices, d_logReturns, Np);
    cudaDeviceSynchronize();

    // ---------------- Reduce mean & variance ----------------
    int redBlocks = 1024;
    double* d_sum;
    double* d_sq;

    cudaMalloc(&d_sum, redBlocks * sizeof(double));
    cudaMalloc(&d_sq,  redBlocks * sizeof(double));

    reduceSumSqKernel<<<redBlocks, threads, 2 * threads * sizeof(double)>>>(
        d_logReturns, d_sum, d_sq, Np - 1
    );
    cudaDeviceSynchronize();

    std::vector<double> h_sum(redBlocks), h_sq(redBlocks);
    cudaMemcpy(h_sum.data(), d_sum, redBlocks * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sq.data(),  d_sq,  redBlocks * sizeof(double), cudaMemcpyDeviceToHost);

    double totalSum = 0.0, totalSq = 0.0;
    for (int i = 0; i < redBlocks; ++i) {
        totalSum += h_sum[i];
        totalSq  += h_sq[i];
    }

    int N = Np - 1;
    double mean = totalSum / N;
    double var  = totalSq / N - mean * mean;

    double drift = mean * 252.0;
    double vol   = sqrt(var) * sqrt(252.0);

    double driftTerm = (drift - 0.5 * vol * vol) * T_YEARS;
    double volTerm   = vol * sqrt(T_YEARS);

    std::cout << "S0: " << S0 << "\n";
    std::cout << "Drift ann.: " << drift * 100 << "%\n";
    std::cout << "Vol ann.: " << vol * 100 << "%\n";

    // ---------------- Monte Carlo ----------------
    double* d_sim;
    cudaMalloc(&d_sim, N_SIMULATIONS * sizeof(double));

    blocks = (N_SIMULATIONS + threads - 1) / threads;

    auto t0 = std::chrono::high_resolution_clock::now();

    monteCarloKernel<<<blocks, threads>>>(
        d_sim, N_SIMULATIONS,
        S0, driftTerm, volTerm,
        12345UL
    );
    cudaDeviceSynchronize();

    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Monte Carlo GPU: "
              << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";

    // ---------------- VaR + Mean ----------------
    thrust::device_ptr<double> d_ptr(d_sim);
    thrust::sort(d_ptr, d_ptr + N_SIMULATIONS);

    int cutoff = static_cast<int>(N_SIMULATIONS * (1.0 - CONFIDENCE_LEVEL));
    double priceAtRisk = d_ptr[cutoff];

    double meanST = thrust::reduce(
        d_ptr, d_ptr + N_SIMULATIONS, 0.0
    ) / N_SIMULATIONS;

    double varAbs = S0 - priceAtRisk;
    double varPct = (varAbs / S0) * 100.0;

    std::cout << "\nVaR " << CONFIDENCE_LEVEL * 100 << "% (" << T_YEARS << " anni)\n";
    std::cout << "Prezzo peggiore: " << priceAtRisk << "\n";
    std::cout << "Perdita stimata: " << varAbs << " (" << varPct << "%)\n";
    std::cout << "Prezzo medio simulato: " << meanST << "\n";

    // ---------------- Cleanup ----------------
    cudaFree(d_prices);
    cudaFree(d_logReturns);
    cudaFree(d_sum);
    cudaFree(d_sq);
    cudaFree(d_sim);

    return 0;
}
