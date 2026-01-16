#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <chrono>

// CUDA
#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

// ---------------- CONFIG ----------------
const std::string CSV_FILENAME = "DATASET/msci_world_prezzi.csv";
const int N_SIMULATIONS = 10'000'000;
const double T_YEARS = 10.0;
const double CONFIDENCE_LEVEL = 0.99;

// ---------------- UTILS ----------------
std::vector<double> readPrices(const std::string& filename) {
    std::vector<double> prices;
    std::ifstream file(filename);
    std::string line;

    std::getline(file, line); // header
    while (std::getline(file, line)) {
        size_t lastComma = line.find_last_of(',');
        if (lastComma != std::string::npos) {
            try {
                prices.push_back(std::stod(line.substr(lastComma + 1)));
            } catch (...) {}
        }
    }
    return prices;
}

void calculateParameters(const std::vector<double>& prices, double& S0, double& drift, double& vol) {
    S0 = prices.back();

    std::vector<double> logReturns;
    for (size_t i = 1; i < prices.size(); ++i)
        logReturns.push_back(std::log(prices[i] / prices[i - 1]));

    double mean = 0.0;
    for (double r : logReturns) mean += r;
    mean /= logReturns.size();

    double sqSum = 0.0;
    for (double r : logReturns) sqSum += r * r;
    double stdev = std::sqrt(sqSum / logReturns.size() - mean * mean);

    drift = mean * 252.0;
    vol   = stdev * std::sqrt(252.0);
}

// ---------------- CUDA KERNELS ----------------

// Kernel 1: Monte Carlo GBM
__global__ void monteCarloKernel(double* d_out, int n, double S0, double driftTerm, double volTerm, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    double Z = curand_normal_double(&state);
    d_out[idx] = S0 * exp(driftTerm + volTerm * Z);
}

// Kernel 2: estrazione VaR (banale, 1 thread)
__global__ void extractVaRKernel(const double* d_sorted, int index, double* d_result) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_result = d_sorted[index];
    }
}

// ---------------- MAIN ----------------
int main() {
    std::cout << "=== Monte Carlo VaR CUDA ===\n";

    // Load data
    auto prices = readPrices(CSV_FILENAME);
    double S0, drift, vol;
    calculateParameters(prices, S0, drift, vol);

    double driftTerm = (drift - 0.5 * vol * vol) * T_YEARS;
    double volTerm   = vol * std::sqrt(T_YEARS);

    std::cout << "S0: " << S0 << "\n";
    std::cout << "Drift: " << drift * 100 << "%\n";
    std::cout << "Vol: " << vol * 100 << "%\n";

    // Device memory
    double* d_prices;
    cudaMalloc(&d_prices, N_SIMULATIONS * sizeof(double));

    // Launch Monte Carlo kernel
    int threads = 256;
    int blocks  = (N_SIMULATIONS + threads - 1) / threads;

    auto t0 = std::chrono::high_resolution_clock::now();

    monteCarloKernel<<<blocks, threads>>>(
        d_prices,
        N_SIMULATIONS,
        S0,
        driftTerm,
        volTerm,
        12345UL
    );
    cudaDeviceSynchronize();

    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> simTime = t1 - t0;
    std::cout << "Simulazione GPU: " << simTime.count() << " s\n";

    // Sort on GPU (VaR prep)
    thrust::device_ptr<double> d_ptr(d_prices);
    thrust::sort(d_ptr, d_ptr + N_SIMULATIONS);

    int cutoff = static_cast<int>(N_SIMULATIONS * (1.0 - CONFIDENCE_LEVEL));

    double* d_var;
    cudaMalloc(&d_var, sizeof(double));

    extractVaRKernel<<<1, 1>>>(d_prices, cutoff, d_var);
    cudaDeviceSynchronize();

    double priceAtRisk;
    cudaMemcpy(&priceAtRisk, d_var, sizeof(double), cudaMemcpyDeviceToHost);

    double varAbs = S0 - priceAtRisk;
    double varPct = (varAbs / S0) * 100.0;

    std::cout << "\nVaR " << CONFIDENCE_LEVEL * 100 << "% (" << T_YEARS << " anni)\n";
    std::cout << "Prezzo peggiore: " << priceAtRisk << "\n";
    std::cout << "Perdita stimata: " << varAbs << " (" << varPct << "%)\n";

    cudaFree(d_prices);
    cudaFree(d_var);
    return 0;
}
