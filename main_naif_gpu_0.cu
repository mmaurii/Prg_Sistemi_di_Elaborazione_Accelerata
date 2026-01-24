#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <string>
#include <sstream>
#include <time.h>

// CUDA
#include <cuda.h>
#include <curand_kernel.h>

// Configurazione
const std::string CSV_FILENAME = "DATASET/msci_world_prezzi.csv";
const int N_SIMULATIONS = 10'000'000;
const double T_YEARS = 10.0;
const double CONFIDENCE_LEVEL = 0.99;
const int SEED = 12345UL;

// Caricamento dati da CSV
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

// Calcolo prestazioni
double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

// Calcolo parametri drift e volatilità
void calculateParameters(const std::vector<double>& prices, double& S0, double& drift, double& vol) {
    S0 = prices.back();

    std::vector<double> logReturns;
    for (size_t i = 1; i < prices.size(); ++i) {
        logReturns.push_back(std::log(prices[i] / prices[i - 1]));
    }

    // Calcolo drift giornaliero
    double mean = 0.0;
    for (double r : logReturns) mean += r;
    mean /= logReturns.size();

    // Calcolo deviazione standard
    double sq = 0.0;
    for (double r : logReturns) sq += r * r;

    double stdev = std::sqrt(sq / logReturns.size() - mean * mean);

    // Annualizzazione
    drift = mean * 252.0;
    vol   = stdev * std::sqrt(252.0);
}

// Kernel cuda per simulazioni Monte Carlo
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

int main(void) {

//    std::cout << "=== Monte Carlo VaR GPU (NAIVE) ===\n";

    // Caricamento dati
//    std::cout << "Lettura dati da " << CSV_FILENAME << "..." << std::endl;
    auto prices = readPrices(CSV_FILENAME);
 //   std::cout << "Letti " << prices.size() << " prezzi storici." << std::endl;

    // Calcolo parametri
    double S0, drift, volatilita;
    calculateParameters(prices, S0, drift, volatilita);

//    std::cout << "Prezzo Iniziale (S0): " << S0 << std::endl;
//    std::cout << "Drift Annualizzato: " << drift << " (" << drift*100 << "%)" << std::endl;
 //   std::cout << "Volatilita' Annualizzata: " << volatilita << " (" << volatilita*100 << "%)" << std::endl;

//    std::cout << "\nAvvio Simulazione (" << N_SIMULATIONS << " iterazioni)..." << std::endl;

    double driftTerm = (drift - 0.5 * volatilita * volatilita) * T_YEARS;
    double volTerm   = volatilita * std::sqrt(T_YEARS);

    // Allocazione variabile su GPU per simulazioni
    double* d_sim;
    cudaMalloc(&d_sim, N_SIMULATIONS * sizeof(double));
    
    // Definizione griglia e blocchi 1D e 1D
    dim3 blockDim(256, 1, 1);
    dim3 gridDim((N_SIMULATIONS + blockDim.x - 1) / blockDim.x, 1, 1);

    // Avvio timer
    auto t0 = std::chrono::high_resolution_clock::now();

    monteCarloKernel<<<gridDim, blockDim>>>(d_sim, N_SIMULATIONS, S0, driftTerm, volTerm, SEED);
    cudaDeviceSynchronize();
    
    // Termine timer
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli> elapsed = t1-t0;
    std::cout << "Simulazione GPU completata in: " << elapsed.count() << " ms." << std::endl;
    
    // Trasferimento prezzi simulati da GPU a CPU
    std::vector<double> simulatedPrices(N_SIMULATIONS);
    cudaMemcpy(simulatedPrices.data(), d_sim, N_SIMULATIONS * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_sim);

    // Calcolo VaR
 //   std::cout << "Calcolo del VaR..." << std::endl;
    t0 = std::chrono::high_resolution_clock::now();

    std::sort(simulatedPrices.begin(), simulatedPrices.end());

    t1 = std::chrono::high_resolution_clock::now();
    elapsed = t1-t0;
    std::cout << "Tempo sort: " << elapsed.count() << " ms." << std::endl;

    int cutoff = static_cast<int>(N_SIMULATIONS * (1.0 - CONFIDENCE_LEVEL));
    double priceAtRisk = simulatedPrices[cutoff];

    double varAbsolute = S0 - priceAtRisk;
    double varPercent = (varAbsolute / S0) * 100.0;

 /*    std::cout << "Risultato VaR " << (CONFIDENCE_LEVEL * 100) << "% (" << T_YEARS << " anni):" << std::endl;
    std::cout << "Prezzo peggiore atteso (" << ((1.0 - CONFIDENCE_LEVEL) * 100) << "% dei casi): " << priceAtRisk << std::endl;
    std::cout << "Perdita Massima Stimata: " << varAbsolute << " (" << varPercent << "%)" << std::endl;
 */

    return 0;
}
