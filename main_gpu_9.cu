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
#include <iomanip>

// CUDA
#include <cuda.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cuda_fp16.h>
#include <nvtx3/nvToolsExt.h> 
// Se non trovi nvtx3, usa #include <nvToolsExt.h>

// Configurazione
const std::string CSV_FILENAME = "DATASET/msci_world_prezzi.csv";
const float T_YEARS = 1.0;
const int SEED = 12345UL;
const int DAYS_OPEN_IN_YEAR = 252;        // Giorni borsa aperta in un anno
const float CAPITALE_INIZIALE = 10000.0; // Investimento ipotetico iniziale

// Caricamento dati da CSV
std::vector<float> readPrices(const std::string& filename) {
    std::vector<float> prices;
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
float cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((float)ts.tv_sec + (float)ts.tv_nsec * 1.e-9);
}

// Calcolo parametri drift e volatilità
void calculateParameters(const std::vector<float>& prices, float& S0, float& drift, float& vol) {
    S0 = prices.back();

    std::vector<float> logReturns;
    for (size_t i = 1; i < prices.size(); ++i) {
        logReturns.push_back(std::log(prices[i] / prices[i - 1]));
    }

    // Calcolo drift giornaliero
    float mean = 0.0;
    for (float r : logReturns) mean += r;
    mean /= logReturns.size();

    // Calcolo deviazione standard
    float sq = 0.0;
    for (float r : logReturns) sq += r * r;

    float stdev = std::sqrt(sq / logReturns.size() - mean * mean);

    // Annualizzazione
    drift = mean * DAYS_OPEN_IN_YEAR;
    vol   = stdev * std::sqrt(DAYS_OPEN_IN_YEAR);
}

// 1. KERNEL DI INIZIALIZZAZIONE (Si lancia UNA SOLA VOLTA)
__global__ void initRNG(curandStatePhilox4_32_10_t* states, unsigned long seed, int nThreads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nThreads) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

// Kernel cuda per simulazioni Monte Carlo
__global__ void monteCarloKernel(float* __restrict__ out, curandStatePhilox4_32_10_t* states, int nCycles,float S0, float driftTerm, float volTerm) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Ogni thread carica il SUO stato personale dalla memoria globale
    // Nota: copiamo lo stato in registro locale per velocità durante il loop
    curandStatePhilox4_32_10_t localState = states[tid]; 

    float4 res;
    float4 Z;

    for (size_t i = tid; i < nCycles; i += stride) {
        // Generazione veloce (lo stato è già pronto!)
        Z = curand_normal4(&localState);

        res.x = S0 * __expf(driftTerm + volTerm * Z.x);
        res.y = S0 * __expf(driftTerm + volTerm * Z.y);
        res.z = S0 * __expf(driftTerm + volTerm * Z.z);
        res.w = S0 * __expf(driftTerm + volTerm * Z.w);

        reinterpret_cast<float4*>(out)[i] = res;
    }

}

int main(int argc, char* argv[]) {
    // Valore di default se l'utente non inserisce argomenti
    // Il numero di simulazioni deve essere multiplo di 4
    long nSimulations = 10000000; 
    if (argc > 1) {
        // Converte l'argomento della riga di comando in numero
        nSimulations = std::stol(argv[1]);
    }

//    std::cout << "=== Monte Carlo VaR GPU (NAIVE) ===\n";

    // Caricamento dati
//    std::cout << "Lettura dati da " << CSV_FILENAME << "..." << std::endl;
    auto prices = readPrices(CSV_FILENAME);
//    std::cout << "Letti " << prices.size() << " prezzi storici." << std::endl;

    // Calcolo parametri
    float S0, drift, volatilita;
    calculateParameters(prices, S0, drift, volatilita);

//    std::cout << "Prezzo Iniziale (S0): " << S0 << std::endl;
//    std::cout << "Drift Annualizzato: " << drift << " (" << drift*100 << "%)" << std::endl;
//    std::cout << "Volatilita' Annualizzata: " << volatilita << " (" << volatilita*100 << "%)" << std::endl;

//    std::cout << "\nAvvio Simulazione (" << nSimulations << " iterazioni)..." << std::endl;

    float driftTerm = (drift - 0.5 * volatilita * volatilita) * T_YEARS;
    float volTerm   = volatilita * std::sqrt(T_YEARS);

    // Definizione griglia e blocchi 1D e 1D
    dim3 blockDim(256, 1, 1);
    // L' implementazione andrebbe adattata nel caso il numero di simulazioni non sia multiplo di 4
    dim3 gridDim((nSimulations + (blockDim.x*4) - 1) / (blockDim.x*4), 1, 1);

    int totalThreads = gridDim.x * blockDim.x;
    int nCycles=nSimulations/4;
    
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Inizio registrazione evento GPU
    cudaEventRecord(start);

    // Allocazione variabile su GPU per salvare simulazioni su device
    float* dSimDevice;
    cudaMalloc(&dSimDevice, nSimulations * sizeof(float));
    // Allocazione stati RNG
    curandStatePhilox4_32_10_t* dStatesDevice;
    cudaMalloc(&dStatesDevice, totalThreads * sizeof(curandStatePhilox4_32_10_t));

//    std::cout << "Inizializzazione RNG..." << std::endl;
    initRNG<<<gridDim, blockDim>>>(dStatesDevice, SEED, totalThreads);
    cudaDeviceSynchronize(); // Aspettiamo che finisca

    monteCarloKernel<<<gridDim, blockDim>>>(dSimDevice, dStatesDevice, nCycles, S0, driftTerm, volTerm);
    cudaDeviceSynchronize();

    // Wrapping del puntatore raw per thrust (sort)
    thrust::device_ptr<float> dPtr(dSimDevice);
    thrust::sort(dPtr, dPtr + nSimulations);

    
    // Analisi dei risultati
    float priceWorst, priceMed, priceBest;
    
    int idxWorst = (int)(nSimulations * 0.01f);
    int idxMed   = (int)(nSimulations * 0.50f);
    int idxBest  = (int)(nSimulations * 0.99f);

    // Copia da (dSimDevice + offset) a variabile CPU
    cudaMemcpy(&priceWorst, dSimDevice + idxWorst, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&priceMed,   dSimDevice + idxMed,   sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&priceBest,  dSimDevice + idxBest,  sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(dSimDevice);
    cudaFree(dStatesDevice);

    // Fine registrazione evento GPU
    cudaEventRecord(stop);
    // Aspettiamo che l'evento "stop" sia stato registrato realmente
    cudaEventSynchronize(stop);

    // Calcolo delta
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "GPU Kernel Time: " << milliseconds << " ms" << std::endl;

    
//    std::cout << "Analisi dei risultati..." << std::endl;
    
    // Scenario Peggiore (1% percentile - Potential Downside)
    float portfolioWorst = CAPITALE_INIZIALE * (priceWorst / S0);
    
    // Scenario Mediano (50% percentile - Valore più probabile)
    float portfolioMed = CAPITALE_INIZIALE * (priceMed / S0);
    
    // Scenario Migliore (99% percentile - Potential Upside)
    float portfolioBest = CAPITALE_INIZIALE * (priceBest / S0);
    
    std::cout << std::fixed << std::setprecision(2);
  /*   std::cout << "\n--- PROIEZIONE PATRIMONIO (Investimento: " << CAPITALE_INIZIALE << " EUR) ---" << std::endl;
    std::cout << "Scenario migliore (1% percentile):   " << portfolioBest << " EUR (+" << (portfolioBest / CAPITALE_INIZIALE - 1) * 100 << "%)" << std::endl;
    std::cout << "Scenario medio (50% percentile): " << portfolioMed << " EUR (+" << (portfolioMed / CAPITALE_INIZIALE - 1) * 100 << "%)"<< std::endl;
    std::cout << "Scenario pessimo (99% percentile):  " << portfolioWorst << " EUR (-" << (1 - portfolioWorst / CAPITALE_INIZIALE) * 100 << "%)" << std::endl;
 */
    return 0;
}
