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

// Configurazione
const std::string CSV_FILENAME = "DATASET/msci_world_prezzi.csv";
const float T_YEARS = 1.0;
const int SEED = 12345UL;
const int DAYS_OPEN_IN_YEAR = 252;        // Giorni borsa aperta in un anno
const float CAPITALE_INIZIALE = 10000.0; // Investimento ipotetico iniziale
const int N_CYCLES = 4;
const int SIMS_PER_THREAD = 4 * N_CYCLES; // Ogni thread calcola 4 simulazioni in 4 cicli
// Il numero di simulazioni deve essere multiplo di SIMS_PER_THREAD

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

// Kernel cuda per simulazioni Monte Carlo
__global__ void monteCarloKernel(float* __restrict__ out, int n, float S0, float driftTerm, float volTerm, unsigned long seed) {
    // Indice globale del thread e passo (stride) della griglia
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int baseIdx = tid * SIMS_PER_THREAD;

    if (baseIdx+SIMS_PER_THREAD > n) return;

    float4 Z;
    float4 res;
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);
    
    #pragma unroll
    for (int i = 0; i < N_CYCLES; i++) {
        int currentOffset = i * 4;

        // Genero 4 numeri casuali in un colpo solo (istruzione vettoriale)
        Z = curand_normal4(&state);
        
        // Calcolo simulazione (__expf che è l'intrinseco veloce, approssimativa)
        res.x = S0 * __expf(driftTerm + volTerm * Z.x);
        res.y = S0 * __expf(driftTerm + volTerm * Z.y);
        res.z = S0 * __expf(driftTerm + volTerm * Z.z);
        res.w = S0 * __expf(driftTerm + volTerm * Z.w);

        // Scrittura vettorizzata in memoria globale (1 transazione per 4 float)
        reinterpret_cast<float4*>(&out[baseIdx + currentOffset])[0] = res;
    }
}

int main(int argc, char* argv[]) {
// Valore di default se l'utente non inserisce argomenti
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

  //  std::cout << "\nAvvio Simulazione (" << nSimulations << " iterazioni)..." << std::endl;

    float driftTerm = (drift - 0.5 * volatilita * volatilita) * T_YEARS;
    float volTerm   = volatilita * std::sqrt(T_YEARS);

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Inizio registrazione evento GPU
    cudaEventRecord(start);

    // Allocazione variabile su GPU per salvare simulazioni su device
    float* dSimDevice;
    cudaMalloc(&dSimDevice, nSimulations * sizeof(float));
    
    // Allocazione pinned memory per salvare simulazioni su host
    float* dSimHost;
    cudaMallocHost(&dSimHost, nSimulations * sizeof(float));

    // Definizione griglia e blocchi 1D e 1D
    dim3 blockDim(256, 1, 1);
    dim3 gridDim((nSimulations + (blockDim.x * SIMS_PER_THREAD) - 1) / (blockDim.x * SIMS_PER_THREAD));

    monteCarloKernel<<<gridDim, blockDim>>>(dSimDevice, nSimulations, S0, driftTerm, volTerm, SEED);
    cudaDeviceSynchronize();

    // Wrapping del puntatore raw per thrust (sort)
    thrust::device_ptr<float> dPtr(dSimDevice);
    thrust::sort(dPtr, dPtr + nSimulations);

    cudaMemcpy(dSimHost, dSimDevice, nSimulations * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(dSimDevice);
    // Analisi dei risultati
//    std::cout << "Analisi dei risultati..." << std::endl;
    
    // Scenario Peggiore (1% percentile - Potential Downside)
    int idxWorst = (int)(nSimulations * 0.01f);
    float priceWorst = dSimHost[idxWorst];
    float portfolioWorst = CAPITALE_INIZIALE * (priceWorst / S0);
    
    // Scenario Mediano (50% percentile - Valore più probabile)
    int idxMed = (int)(nSimulations * 0.50f);
    float priceMed = dSimHost[idxMed];
    float portfolioMed = CAPITALE_INIZIALE * (priceMed / S0);
    
    // Scenario Migliore (99% percentile - Potential Upside)
    int idxBest = (int)(nSimulations * 0.99f);
    float priceBest = dSimHost[idxBest];
    float portfolioBest = CAPITALE_INIZIALE * (priceBest / S0);
    
    std::cout << std::fixed << std::setprecision(2);
//    std::cout << "\n--- PROIEZIONE PATRIMONIO (Investimento: " << CAPITALE_INIZIALE << " EUR) ---" << std::endl;
//    std::cout << "Scenario migliore (1% percentile):   " << portfolioBest << " EUR (+" << (portfolioBest / CAPITALE_INIZIALE - 1) * 100 << "%)" << std::endl;
 //   std::cout << "Scenario medio (50% percentile): " << portfolioMed << " EUR (+" << (portfolioMed / CAPITALE_INIZIALE - 1) * 100 << "%)"<< std::endl;
//    std::cout << "Scenario pessimo (99% percentile):  " << portfolioWorst << " EUR (-" << (1 - portfolioWorst / CAPITALE_INIZIALE) * 100 << "%)" << std::endl;
    cudaFreeHost(dSimHost);
    
    // Fine registrazione evento GPU
    cudaEventRecord(stop);
    // Aspettiamo che l'evento "stop" sia stato registrato realmente
    cudaEventSynchronize(stop);

    // Calcolo delta
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "GPU Kernel Time: " << milliseconds << " ms" << std::endl;

    return 0;
}
