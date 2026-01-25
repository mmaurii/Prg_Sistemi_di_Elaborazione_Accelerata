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

struct Index
{
    std::string name;
    std::string filename;
    float S0;
    float drift;
    float vol;
    // Questi valori li leggeremo dal buffer pinned alla fine
    float priceWorst; 
    float priceMed;
    float priceBest;
    float* dSimDevice = nullptr; // Memoria GPU
    std::vector<float> prices;
    cudaStream_t stream = nullptr;
};

// Configurazione
const std::string CSV_FILENAME_MSCI = "DATASET/msci_world_prezzi.csv";
const std::string CSV_FILENAME_SP500 = "DATASET/S&P500_prezzi.csv";
const std::string CSV_FILENAME_GDAXI = "DATASET/GDAXI_prezzi.csv";
const std::string CSV_FILENAME_N225 = "DATASET/N225_prezzi.csv";

const float T_YEARS = 1.0f;
const int SEED = 12345UL;
const int DAYS_OPEN_IN_YEAR = 252;
const float CAPITALE_INIZIALE = 10000.0f;

Index indexes[4] = {
    {"MSCI", CSV_FILENAME_MSCI},
    {"S&P500", CSV_FILENAME_SP500},
    {"GDAXI", CSV_FILENAME_GDAXI},
    {"N225", CSV_FILENAME_N225}
};

// ... (Funzioni readPrices e calculateParameters invariate, omesse per brevità) ...
std::vector<float> readPrices(const std::string& filename) {
    std::vector<float> prices;
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()) return prices;
    std::getline(file, line); 
    while (std::getline(file, line)) {
        size_t lastComma = line.find_last_of(',');
        if (lastComma != std::string::npos) {
            try { prices.push_back(std::stod(line.substr(lastComma + 1))); } catch (...) {}
        }
    }
    return prices;
}

void calculateParameters(const std::vector<float>& prices, float& S0, float& drift, float& vol) {
    if (prices.empty()) { S0 = 100.0f; drift = 0.05f; vol = 0.2f; return; }
    S0 = prices.back();
    std::vector<float> logReturns;
    for (size_t i = 1; i < prices.size(); ++i) logReturns.push_back(std::log(prices[i] / prices[i - 1]));
    float mean = 0.0; for (float r : logReturns) mean += r; mean /= logReturns.size();
    float sq = 0.0; for (float r : logReturns) sq += r * r;
    float stdev = std::sqrt(sq / logReturns.size() - mean * mean);
    drift = mean * DAYS_OPEN_IN_YEAR;
    vol   = stdev * std::sqrt(static_cast<float>(DAYS_OPEN_IN_YEAR));
}
// ... (Fine funzioni helper) ...

// Kernel cuda per simulazioni Monte Carlo
__global__ void monteCarloKernel(float* __restrict__ out, int n, float S0, float driftTerm, float volTerm, unsigned long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid * 4 >= n) return; // Ogni thread calcola 4 simulazioni

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    // Genero 4 numeri casuali in un colpo solo (istruzione vettoriale)
    float4 Z = curand_normal4(&state);
    
    float4 res;
    // Calcolo simulazione (__expf che è l'intrinseco veloce, approssimativa)
    res.x = S0 * __expf(driftTerm + volTerm * Z.x);
    res.y = S0 * __expf(driftTerm + volTerm * Z.y);
    res.z = S0 * __expf(driftTerm + volTerm * Z.z);
    res.w = S0 * __expf(driftTerm + volTerm * Z.w);

    // Scrittura vettorizzata in memoria globale (1 transazione per 4 float)
    reinterpret_cast<float4*>(out)[tid] = res;
}

int main(int argc, char* argv[]) {
    long nSimulations = 10000000; 
    if (argc > 1) nSimulations = std::stol(argv[1]);

    if (nSimulations % 4 != 0) nSimulations += (4 - (nSimulations % 4));

    dim3 blockDim(256);
    dim3 gridDim((nSimulations/4 + blockDim.x - 1) / blockDim.x);

    int idxWorst = (int)(nSimulations * 0.01f);
    int idxMed   = (int)(nSimulations * 0.50f);
    int idxBest  = (int)(nSimulations * 0.99f);

    // Preparazione parametri CPU
    for (auto& index : indexes) {
        index.prices = readPrices(index.filename);
        calculateParameters(index.prices, index.S0, index.drift, index.vol);
        
        // Pre-calcolo termini GBM
        float mu = index.drift;
        float sigma = index.vol;
        index.drift = (mu - 0.5f * sigma * sigma) * T_YEARS; 
        index.vol   = sigma * std::sqrt(T_YEARS);            
    }

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
     
    // Inizio registrazione evento GPU
    cudaEventRecord(start);


    // Allocazione buffer Pinned per i risultati (3 float per ogni indice)
    float* h_pinnedResults; 
    cudaMallocHost(&h_pinnedResults, 4 * 3 * sizeof(float)); 
    
    // FASE 1: Setup Sincrono (Malloc & Stream Create)
    for (auto& index : indexes) {
        cudaStreamCreate(&index.stream);
        cudaMalloc(&index.dSimDevice, nSimulations * sizeof(float));
    }
    
//    std::cout << "\nAvvio Simulazione GPU (Pinned Memory + Streams)..." << std::endl;
    cudaDeviceSynchronize(); 
    


    // FASE 2: Esecuzione Asincrona
    int i = 0;
    for (auto& index : indexes) {
        
        // A. Lancio Kernel
        monteCarloKernel<<<gridDim, blockDim, 0, index.stream>>>(
            index.dSimDevice, nSimulations, index.S0, index.drift, index.vol, SEED + i
        );
        
        // B. Sort Asincrono
        thrust::sort(thrust::cuda::par.on(index.stream), 
                     thrust::device_pointer_cast(index.dSimDevice), 
                     thrust::device_pointer_cast(index.dSimDevice + nSimulations));
        
        // C. Copia Asincrona su PINNED MEMORY 
        // Calcoliamo gli offset nel buffer pinned
        // Struttura buffer: [Indice0_Worst, Indice0_Med, Indice0_Best, Indice1_Worst...]
        int baseOffset = i * 3;
        
        cudaMemcpyAsync(&h_pinnedResults[baseOffset + 0], index.dSimDevice + idxWorst, sizeof(float), cudaMemcpyDeviceToHost, index.stream);
        cudaMemcpyAsync(&h_pinnedResults[baseOffset + 1], index.dSimDevice + idxMed,   sizeof(float), cudaMemcpyDeviceToHost, index.stream);
        cudaMemcpyAsync(&h_pinnedResults[baseOffset + 2], index.dSimDevice + idxBest,  sizeof(float), cudaMemcpyDeviceToHost, index.stream);
        
        i++;
    }
    
    // Attesa fine lavori
    cudaDeviceSynchronize();
    
    // FASE 3: Output e Pulizia
    int k = 0;
    for(auto& index : indexes) {
        float portfolioWorst = CAPITALE_INIZIALE * (index.priceWorst / index.S0);
        float portfolioMed   = CAPITALE_INIZIALE * (index.priceMed / index.S0);
        float portfolioBest  = CAPITALE_INIZIALE * (index.priceBest / index.S0);
        
        // Pulizia memoria device
        cudaFree(index.dSimDevice);
        cudaStreamDestroy(index.stream);
        
        /*         std::cout << "\n--- PROIEZIONE PATRIMONIO (Investimento: " << CAPITALE_INIZIALE << ") ---" << std::endl;
        std::cout << "Scenario migliore (1% percentile):   " << portfolioBest << " (+" << (portfolioBest / CAPITALE_INIZIALE - 1) * 100 << "%)" << std::endl;
        std::cout << "Scenario medio (50% percentile): " << portfolioMed << " (+" << (portfolioMed / CAPITALE_INIZIALE - 1) * 100 << "%)"<< std::endl;
        std::cout << "Scenario pessimo (99% percentile):  " << portfolioWorst << " (-" << (1 - portfolioWorst / CAPITALE_INIZIALE) * 100 << "%)" << std::endl;
        */        k++;
    }
    
    cudaFreeHost(h_pinnedResults);
    
    // Fine registrazione evento GPU
    cudaEventRecord(stop);
    // Aspettiamo che l'evento "stop" sia stato registrato realmente
    cudaEventSynchronize(stop);
    
    // Calcolo delta
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "GPU Kernel Time: " << milliseconds << " ms" << std::endl;

    return 0;
}