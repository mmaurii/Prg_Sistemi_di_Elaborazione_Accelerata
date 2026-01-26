/*
    Questo codice è parte del progetto di SISTEMI DI ELABORAZIONE ACCELLERATA M, implementa una simulazione 
    montecarlo partendo da dati storici scaricati da yfinance. L'obiettivo è stimare il valore futuro di un asset
    o un portafoglio di asset, basandosi su modelli stocastici. In questo modo da possiamo valutare il rischio e il
    potenziale rendimento dell'investimento in un orizzonte temporale definito. 
*/

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

// CONFIGURAZIONE
const std::string CSV_FILENAME = "DATASET/msci_world_prezzi.csv";
const float T_YEARS = 1.0;
const int SEED = 12345UL;
const int DAYS_OPEN_IN_YEAR = 252;        // Giorni borsa aperta in un anno
const float CAPITALE_INIZIALE = 10000.0; // Investimento ipotetico iniziale

// FUNZIONI DI UTILITA'

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

// Kernel cuda per simulazioni Monte Carlo path dependent
__global__ void monteCarloKernel(float *dResults, float S0, float driftPart, float volPart, int nSimulations, int nDays) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nSimulations) {
        curandStatePhilox4_32_10_t state;
        curand_init(SEED, idx, 0, &state);        

        float logSum = 0.0; // Accumuliamo qui invece di moltiplicare il prezzo
        
        #pragma unroll
        for (int t = 0; t < nDays; ++t) {
            float Z = curand_normal(&state);
            
            // Aggiorna rendita logaritmica cumulativa
            logSum += driftPart + volPart * Z;
        }

        // Scriviamo solo il risultato finale in memoria globale
        dResults[idx] = S0 * expf(logSum);
    }
}

int main(int argc, char* argv[]) {
// Valore di default se l'utente non inserisce argomenti
    long nSimulations = 10000000; 
    if (argc > 1) {
        nSimulations = std::stol(argv[1]);
    }

    std::cout << "=== Monte Carlo GPU Path Dependent con Sort e proprietà dei Logaritmi  ===\n";

    // Caricamento dati
    std::cout << "Lettura dati da " << CSV_FILENAME << "..." << std::endl;
    auto prices = readPrices(CSV_FILENAME);
    std::cout << "Letti " << prices.size() << " prezzi storici." << std::endl;

    // Calcolo parametri
    float S0, drift, volatilita;
    calculateParameters(prices, S0, drift, volatilita);

    std::cout << "Prezzo Iniziale (S0): " << S0 << std::endl;
   std::cout << "Drift Annualizzato: " << drift << " (" << drift*100 << "%)" << std::endl;
    std::cout << "Volatilita' Annualizzata: " << volatilita << " (" << volatilita*100 << "%)" << std::endl;

    const float DT = 1.0 / static_cast<float>(DAYS_OPEN_IN_YEAR);
    const float driftStep = (drift - 0.5 * volatilita * volatilita) * DT;
    const float volStep = volatilita * std::sqrt(DT);

    std::cout << "\nAvvio simulazione (" << nSimulations << " cammini x " << T_YEARS << " anni)..." << std::endl;

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Inizio registrazione evento GPU
    cudaEventRecord(start);

    // Allocazione variabile su GPU per simulazioni
    float* dSim;
    cudaMalloc(&dSim, nSimulations * sizeof(float));
    
    // Definizione griglia e blocchi 1D e 1D
    dim3 blockDim(256, 1, 1);
    dim3 gridDim((nSimulations + blockDim.x - 1) / blockDim.x, 1, 1);

    monteCarloKernel<<<gridDim, blockDim>>>(dSim, S0, driftStep, volStep, nSimulations, DAYS_OPEN_IN_YEAR * T_YEARS);
    cudaDeviceSynchronize();
    
    // Wrapping del puntatore raw per thrust (sort)
    thrust::device_ptr<float> dPtr(dSim);
    thrust::sort(dPtr, dPtr + nSimulations);
    
    // Trasferimento prezzi simulati da GPU a CPU
    std::vector<float> simulatedPortfolioValues(nSimulations);
    cudaMemcpy(simulatedPortfolioValues.data(), dSim, nSimulations * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dSim);

    // Fine registrazione evento GPU
    cudaEventRecord(stop);
    // Aspettiamo che l'evento "stop" sia stato registrato realmente
    cudaEventSynchronize(stop);

    // Calcolo delta
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "GPU Kernel Time: " << milliseconds << " ms" << std::endl;

    // Analisi dei risultati
    std::cout << "Analisi dei risultati..." << std::endl;

    // Scenario Peggiore (1% percentile - Potential Downside)
    int idxWorst = (int)(nSimulations * 0.01f);
    float priceWorst = simulatedPortfolioValues[idxWorst];
    float portfolioWorst = CAPITALE_INIZIALE * (priceWorst / S0);

    // Scenario Mediano (50% percentile - Valore più probabile)
    int idxMed = (int)(nSimulations * 0.50f);
    float priceMed = simulatedPortfolioValues[idxMed];
    float portfolioMed = CAPITALE_INIZIALE * (priceMed / S0);

    // Scenario Migliore (99% percentile - Potential Upside)
    int idxBest = (int)(nSimulations * 0.99f);
    float priceBest = simulatedPortfolioValues[idxBest];
    float portfolioBest = CAPITALE_INIZIALE * (priceBest / S0);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n--- PROIEZIONE PATRIMONIO (Investimento: " << CAPITALE_INIZIALE << " ) ---" << std::endl;
    std::cout << "Scenario migliore (1% percentile):   " << portfolioBest << " (+" << (portfolioBest / CAPITALE_INIZIALE - 1) * 100 << "%)" << std::endl;
    std::cout << "Scenario medio (50% percentile): " << portfolioMed << " (+" << (portfolioMed / CAPITALE_INIZIALE - 1) * 100 << "%)"<< std::endl;
    std::cout << "Scenario pessimo (99% percentile):  " << portfolioWorst << " (-" << (1 - portfolioWorst / CAPITALE_INIZIALE) * 100 << "%)" << std::endl;

    return 0;
}
