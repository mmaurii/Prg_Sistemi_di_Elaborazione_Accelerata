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

// CONFIGURAZIONE
const std::string CSV_FILENAME = "DATASET/msci_world_prezzi.csv";
const double T_YEARS = 1.0;
const int SEED = 12345UL;
const int DAYS_OPEN_IN_YEAR = 252;        // Giorni borsa aperta in un anno
const double CAPITALE_INIZIALE = 10000.0; // Investimento ipotetico iniziale

// FUNZIONI DI UTILITA'

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
    drift = mean * DAYS_OPEN_IN_YEAR;
    vol   = stdev * std::sqrt(DAYS_OPEN_IN_YEAR);
}

// Kernel cuda per simulazioni Monte Carlo
__global__ void monteCarloKernel(double* out,int n,double S0,double driftTerm,double volTerm,unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    double Z = curand_normal_double(&state);
    out[idx] = S0 * exp(driftTerm + volTerm * Z);
}

int main(int argc, char* argv[]) {
    // Valore di default se l'utente non inserisce argomenti
    long nSimulations = 10000000; 
    if (argc > 1) {
        nSimulations = std::stol(argv[1]);
    }
    std::cout << "=== Monte Carlo NAIVE GPU Double ===\n";

    // Caricamento dati
    std::cout << "Lettura dati da " << CSV_FILENAME << "..." << std::endl;
    auto prices = readPrices(CSV_FILENAME);
    std::cout << "Letti " << prices.size() << " prezzi storici." << std::endl;

    // Calcolo parametri
    double S0, drift, volatilita;
    calculateParameters(prices, S0, drift, volatilita);

    std::cout << "Prezzo Iniziale (S0): " << S0 << std::endl;
    std::cout << "Drift Annualizzato: " << drift << " (" << drift*100 << "%)" << std::endl;
    std::cout << "Volatilita' Annualizzata: " << volatilita << " (" << volatilita*100 << "%)" << std::endl;

    std::cout << "\nAvvio Simulazione (" << nSimulations << " iterazioni)..." << std::endl;

    double driftTerm = (drift - 0.5 * volatilita * volatilita) * T_YEARS;
    double volTerm   = volatilita * std::sqrt(T_YEARS);

    // Allocazione variabile su GPU per simulazioni
    double* dSim;
    cudaMalloc(&dSim, nSimulations * sizeof(double));
    
    // Definizione griglia e blocchi 1D e 1D
    dim3 blockDim(256, 1, 1);
    dim3 gridDim((nSimulations + blockDim.x - 1) / blockDim.x, 1, 1);

    // Avvio timer
    auto t0 = std::chrono::high_resolution_clock::now();

    monteCarloKernel<<<gridDim, blockDim>>>(dSim, nSimulations, S0, driftTerm, volTerm, SEED);
    cudaDeviceSynchronize();
    
    // Termine timer
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::milli> elapsed = t1-t0;
    std::cout << "Simulazione GPU completata in: " << elapsed.count() << " ms." << std::endl;
    
    // Trasferimento prezzi simulati da GPU a CPU
    std::vector<double> simulatedPrices(nSimulations);
    cudaMemcpy(simulatedPrices.data(), dSim, nSimulations * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(dSim);

    std::cout << "Analisi dei Risultati..." << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    
    // Ordinamento prezzi simulati
    std::sort(simulatedPrices.begin(), simulatedPrices.end());

    t1 = std::chrono::high_resolution_clock::now();
    elapsed = t1-t0;
    std::cout << "Tempo sort: " << elapsed.count() << " ms." << std::endl;

    // Scenario Peggiore (1% percentile - Potential Downside)
    int idxWorst = (int)(nSimulations * 0.01f);
    double priceWorst = simulatedPrices[idxWorst];
    double portfolioWorst = CAPITALE_INIZIALE * (priceWorst / S0);

    // Scenario Mediano (50% percentile - Valore più probabile)
    int idxMed = (int)(nSimulations * 0.50f);
    double priceMed = simulatedPrices[idxMed];
    double portfolioMed = CAPITALE_INIZIALE * (priceMed / S0);

    // Scenario Migliore (99% percentile - Potential Upside)
    int idxBest = (int)(nSimulations * 0.99f);
    double priceBest = simulatedPrices[idxBest];
    double portfolioBest = CAPITALE_INIZIALE * (priceBest / S0);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n--- PROIEZIONE PATRIMONIO (Investimento: " << CAPITALE_INIZIALE << " ) ---" << std::endl;
    std::cout << "Scenario migliore (1% percentile):   " << portfolioBest << " (+" << (portfolioBest / CAPITALE_INIZIALE - 1) * 100 << "%)" << std::endl;
    std::cout << "Scenario medio (50% percentile): " << portfolioMed << " (+" << (portfolioMed / CAPITALE_INIZIALE - 1) * 100 << "%)"<< std::endl;
    std::cout << "Scenario pessimo (99% percentile):  " << portfolioWorst << " (-" << (1 - portfolioWorst / CAPITALE_INIZIALE) * 100 << "%)" << std::endl;

    return 0;
}
