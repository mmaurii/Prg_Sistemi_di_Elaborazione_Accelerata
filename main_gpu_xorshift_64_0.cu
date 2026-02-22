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
const std::string CSV_FILENAME = "DATASET/S&P500_prezzi.csv";
const float T_YEARS = 1.0;
const int SEED = 12345UL;
const int DAYS_OPEN_IN_YEAR = 252;        // Giorni borsa aperta in un anno
const float CAPITALE_INIZIALE = 10000.0; // Investimento ipotetico iniziale

/* ==== STRUTTURA E FUNZIONI PER RNG XORSHIFT64* ==== */

// Struttura di stato
struct MyXS64State {
    uint64_t s;
    float spare;
    bool hasSpare;
};

// Funzione ausiliaria di inizializzazione robusta
__device__ uint64_t myxs64_splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Init random generator
__device__ void myxs64_init_rng(MyXS64State& st, uint64_t seed, int tid) {
    uint64_t x = seed ^ (uint64_t)tid;
    st.s = myxs64_splitmix64(x);
    st.hasSpare = false;
}

// Step RNG
__device__ __forceinline__
uint64_t myxs64_step(uint64_t& x) {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return x * 0x2545F4914F6CDD1DULL;
}

// Da variabile uniforme a float (Intervallo 0..1)
__device__ __forceinline__
float myxs64_u01(uint64_t x) {
    return (x >> 40) * (1.0f / (1ULL << 24));
}

// Conversione da distribuzione normale a gaussiana
// Metodo di Marsaglia
__device__ float myxs64_normal(MyXS64State& st) {
    if (st.hasSpare) {
        st.hasSpare = false;
        return st.spare;
    }

    float x, y, s;
    do {
        x = 2.0f * myxs64_u01(myxs64_step(st.s)) - 1.0f;
        y = 2.0f * myxs64_u01(myxs64_step(st.s)) - 1.0f;
        s = x*x + y*y;
    } while (s >= 1.0f || s == 0.0f);

    float m = sqrtf(-2.0f * logf(s) / s);
    st.spare = y * m;
    st.hasSpare = true;

    return x * m;
}


// Funzione ausiliaria per CUDA
__device__ float4 myxs64_normal4(MyXS64State& st) {
    return make_float4(
        myxs64_normal(st),
        myxs64_normal(st),
        myxs64_normal(st),
        myxs64_normal(st)
    );
}

/* ================================================== */

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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid * 4 >= n) return; // Ogni thread calcola 4 simulazioni

    MyXS64State state;
    myxs64_init_rng(state, seed, tid);

    // Genero 4 numeri casuali in un colpo solo (istruzione vettoriale)
    float4 Z = myxs64_normal4(state);
    
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

    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Inizio registrazione evento GPU
    cudaEventRecord(start);

    // Allocazione variabile su GPU per salvare simulazioni su device
    float* dSimDevice;
    cudaMalloc(&dSimDevice, nSimulations * sizeof(float));
    
    // Definizione griglia e blocchi 1D e 1D
    dim3 blockDim(256, 1, 1);
    // L' implementazione andrebbe adattata nel caso il numero di simulazioni non sia multiplo di 4
    dim3 gridDim((nSimulations + (blockDim.x*4) - 1) / (blockDim.x*4), 1, 1);

    monteCarloKernel<<<gridDim, blockDim>>>(dSimDevice, nSimulations, S0, driftTerm, volTerm, SEED);
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
/*     std::cout << "\n--- PROIEZIONE PATRIMONIO (Investimento: " << CAPITALE_INIZIALE << ") ---" << std::endl;
    std::cout << "Scenario migliore (1% percentile):   " << portfolioBest << " (+" << (portfolioBest / CAPITALE_INIZIALE - 1) * 100 << "%)" << std::endl;
    std::cout << "Scenario medio (50% percentile): " << portfolioMed << " (+" << (portfolioMed / CAPITALE_INIZIALE - 1) * 100 << "%)"<< std::endl;
    std::cout << "Scenario pessimo (99% percentile):  " << portfolioWorst << " (-" << (1 - portfolioWorst / CAPITALE_INIZIALE) * 100 << "%)" << std::endl;
 */
    return 0;
}
