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

/* ==== STRUTTURA E FUNZIONI PER RNG PCG32 ==== */

// Struttura di stato
struct MyPCG32State { uint64_t state; uint64_t inc; };

//Step RNG
__device__ uint32_t mypcg32_step(MyPCG32State* rng) {
    uint64_t oldstate = rng->state;
    // LCG classico: stato successivo
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    // Permutazione: XOR-shift e rotazione bit
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// Init random generator (coerente con bench_pcg32.cu)
__device__ void mypcg32_init_rng(MyPCG32State& st, uint64_t initstate, uint64_t initseq) {
    st.state = 0U;
    st.inc   = (initseq << 1u) | 1u;

    mypcg32_step(&st);      // warm-up
    st.state += initstate;
    mypcg32_step(&st);      // decorrelazione
}

// Da variabile uniforme a float (Intervallo 0..1)
__device__ __forceinline__
float mypcg32_u01(uint32_t x) {
    return (x >> 8) * (1.0f / (1u << 24));
}

// Conversione da distribuzione normale a gaussiana
// Metodo Box-Muller - Genera 2 numeri gaussiani
__device__ float2 mypcg32_normal2(MyPCG32State& st) {
    float u1 = mypcg32_u01(mypcg32_step(&st));
    float u2 = mypcg32_u01(mypcg32_step(&st));

    // Assicuriamoci che u1 non sia esattamente 0 per evitare log(0) = -inf
    // fmaxf è un'istruzione hardware rapida che prende il massimo tra u1 e 
    // un piccolo valore positivo (epsilon) per evitare problemi numerici con log(0)
    u1 = fmaxf(u1, 5.9604644775390625e-08f); 

    // Calcolo del raggio (utilizzando l'intrinseco per log)
    float r = sqrtf(-2.0f * __logf(u1));

    float s, c;
    // sincospif calcola simultaneamente sin(pi * x) e cos(pi * x)
    // Moltiplicando u2 per 2.0f otteniamo l'angolo corretto
    sincospif(2.0f * u2, &s, &c);

    // Restituisce la coppia Gaussiana
    return make_float2(r * c, r * s);
}

// Funzione ausiliaria per CUDA - Genera 4 numeri gaussiani con 2 iterazioni
__device__ float4 mypcg32_normal4(MyPCG32State& st) {
    float2 pair1 = mypcg32_normal2(st);
    float2 pair2 = mypcg32_normal2(st);
    return make_float4(pair1.x, pair1.y, pair2.x, pair2.y);
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

    MyPCG32State state;
    mypcg32_init_rng(state, seed, tid);

    // Genero 4 numeri casuali in un colpo solo (istruzione vettoriale)
    float4 Z = mypcg32_normal4(state);
    
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
