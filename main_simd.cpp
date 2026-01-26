/*
    Questo codice è parte del progetto di SISTEMI DI ELABORAZIONE ACCELLERATA M, implementa una simulazione 
    montecarlo partendo da dati storici scaricati da yfinance. L'obiettivo è stimare il valore futuro di un asset
    o un portafoglio di asset, basandosi su modelli stocastici. In questo modo da possiamo valutare il rischio e il
    potenziale rendimento dell'investimento in un orizzonte temporale definito. 
*/

#pragma GCC target("sse4.2")
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>
#include <immintrin.h>
#include <cmath>
//Aggiunta perchè non trovata in cmath
#define M_PI 3.14159265358979323846

// CONFIGURAZIONE 
const std::string CSV_FILENAME = "DATASET/msci_world_prezzi.csv";
const float T_YEARS = 1.0;         // Orizzonte temporale: 1 anno
const int SEED = 12345;
const float CAPITALE_INIZIALE = 10000.0; // Capitale iniziale investito
const int DAYS_OPEN_IN_YEAR = 252;        // Giorni borsa aperta in un anno

// FUNZIONI DI UTILITÀ

// Funzione per leggere i prezzi dal CSV
std::vector<float> readPrices(const std::string& filename) {
    std::vector<float> prices;
    std::ifstream file(filename);
    std::string line;
    float price;

    if (!file.is_open()) {
        std::cerr << "Errore: Impossibile aprire il file " << filename << std::endl;
        exit(1);
    }

    // Salta l'header (rimuovi questa riga se il CSV non ha intestazione)
    std::getline(file, line); 

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        
        // Se il CSV è "Data,Prezzo", dobbiamo prendere la seconda colonna.
        // Qui assumiamo che l'ultima colonna sia il prezzo.
        float val = 0.0;

        // Suppongo formato CSV: xxx,xxx,prezzo. Proviamo a prendere l'ultimo token.
        size_t lastComma = line.find_last_of(',');
        if (lastComma != std::string::npos) {
            try {
                val = std::stod(line.substr(lastComma + 1));
                prices.push_back(val);
            } catch (...) { continue; }
        }
    }
    return prices;
}

// Calcolo dei parametri Drift e Volatilità
void calculateParameters(const std::vector<float>& prices, float& S0, float& drift, float& volatilita) {
    if(prices.size() < 2) return;

    S0 = prices.back();

    const size_t N = prices.size() - 1;

    __m128 sumVec = _mm_setzero_ps();
    __m128 sqSumVec = _mm_setzero_ps();

//    float r[4] __attribute__((aligned(16)));
    alignas(16) float r[4];

    size_t i = 1;

    for(; i + 3 < prices.size(); i += 4){

        __m128 p1 = _mm_loadu_ps(&prices[i]);
        __m128 p0 = _mm_loadu_ps(&prices[i - 1]);

        __m128 ratio = _mm_div_ps(p1, p0);

        _mm_store_ps(r, ratio);

        for(int k = 0; k < 4; ++k){
            r[k] = std::log(r[k]);
        }

        __m128 ret = _mm_load_ps(r);

        sumVec = _mm_add_ps(sumVec, ret);
        sqSumVec = _mm_add_ps(sqSumVec, _mm_mul_ps(ret, ret));
    }

    __m128 tmp = _mm_hadd_ps(sumVec, sumVec);
    tmp = _mm_hadd_ps(tmp, tmp);

    float sum = _mm_cvtss_f32(tmp);
     tmp = _mm_hadd_ps(sqSumVec, sqSumVec);
    tmp = _mm_hadd_ps(tmp, tmp);
    float sqSum = _mm_cvtss_f32(tmp);

    // Tail scalare
    for (; i < prices.size(); ++i) {
        float r0 = std::log(prices[i] / prices[i - 1]);
        sum   += r0;
        sqSum += r0 * r0;
    }

    float mean  = sum / N;
    float stdev = std::sqrt(sqSum / N - mean * mean);

    drift = mean * DAYS_OPEN_IN_YEAR;
    volatilita = stdev * std::sqrt(DAYS_OPEN_IN_YEAR);

}

// FUNZIONI SIMD RANDOM XORSHIFT
inline __m128i xorshift128(__m128i x){
    x = _mm_xor_si128(x, _mm_slli_epi32(x, 13));
    x = _mm_xor_si128(x, _mm_srli_epi32(x, 17));
    x = _mm_xor_si128(x, _mm_slli_epi32(x, 5));
    return x;
}

// Passaggio da interi a float tra 0 e 1
inline __m128 uniform01SIMD(__m128i& state) {
    state = xorshift128(state);

    __m128i masked = _mm_and_si128(state, _mm_set1_epi32(0x7fffffff));
    __m128 f = _mm_cvtepi32_ps(masked);

    return _mm_mul_ps(f, _mm_set1_ps(1.0f / 2147483648.0f));
}


// Passaggio da distribuzione uniforme a normale dei valori casuali (Box-Muller)
inline __m128 normalSIMD(__m128i& state) {
    __m128 u1 = uniform01SIMD(state);
    __m128 u2 = uniform01SIMD(state);

//    float U1[4] __attribute__((aligned(16)));
//    float U2[4] __attribute__((aligned(16)));
//    float Z[4] __attribute__((aligned(16)));
    alignas(16) float U1[4];
    alignas(16) float U2[4];
    alignas(16) float Z[4];

    _mm_store_ps(U1, u1);
    _mm_store_ps(U2, u2);

    for (int i = 0; i < 4; ++i) {
        U1[i] = std::max(U1[i], 1e-7f);
        Z[i] = std::sqrt(-2.0f * std::log(U1[i])) *
               std::cos(2.0f * M_PI * U2[i]);
    }

    return _mm_load_ps(Z);
}

// FUNZIONI SEQUENZIALI RANDOM XORSHIFT
inline uint32_t xorshift32(uint32_t& state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

inline float uniform01Seq(uint32_t& state) {
    return (xorshift32(state) >> 8) * (1.0f / 16777216.0f);
}

inline float normalSeq(uint32_t& state) {
    float u1 = uniform01Seq(state);
    float u2 = uniform01Seq(state);

    return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
}

int main(int argc, char* argv[]) {
    // Valore di default se l'utente non inserisce argomenti
    long nSimulations = 10000000; 

    if (argc > 1) {
        nSimulations = std::stol(argv[1]);
    }
    std::cout << "=== Monte Carlo SIMD CPU ===" << std::endl;
    
    // Caricamento Dati
    std::cout << "Lettura dati da " << CSV_FILENAME << "..." << std::endl;
    std::vector<float> prices = readPrices(CSV_FILENAME);
    std::cout << "Letti " << prices.size() << " prezzi storici." << std::endl;

    // Calcolo Parametri
    float S0, drift, volatilita;
    auto startTime = std::chrono::high_resolution_clock::now();
    calculateParameters(prices, S0, drift, volatilita);
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = endTime - startTime;

    std::cout << "Cariamento parametri: " << elapsed.count() << " ms." << std::endl;

    std::cout << "Prezzo Iniziale (S0): " << S0 << std::endl;
    std::cout << "Drift Annualizzato: " << drift << " (" << drift*100 << "%)" << std::endl;
    std::cout << "Volatilita' Annualizzata: " << volatilita << " (" << volatilita*100 << "%)" << std::endl;

    // Simulazione Monte Carlo SIMD su CPU
    std::cout << "\nAvvio Simulazione (" << nSimulations << " iterazioni)..." << std::endl;

    std::vector<float> simulatedPortfolioValues(nSimulations);
    
    // Setup Random Number Generator (Xorshift)
    // Usiamo un seed fisso per riproducibilità dei risultati
    __m128i rngState;                    
    rngState = _mm_set_epi32(SEED ^ 0x12345678, SEED ^ 0xABCDEF09, SEED ^ 0x19102001, SEED ^ 0x01012003);

    // Set up parametri
    float driftTerm = (drift - 0.5 * volatilita * volatilita) * T_YEARS;
    float volTerm = volatilita * std::sqrt(T_YEARS);

    // Parametri vettorializzati per SIMD
    __m128 driftV = _mm_set1_ps(driftTerm);
    __m128 volV = _mm_set1_ps(volTerm);
    __m128 S0V = _mm_set1_ps(S0);

//    float out[4] __attribute__((aligned(16)));
    alignas(16) float out[4];
      
    // Timer Start
    startTime = std::chrono::high_resolution_clock::now();

    // Simulazione Monte Carlo SIMD
    for(long i = 0; i + 3 < nSimulations; i += 4){
        __m128 Z = normalSIMD(rngState);
        __m128 X = _mm_add_ps(driftV, _mm_mul_ps(volV, Z));

        _mm_store_ps(out, X);
        for(int k = 0; k < 4; ++k){
            out[k] = S0 * std::exp(out[k]);
        }

        __m128 ST = _mm_load_ps(out);
        _mm_storeu_ps(&simulatedPortfolioValues[i], ST);
    }

    // Stato random per parte sequenziale
    uint32_t seqState = 0x1A2B3C4Du + nSimulations;

    // Tail sequenziale finale (da 1 a 3 simulazioni) per nSimulations non multiplo di 4
    for(long i = (nSimulations & ~3); i < nSimulations; ++i){
        float Z = normalSeq(seqState);
        simulatedPortfolioValues[i] = S0 * std::exp(driftTerm + volTerm * Z);
    }

    // Timer End
    endTime = std::chrono::high_resolution_clock::now();
    elapsed = endTime - startTime;

    std::cout << "Simulazione CPU completata in: " << elapsed.count() << " ms." << std::endl;

    std::cout << "Analisi dei Risultati..." << std::endl;
    
    startTime = std::chrono::high_resolution_clock::now();

    // Ordiniamo per trovare il percentile
    std::sort(simulatedPortfolioValues.begin(), simulatedPortfolioValues.end());

    // Timer End
    endTime = std::chrono::high_resolution_clock::now();
    elapsed = endTime - startTime;
    std::cout << "Tempo sort: " << elapsed.count() << " ms." << std::endl;

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
