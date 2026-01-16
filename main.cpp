/*
Questo codice implementa una simulazione montecarlo partendo da dati storici scaricati da yfinance.
Il codice parte con una soluzione naive su CPU e prosegue con una versione ottimizzata su GPU usando CUDA.
L'obiettivo è confrontare le prestazioni delle due implementazioni e cercare di ottenere le prestazioni migliori 
possibili per il kernel CUDA, seguendo le best practice per la programmazione GPU, che abbiamo visto a lezione.
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <string>
#include <sstream>

// --- CONFIGURAZIONE 
const std::string CSV_FILENAME = "DATASET/msci_world_prezzi.csv";
const int N_SIMULATIONS = 10000000; // 10 Milioni di simulazioni
const double T_YEARS = 10.0;         // Orizzonte temporale: 10 anni
const double CONFIDENCE_LEVEL = 0.99; // VaR al 99%

// --- FUNZIONI DI UTILITÀ ---

// Funzione per leggere i prezzi dal CSV
std::vector<double> readPrices(const std::string& filename) {
    std::vector<double> prices;
    std::ifstream file(filename);
    std::string line;
    double price;

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
        double val = 0.0;

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
void calculateParameters(const std::vector<double>& prices, double& S0, double& drift, double& volatilita) {
    if (prices.empty()) {
        return;
    }

    S0 = prices.back(); // L'ultimo prezzo è il punto di partenza (S0)
    
    std::vector<double> logReturns;
    for (size_t i = 1; i < prices.size(); ++i) {
        double r = std::log(prices[i] / prices[i-1]);
        logReturns.push_back(r);
    }

    // Calcolo Media (Drift giornaliero)
    double mean = 0.0;
    for(double& r : logReturns) {
        mean += r;
    }
    mean /= logReturns.size();

    // Calcolo Deviazione Standard (Volatilità giornaliera)
    double sqSum = 0.0;
    for(double& r : logReturns) {
        sqSum += r * r;
    }

    double stdev = std::sqrt(sqSum / logReturns.size() - mean * mean);

    // Annualizzazione (assumendo 252 giorni di trading, migliorabile)
    drift = mean * 252.0;
    volatilita = stdev * std::sqrt(252.0);
}

// --- MAIN ---
int main() {
    std::cout << "=== Monte Carlo VaR CPU Baseline ===" << std::endl;
    
    // 1. Caricamento Dati
    std::cout << "Lettura dati da " << CSV_FILENAME << "..." << std::endl;
    std::vector<double> prices = readPrices(CSV_FILENAME);
    std::cout << "Letti " << prices.size() << " prezzi storici." << std::endl;

    // 2. Calcolo Parametri
    double S0, drift, volatilita;
    calculateParameters(prices, S0, drift, volatilita);
    
    std::cout << "Prezzo Iniziale (S0): " << S0 << std::endl;
    std::cout << "Drift Annualizzato: " << drift << " (" << drift*100 << "%)" << std::endl;
    std::cout << "Volatilita' Annualizzata: " << volatilita << " (" << volatilita*100 << "%)" << std::endl;

    // 3. Simulazione Monte Carlo Naive su CPU
    std::cout << "\nAvvio Simulazione (" << N_SIMULATIONS << " iterazioni)..." << std::endl;

    std::vector<double> simulatedPrices(N_SIMULATIONS);
    
    // Setup Random Number Generator (Standard C++)
    // Usiamo un seed fisso per riproducibilità dei risultati
    std::mt19937 generator(12345); 
    std::normal_distribution<double> distribution(0.0, 1.0);

    
    // Timer Start
    auto startTime = std::chrono::high_resolution_clock::now();

    // Loop Principale (Collo di bottiglia)
    double driftTerm = (drift - 0.5 * volatilita * volatilita) * T_YEARS;
    double volTerm = volatilita * std::sqrt(T_YEARS);

    for (int i = 0; i < N_SIMULATIONS; ++i) {
        double Z = distribution(generator); // Generazione numero casuale
        double ST = S0 * std::exp(driftTerm + volTerm * Z); // Formula GBM
        simulatedPrices[i] = ST;
    }

    // Timer End
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;

    std::cout << "Simulazione completata in: " << elapsed.count() << " secondi." << std::endl;

    // 4. Calcolo VaR (Post-processing)
    std::cout << "Calcolo del VaR..." << std::endl;
    
    // Timer Start
    startTime = std::chrono::high_resolution_clock::now();

    // Ordiniamo per trovare il percentile
    std::sort(simulatedPrices.begin(), simulatedPrices.end());

    // Timer End
    endTime = std::chrono::high_resolution_clock::now();
    elapsed = endTime - startTime;

    std::cout << "Tempo sort: " << elapsed.count() << " secondi." << std::endl;

    // Indice per il percentile (es. 5% per confidenza 95%)
    int indexCutoff = static_cast<int>(N_SIMULATIONS * (1.0 - CONFIDENCE_LEVEL));
    double priceAtRisk = simulatedPrices[indexCutoff];
    double varAbsolute = S0 - priceAtRisk;
    double varPercent = (varAbsolute / S0) * 100.0;

    std::cout << "Risultato VaR " << (CONFIDENCE_LEVEL * 100) << "% (" << T_YEARS << " anni):" << std::endl;
    std::cout << "Prezzo peggiore atteso (" << ((1.0 - CONFIDENCE_LEVEL) * 100) << "% dei casi): " << priceAtRisk << std::endl;
    std::cout << "Perdita Massima Stimata: " << varAbsolute << " (" << varPercent << "%)" << std::endl;

    return 0;
}
