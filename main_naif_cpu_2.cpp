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
#include <iomanip>

struct index
{
    std::string name;
    std::string filename;
};

// --- CONFIGURAZIONE
const std::string CSV_FILENAME_MSCI = "DATASET/msci_world_prezzi.csv";
const std::string CSV_FILENAME_SP500 = "DATASET/S&P500_prezzi.csv";
const std::string CSV_FILENAME_GDAXI = "DATASET/GDAXI_prezzi.csv";
const std::string CSV_FILENAME_N225 = "DATASET/N225_prezzi.csv";
const std::string MSCI = "MSCI";
const std::string SP500 = "S&P500";
const std::string GDAXI = "GDAXI";
const std::string N225 = "N225";
const index indexes[4] = {
    {MSCI, CSV_FILENAME_MSCI},
    {SP500, CSV_FILENAME_SP500},
    {GDAXI, CSV_FILENAME_GDAXI},
    {N225, CSV_FILENAME_N225}
};

const float T_YEARS = 1.0; // Orizzonte temporale: 10 anni
const int SEED = 12345;
const float CAPITALE_INIZIALE = 10000.0; // Capitale iniziale investito
const int DAYS_OPEN_IN_YEAR = 252;       // Giorni borsa aperta in un anno

// --- FUNZIONI DI UTILITÀ ---
// Funzione per leggere i prezzi dal CSV
std::vector<float> readPrices(const std::string &filename)
{
    std::vector<float> prices;
    std::ifstream file(filename);
    std::string line;
    float price;

    if (!file.is_open())
    {
        std::cerr << "Errore: Impossibile aprire il file " << filename << std::endl;
        exit(1);
    }

    // Salta l'header (rimuovi questa riga se il CSV non ha intestazione)
    std::getline(file, line);

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;

        // Se il CSV è "Data,Prezzo", dobbiamo prendere la seconda colonna.
        // Qui assumiamo che l'ultima colonna sia il prezzo.
        float val = 0.0;

        // Suppongo formato CSV: xxx,xxx,prezzo. Proviamo a prendere l'ultimo token.
        size_t lastComma = line.find_last_of(',');
        if (lastComma != std::string::npos)
        {
            try
            {
                val = std::stod(line.substr(lastComma + 1));
                prices.push_back(val);
            }
            catch (...)
            {
                continue;
            }
        }
    }
    return prices;
}

// Calcolo dei parametri Drift e Volatilità
void calculateParameters(const std::vector<float> &prices, float &S0, float &drift, float &volatilita)
{
    if (prices.empty())
    {
        return;
    }

    S0 = prices.back(); // L'ultimo prezzo è il punto di partenza (S0)

    std::vector<float> logReturns;
    for (size_t i = 1; i < prices.size(); ++i)
    {
        float r = std::log(prices[i] / prices[i - 1]);
        logReturns.push_back(r);
    }

    // Calcolo Media (Drift giornaliero)
    float mean = 0.0;
    for (float &r : logReturns)
    {
        mean += r;
    }
    mean /= logReturns.size();

    // Calcolo Deviazione Standard (Volatilità giornaliera)
    float sqSum = 0.0;
    for (float &r : logReturns)
    {
        sqSum += r * r;
    }

    float stdev = std::sqrt(sqSum / logReturns.size() - mean * mean);

    // Annualizzazione
    drift = mean * DAYS_OPEN_IN_YEAR;
    volatilita = stdev * std::sqrt(DAYS_OPEN_IN_YEAR);
}

// --- MAIN ---
int main(int argc, char *argv[])
{
    std::chrono::duration<float,std::milli> elapsedSort;
    std::chrono::duration<float,std::milli> elapsedMC;
    std::chrono::duration<float,std::milli> elapsedParam;

    // Valore di default se l'utente non inserisce argomenti
    long nSimulations = 10000000;

    if (argc > 1)
    {
        // Converte l'argomento della riga di comando in numero
        nSimulations = std::stol(argv[1]);
    }

    
    for (const auto &index : indexes)
    {
        // 1. lettura dati
        std::vector<float> prices = readPrices(index.filename);

        std::cout << "=== Monte Carlo CPU Baseline per " << index.name << " ===" << std::endl;

        // 2. Calcolo Parametri
        float S0, drift, volatilita;

        auto startTime = std::chrono::high_resolution_clock::now();
        calculateParameters(prices, S0, drift, volatilita);
        auto endTime = std::chrono::high_resolution_clock::now();
        elapsedParam+= endTime - startTime;
        
        std::cout << "Prezzo Iniziale (S0): " << S0 << std::endl;
        std::cout << "Drift Annualizzato: " << drift << " (" << drift * 100 << "%)" << std::endl;
        std::cout << "Volatilita' Annualizzata: " << volatilita << " (" << volatilita * 100 << "%)" << std::endl;

        // 3. Simulazione Monte Carlo Naive su CPU
        std::cout << "\nAvvio Simulazione (" << nSimulations << " iterazioni)..." << std::endl;
        
        std::vector<float> simulatedIndexValues(nSimulations);
        
        // Setup Random Number Generator (Standard C++)
        // Usiamo un seed fisso per riproducibilità dei risultati
        std::mt19937 generator(SEED);
        std::normal_distribution<float> distribution(0.0, 1.0);
    
        
        // Timer Start
        startTime = std::chrono::high_resolution_clock::now();
        
        // Loop Principale (Collo di bottiglia)
        float driftTerm = (drift - 0.5 * volatilita * volatilita) * T_YEARS;
        float volTerm = volatilita * std::sqrt(T_YEARS);

        for (int i = 0; i < nSimulations; ++i)
        {
            float Z = distribution(generator);                 // Generazione numero casuale
            float ST = S0 * std::exp(driftTerm + volTerm * Z); // Formula GBM
            simulatedIndexValues[i] = ST;
        }
        
        // Timer End
        endTime = std::chrono::high_resolution_clock::now();
        elapsedMC += endTime - startTime;
        
        startTime = std::chrono::high_resolution_clock::now();
        
        // Ordiniamo per trovare il percentile
        std::sort(simulatedIndexValues.begin(), simulatedIndexValues.end());
        
        // Timer End
        endTime = std::chrono::high_resolution_clock::now();
        elapsedSort += endTime - startTime;
        
        // Scenario Peggiore (1% percentile - Potential Downside)
        int idxWorst = (int)(nSimulations * 0.01f);
        float priceWorst = simulatedIndexValues[idxWorst];
        float portfolioWorst = CAPITALE_INIZIALE * (priceWorst / S0);
        
        // Scenario Mediano (50% percentile - Valore più probabile)
        int idxMed = (int)(nSimulations * 0.50f);
        float priceMed = simulatedIndexValues[idxMed];
        float portfolioMed = CAPITALE_INIZIALE * (priceMed / S0);
        
        // Scenario Migliore (99% percentile - Potential Upside)
        int idxBest = (int)(nSimulations * 0.99f);
        float priceBest = simulatedIndexValues[idxBest];
        float portfolioBest = CAPITALE_INIZIALE * (priceBest / S0);
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\n--- PROIEZIONE PATRIMONIO (Investimento: " << CAPITALE_INIZIALE << ") ---" << std::endl;
        std::cout << "Scenario migliore (1% percentile):   " << portfolioBest << " (+" << (portfolioBest / CAPITALE_INIZIALE - 1) * 100 << "%)" << std::endl;
        std::cout << "Scenario medio (50% percentile): " << portfolioMed << " (+" << (portfolioMed / CAPITALE_INIZIALE - 1) * 100 << "%)" << std::endl;
        std::cout << "Scenario pessimo (99% percentile):  " << portfolioWorst << " (-" << (1 - portfolioWorst / CAPITALE_INIZIALE) * 100 << "%)" << std::endl;
    }
    
    std::cout << "\nPARAMETRI: " << elapsedParam.count() << " ms." << std::endl;
    std::cout << "MONTECARLO: " << elapsedMC.count() << " ms." << std::endl;
    std::cout << "SORT: " << elapsedParam.count() << " ms." << std::endl;
    
    return 0;
}
