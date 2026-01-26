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
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>

// CONFIGURAZIONE
const std::string CSV_FILENAME = "DATASET/msci_world_prezzi.csv";
const double T_YEARS = 1.0;               // Orizzonte temporale: 10 anni
const double CAPITALE_INIZIALE = 10000.0; // Investimento ipotetico iniziale
const int DAYS_OPEN_IN_YEAR = 252;        // Giorni borsa aperta in un anno
const int SEED = 12345;

// FUNZIONI DI UTILITÀ 

// Funzione per leggere i prezzi dal CSV
std::vector<double> readPrices(const std::string &filename)
{
    std::vector<double> prices;
    std::ifstream file(filename);
    std::string line;
    double price;

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
        double val = 0.0;

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
void calculateParameters(const std::vector<double> &prices, double &S0, double &drift, double &volatilita)
{
    if (prices.empty())
    {
        return;
    }

    S0 = prices.back(); // L'ultimo prezzo è il punto di partenza (S0)

    std::vector<double> logReturns;
    for (size_t i = 1; i < prices.size(); ++i)
    {
        double r = std::log(prices[i] / prices[i - 1]);
        logReturns.push_back(r);
    }

    // Calcolo Media (Drift giornaliero)
    double mean = 0.0;
    for (double &r : logReturns)
    {
        mean += r;
    }
    mean /= logReturns.size();

    // Calcolo Deviazione Standard (Volatilità giornaliera)
    double sqSum = 0.0;
    for (double &r : logReturns)
    {
        sqSum += r * r;
    }

    double stdev = std::sqrt(sqSum / logReturns.size() - mean * mean);

    // Annualizzazione
    drift = mean * DAYS_OPEN_IN_YEAR;
    volatilita = stdev * std::sqrt(DAYS_OPEN_IN_YEAR);
}

int main(int argc, char* argv[]) {
// Valore di default se l'utente non inserisce argomenti
    long nSimulations = 10000000; 

    if (argc > 1) {
        // Converte l'argomento della riga di comando in numero
        nSimulations = std::stol(argv[1]);
    }
    std::cout << "=== Monte Carlo NAIF CPU Path Dependent ===" << std::endl;

    // Caricamento Dati
    std::cout << "Lettura dati da " << CSV_FILENAME << "..." << std::endl;
    std::vector<double> prices = readPrices(CSV_FILENAME);
    std::cout << "Letti " << prices.size() << " prezzi storici." << std::endl;

    // Calcolo Parametri
    double S0, drift, volatilita;
    calculateParameters(prices, S0, drift, volatilita);

    std::cout << "Prezzo Iniziale (S0): " << S0 << std::endl;
    std::cout << "Drift Annualizzato: " << drift << " (" << drift * 100 << "%)" << std::endl;
    std::cout << "Volatilita' Annualizzata: " << volatilita << " (" << volatilita * 100 << "%)" << std::endl;

    const double DT = 1.0 / static_cast<double>(DAYS_OPEN_IN_YEAR);
    const double driftStep = (drift - 0.5 * volatilita * volatilita) * DT;
    const double volStep = volatilita * std::sqrt(DT);

    // Simulazione Monte Carlo Naive su CPU
    std::cout << "\nAvvio simulazione (" << nSimulations << " cammini x " << T_YEARS << " anni)..." << std::endl;
    std::vector<double> simulatedPortfolioValues(nSimulations);

    // Setup Random Number Generator (Standard C++)
    // Usiamo un seed fisso per riproducibilità dei risultati
    std::mt19937 generator(SEED);
    std::normal_distribution<double> distribution(0.0, 1.0);

    // Timer Start
    auto startTime = std::chrono::high_resolution_clock::now();

    // CORE SIMULATION (Multi-Step)
    for (int i = 0; i < nSimulations; ++i)
    {
        double currentPrice = S0;

        // Loop interno: cammina giorno per giorno
        for (int day = 0; day < DAYS_OPEN_IN_YEAR * T_YEARS; ++day)
        {
            double Z = distribution(generator);
            // Formula esponenziale incrementale
            currentPrice = currentPrice * std::exp(driftStep + volStep * Z);
        }

        // Salviamo il valore dell'indice raggiunto dalla simulazione
        simulatedPortfolioValues[i] = currentPrice;
    }

    // Timer End
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = endTime - startTime;

    std::cout << "Simulazione completata in: " << elapsed.count() << " ms." << std::endl;

    // Analisi dei risultati
    std::cout << "Analisi dei risultati..." << std::endl;
    startTime = std::chrono::high_resolution_clock::now();

    std::sort(simulatedPortfolioValues.begin(), simulatedPortfolioValues.end());

    endTime = std::chrono::high_resolution_clock::now();
    elapsed = endTime - startTime;
    std::cout << "Tempo sort: " << elapsed.count() << " ms." << std::endl;

    // Scenario Peggiore (1% percentile - Potential Downside)
    int idxWorst = (int)(nSimulations * 0.01f);
    double priceWorst = simulatedPortfolioValues[idxWorst];
    double portfolioWorst = CAPITALE_INIZIALE * (priceWorst / S0);

    // Scenario Mediano (50% percentile - Valore più probabile)
    int idxMed = (int)(nSimulations * 0.50f);
    double priceMed = simulatedPortfolioValues[idxMed];
    double portfolioMed = CAPITALE_INIZIALE * (priceMed / S0);

    // Scenario Migliore (99% percentile - Potential Upside)
    int idxBest = (int)(nSimulations * 0.99f);
    double priceBest = simulatedPortfolioValues[idxBest];
    double portfolioBest = CAPITALE_INIZIALE * (priceBest / S0);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n--- PROIEZIONE PATRIMONIO (Investimento: " << CAPITALE_INIZIALE << " ) ---" << std::endl;
    std::cout << "Scenario migliore (1% percentile):   " << portfolioBest << " (+" << (portfolioBest / CAPITALE_INIZIALE - 1) * 100 << "%)" << std::endl;
    std::cout << "Scenario medio (50% percentile): " << portfolioMed << " (+" << (portfolioMed / CAPITALE_INIZIALE - 1) * 100 << "%)"<< std::endl;
    std::cout << "Scenario pessimo (99% percentile):  " << portfolioWorst << " (-" << (1 - portfolioWorst / CAPITALE_INIZIALE) * 100 << "%)" << std::endl;

    return 0;
}
