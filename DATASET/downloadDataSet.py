import os
import yfinance as yf
import pandas as pd

# 1. Scarica i dati per gli ultimi 10 anni
dirName = "dataset"
fileName = "msci_world_prezzi.csv"

path = os.path.join(dirName, fileName)

if not os.path.isfile(path):
    ticker = "IWDA.AS"
    dati = yf.download(ticker, start="2015-12-05", end="2025-12-05")

    # 2. Salva solo i prezzi di chiusura in un file CSV
    prezzi_chiusura = dati['Close']
    prezzi_chiusura.to_csv(path)

# 3. Carica i dati dal file CSV
dati_caricati = pd.read_csv(path, parse_dates=True)
dati_caricati['Date'] = pd.to_datetime(dati_caricati.index)
print(dati_caricati.head())

# 4. Data cleaning: controllo se ci sono valori mancanti
print("\nNumber of null values: "+str(dati_caricati.isnull().sum()["IWDA.AS"]))
print("Number of Nan values: "+str(dati_caricati.isna().sum()["IWDA.AS"]))
print("Number of rows: "+str(len(dati_caricati)))

duplicati = dati_caricati.index[dati_caricati.index.duplicated()]
print("Duplicated dates: "+str(len(duplicati)))