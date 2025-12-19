import yfinance as yf
import pandas as pd

# 1. Scarica i dati per gli ultimi 10 anni
ticker = "IWDA.AS"
dati = yf.download(ticker, start="2015-12-05", end="2025-12-05")

# 2. Salva solo i prezzi di chiusura in un file CSV
prezzi_chiusura = dati['Close']
prezzi_chiusura.to_csv('msci_world_prezzi.csv')