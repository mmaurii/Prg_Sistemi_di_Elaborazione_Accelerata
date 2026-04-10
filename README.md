# Progetto di Sistemi di Elaborazione Accelerata M

## Descrizione del Progetto
Questo progetto implementa una simulazione Monte Carlo utilizzando dati storici scaricati da yfinance. L'obiettivo è stimare il valore futuro di un asset o di un portafoglio di asset, basandosi su modelli stocastici. Questo approccio consente di valutare il rischio e il potenziale rendimento dell'investimento in un orizzonte temporale definito.

## File Principali
- [**main_gpu_10.cu**](main_gpu_10.cu): Implementa la simulazione Monte Carlo utilizzando CUDA per sfruttare la potenza di calcolo delle GPU.
- [**main_naif_cpu_0.cpp**](main_naif_cpu_0.cpp): Versione della simulazione che utilizza la CPU, utile per confronti di prestazioni.

## Risorse Utili
- [**cheatsheet_CUDA.cu**](risorse/cheatsheet_CUDA.cu): Contiene informazioni utili per la programmazione CUDA, inclusa la definizione di un kernel e la sincronizzazione dei thread.
- [**simdCheatsheet.cpp**](risorse/simdCheatsheet.cpp): Fornisce una panoramica sui tipi di dati SIMD e le direttive di compilazione necessarie per l'ottimizzazione delle prestazioni.

## Compilazione
Per compilare i file CUDA, utilizzare il comando:
```
nvcc nomefile.cu -o nomeexec
```
Per i file C++, utilizzare il compilatore g++:
```
g++ nomefile.cpp -o nomeexec
```

## Esecuzione
Eseguire i file compilati per avviare la simulazione e analizzare i risultati.

## Profiling CUDA
Per analizzare le prestazioni dei file CUDA, si consiglia di utilizzare i seguenti strumenti:

### NVIDIA Systems Profiler (nsys)
Profiler di sistema che fornisce una vista completa delle prestazioni, inclusi i kernel CUDA, i trasferimenti di memoria e le operazioni di CPU/GPU:
```
nsys profile -o nomefile ./executable
```

### NVIDIA Compute Profiler (ncu)
Profiler dettagliato per analizzare le metriche specifiche dei kernel CUDA, come l'utilizzo della memoria, la larghezza di banda e l'occupancy:
```
ncu --set full -o nomefile ./executable
ncu -i nomefile.ncu-rep
```

Questi strumenti permettono di identificare colli di bottiglia e ottimizzare il codice CUDA per ottenere migliori prestazioni.

## Cartella BENCHMARK
La cartella [BENCHMARK/](BENCHMARK/) contiene una suite dedicata al confronto qualitativo dei generatori pseudo-casuali usati nelle versioni GPU.

### Struttura
- [BENCHMARK/code/](BENCHMARK/code/): programmi CUDA per generare stream binari (`.bin`) da diversi RNG (CURAND, xorshift64, xoroshiro128, splitmix64, pcg32, incluse alcune varianti `_s64`).
- [BENCHMARK/results/dieharder/](BENCHMARK/results/dieharder/): output dei test `dieharder`.
- [BENCHMARK/results/practrand/](BENCHMARK/results/practrand/): output dei test `PractRand`.

### Esempio di utilizzo
1. Compilare un benchmark RNG:
	```bash
	nvcc BENCHMARK/code/bench_xorshift64.cu -O3 -o bench_xorshift64
	```
2. Generare il file binario:
	```bash
	./bench_xorshift64
	```
3. Eseguire i test statistici:
	```bash
	dieharder -a -g 201 -f xorshift64.bin > BENCHMARK/results/dieharder/die_xorshift64.txt
	RNG_test stdin64 -tf 512MB < xorshift64.bin > BENCHMARK/results/practrand/pra_xorshift64.txt
	```

I risultati già inclusi in [BENCHMARK/results/](BENCHMARK/results/) permettono di confrontare rapidamente i vari generatori anche senza rieseguire tutta la pipeline.

## Presentazione del Progetto
Per una visione più completa del progetto, inclusi i risultati ottenuti, le analisi di prestazioni e i confronti tra le diverse implementazioni (CPU vs GPU, SIMD, etc.), consultare [la presentazione associata](risorse/PRG_SISTEMI_DI_ELABORAZIONE_ACCELERATA.pdf).

## Dataset
Il progetto utilizza dati storici di prezzi scaricati da yfinance tramite gli script [collectData.py](collectData.py) e [collectData1.py](collectData1.py). I dataset sono memorizzati nella cartella [DATASET/](DATASET/) e includono i prezzi di indici e asset finanziari come S&P500, MSCI World, Nikkei 225 e DAX.