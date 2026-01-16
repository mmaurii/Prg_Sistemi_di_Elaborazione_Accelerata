// Per compilare: nvcc nomefile.cu -o nomeexec

// Definizione tipica di un kernel
__global__ void mioKernel(){
    // Identificatori
    int idThreadX = threadIdx.x;
    int idBlockX = blockIdx.x;
}

int main(void){

    /* INTRO */

    // Lancio del kernel
    // 1 blocco, 10 thread
    mioKernel<<<1, 10>>>();

    // Aspetta che la GPU abbia finito su tutti i thread
    cudaDeviceSynchronize();


    /* INFO DETTAGLIATE SULLA GPU */
    cudaDeviceProp prop;

    // Ottieni info su dispositivo 0
    cudaGetDeviceProperties(&prop, 0);
    // Nome
    prop.name;
    // Memoria globale totale in byte
    prop.totalGlobalMem;
    // Clock dei core in Hz
    prop.clockRate;
    // Compute capability
    prop.major, prop.minor;
    

    /* OPERAZIONI IN MEMORIA CUDA */
    float* d_array;
    size_t size = 10 * sizeof(float);
    
    // Lancio con controllo errori
    cudaError_t e = cudaMalloc((void**) &d_array, size);
    if(e != cudaSuccess){
        // Restituisce stringa d'errore
        cudaGetErrorString(e);
    }

    // Trasferimento dati
    size_t size = 10 * sizeof(float);
    // Dati su host
    float* h_data = (float*)malloc(size);
    for(int i = 0; i < 10; ++i) h_data[i] = (float)i;
    // Dati su device
    float* d_data;
    cudaError_t err = cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    // Stessa gestione degli errori
    /*
        Tipi di trasferimento:
        - cudaMemcpyHostToHost
        - cudaMemcpyHostToDevice
        - cudaMemcpyDeviceToHost
        - cudaMemcpyDeviceToDevice
    */
    // Copia inversa
    err = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    // Liberazione memoria, host e device
    free(h_data);
    cudaFree(d_data);

    /* GERARCHIA DI THREAD */
    // BLOCCO: i thread al suo interno vengono eseguiti LOGICAMENTE in parallelo, si possono sincronizzare e condividere memoria
    // I thread di blocchi diversi NON si possono sincronizzare direttamente, solo tramite memoria globale o kernel successivi
    // 1D, 2D, 3D per griglie e blocchi

    // Nel lancio di un kernel
    // gridSize: numero di blocchi
    // blockSize: numero di thread per blocco
    kernel_name <<<gridSize, blockSize>>>(args);

    // Init griglie e blocchi
    // 3D Grid, 1D Block
    dim3 gridSize(4, 2, 2);
    dim3 blockSize(8);
    // 3D Grid, 2D Block
    dim3 gridSize(4, 2, 2);
    dim3 blockSize(8, 4);

    // MAX thread per blocco: 1024
    // Anche su dimensioni diverse, il prodotto non deve superare 1024 (per esempio (2048, 1, 1) o (64, 32, 1) non vanno bene)

    // COMPUTE CAPABILITY: indica le caratteristiche e le capacita' di una GPU NVIDIA in termini di funzionalita' supportate e limit hardware

    /* QUALIFICATORI CUDA */
    // Chiamabile da CPU, eseguita su GPU
    // __global__ ritorna solo void, le comunicazioni avvengono solo tramite memotia
    /*
        __global__ e __device__:
            - Accedono solo alla memoria della GPU
            - Non hanno un numero variabile di argomenti
            - Tutte le variabili devono essere passate come argomenti o allocate dinamicamente
            - Non supportano i puntatori a funzione
            - Vengono lanciati in modo asincrono, salvo sincronizzazioni implicite
    */ 
    __global__ void kernelFunction(int *data, int size);
    // Chiamabile da GPU, eseguita su GPU
    __device__ int deviceHelper(int x);
    // Eseguibile su CPU
    __host__ int hostFunction(int x);
    // Combinando i qualificatori __host__ e __device__ è possibile richiamare le funzioni sia su CPU che su GPU
    __host__ __device__ int hostDeviceFunction(int x);

    /* SOMMA DI VETTORI IN C E CUDA C */
    void sumArraysOnHost(float *A, float *B, float *C, int N){
        for(int idx = 0; idx < N; idx++){
            C[idx] = A[idx] + B[idx];
        }
    }
    sumArraysOnHost(A, B, C, N);

    __global__ void sumArraysOnGPU(float *A, float *B, float *C, int N){
        int idx = blockDim.x * blockIdx.x + threadIdx.x;
        // L'if si mette per evitare accessi illeciti alla memoria
        if(idx < N) C[idx] = A[idx] + B[idx];
    }
    // Griglia 1D, Blocco 1D, 3 blocchi da 4 thread
    sumArraysOnGPU<<<gridDim, blockDim>>>(A, B, C, N);

    /* CALCOLO INDICI GLOBALI */
    // Griglia 1D, Blocco 1D
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Griglia 1D, Blocco 2D
    // Indice di thread lungo l'asse X
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    // Idem lungo l'asse Y
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    // Numero di thread per riga, 16
    int nx = gridDim.x * blockDim.x;
    // Somma di ix con la coordinata iy moltiplicata per il numero di thread per riga
    int global_idx = iy * nx + ix;

    // In generale
    // Caso 1D
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = x;
    // Caso 2D
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * nx + x;
    // Caso 3D
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = z * (ny * nx) + y * nx + x;

    /* CALCOLO DIMENSIONE GRIGLIA E BLOCCO */
    // A MANO dimensione blocco (thread per blocco)
    // AUTOMATICAMENTE dimensione griglia
    int blockSize = 256;
    int dataSize = 1030;
    dim3 blockDim(blockSize);
    dim3 gridDim((dataSize + blockSize - 1) / blockSize);
    kernel_name<<<gridDim, blockDim>>>(args);

    /* MACRO CHECK */
    // Fornisce file, riga, codice e descrizione errore
    #define CHECK(call){
        const cudaError_t error = call;
        if (error != cudaSuccess){
            printf("Error: %s:%d, ", __FILE__, __LINE__);
            printf("code:%d, reason: %s\n", error,
            cudaGetErrorString(error));
            exit(1);
        }
    }

    // Esempi d'uso
    CHECK(cudaMalloc(&d_input, size));
    CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    // Lancia il kernel
    kernel_function <<<numBlocks, blockSize >>>(argument list);
    // Primo controllo: errori di lancio del kernel
    CHECK(cudaGetLastError());
    // Secondo controllo: errori durante l'esecuzione del kernel. Usare solo in DEBUG (Overhead di performance!)
    CHECK(cudaDeviceSynchronize());

    /* PROFILING PRESTAZIONI */
    // Timer CPU
    #include <time.h>
    double cpuSecond() {
        struct timespec ts;
        timespec_get(&ts, TIME_UTC);
        return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
    }

    // Metodo 1: Timer CPU
    // Registra il tempo di inizio
    double iStart = cpuSecond();
    // Lancia il kernel CUDA
    kernel_name <<<grid, block>>>(argument list);
    // Attende il completamento del kernel
    cudaDeviceSynchronize();
    // Calcola il tempo trascorso
    double iElaps = cpuSecond() - iStart;

    // Metodo 2: NVIDIA Profiler
    // nvprof -o file.nvvp ./app
    // 5.0 <= Compute Capability < 8.0

    // Metodo 3.1: NVIDIA Nsight Systems
    // nsys profile --stats=true ./app

    // Metodo 3.2: NVIDIA Nsight Compute
    // ncu --set full -o test_report ./app

    

    return 0;
}