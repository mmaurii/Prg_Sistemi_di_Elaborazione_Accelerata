#include <iostream>
#include <fstream>

// CUDA
#include <cuda.h>
#include <curand_kernel.h>

#define N (1<<26)

struct MySM64State {
    uint64_t state;
};

// Step RNG
__device__ uint64_t mysm64_step(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Init random generator
__device__ void init(MySM64State& st, uint64_t seed, int tid) {
    uint64_t x = seed + (uint64_t)tid; 
    //Passo di avanzamento
    st.state = mysm64_step(x);
}

__global__ void generate(uint64_t* out, int n, uint64_t seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    MySM64State st;
    init(st, seed, tid);

    int i = tid;
    int stride = blockDim.x * gridDim.x;

    while (i < n) {
        out[i] = mysm64_step(st.state);
        i += stride;
    }
}

int main(void) {
    uint64_t* d_out;
    uint64_t* h_out = new uint64_t[N];

    cudaMalloc(&d_out, N * sizeof(uint64_t));

    int blockSize = 256;
    int gridSize = 1024;

    generate<<<gridSize, blockSize>>>(d_out, N, 12345ULL);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Scrittura BINARIA (fondamentale!)
    std::ofstream file("splitmix64.bin", std::ios::binary);
    file.write((char*)h_out, N * sizeof(uint64_t));
    file.close();

    cudaFree(d_out);
    delete[] h_out;

    std::cout << "File generato: splitmix64.bin (" << N << " uint64)\n";
}