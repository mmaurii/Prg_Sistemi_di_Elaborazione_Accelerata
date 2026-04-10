#include <iostream>
#include <fstream>

// CUDA
#include <cuda.h>
#include <curand_kernel.h>

#define N (1<<26)  // numero di uint64

struct MyPCG32 {
    uint64_t state;
    uint64_t inc;
};

// Step RNG
__device__ uint32_t pcg32_random(MyPCG32* rng) {
    uint64_t oldstate = rng->state;

    // Advance LCG
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;

    // Output permutation (XSH RR)
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;

    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// Seeding
__device__ void pcg32_seed(MyPCG32* rng, uint64_t initstate, uint64_t initseq) {
    rng->state = 0U;
    rng->inc   = (initseq << 1u) | 1u;

    pcg32_random(rng);          // warm-up
    rng->state += initstate;
    pcg32_random(rng);          // decorrelazione
}

__global__ void generate(uint64_t* out, int n, uint64_t seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    MyPCG32 rng;

    // Stream indipendente per thread
    pcg32_seed(&rng, seed, tid);

    int i = tid;
    int stride = blockDim.x * gridDim.x;

    while (i < n) {
        // Combina 2 output -> 64 bit veri
        uint32_t a = pcg32_random(&rng);
        uint32_t b = pcg32_random(&rng);

        out[i] = ((uint64_t)a << 32) | b;

        i += stride;
    }
}

int main(void) {
    uint64_t* d_out;
    uint64_t* h_out = new uint64_t[N];

    cudaMalloc(&d_out, N * sizeof(uint64_t));

    int blockSize = 256;
    int gridSize  = 1024;

    generate<<<gridSize, blockSize>>>(d_out, N, 12345ULL);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Scrittura binaria
    std::ofstream file("pcg32.bin", std::ios::binary);
    file.write(reinterpret_cast<char*>(h_out), N * sizeof(uint64_t));
    file.close();

    cudaFree(d_out);
    delete[] h_out;

    std::cout << "File generato: pcg32.bin (" << N << " uint64)\n";
}