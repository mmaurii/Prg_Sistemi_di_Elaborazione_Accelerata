#include <iostream>
#include <fstream>

// CUDA
#include <cuda.h>
#include <curand_kernel.h>

#define N (1<<26)

struct MyXS128State {
    uint64_t s[2];
    float spare;
    bool hasSpare;
};

__device__ uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
__device__ void init(MyXS128State& st, uint64_t seed, int tid) {
    uint64_t x = seed ^ (uint64_t)tid;
    st.s[0] = splitmix64(x);
    st.s[1] = splitmix64(x);
    st.hasSpare = false;
}

// Funzione ausiliaria di rotazione
__device__ __forceinline__
uint64_t myxs128_rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// Step Xoroshiro128+
__device__ __forceinline__
uint64_t myxs128_step(uint64_t s[2]) {
    uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    uint64_t result = s0 + s1;

    s1 ^= s0;
    s[0] = myxs128_rotl(s0, 55) ^ s1 ^ (s1 << 14);
    s[1] = myxs128_rotl(s1, 36);

    return result;
}

__global__ void generate(uint64_t* out, int n, uint64_t seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    MyXS128State st;
    init(st, seed, tid);

    int i = tid;
    int stride = blockDim.x * gridDim.x;

    while (i < n) {
        out[i] = myxs128_step(st.s);
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
    std::ofstream file("xoroshiro128_s64.bin", std::ios::binary);
    file.write((char*)h_out, N * sizeof(uint64_t));
    file.close();

    cudaFree(d_out);
    delete[] h_out;

    std::cout << "File generato: xorshiro128_s64.bin (" << N << " uint64)\n";
}