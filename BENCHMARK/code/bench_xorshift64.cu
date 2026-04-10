#include <iostream>
#include <fstream>

// CUDA
#include <cuda.h>
#include <curand_kernel.h>

#define N (1<<26)

struct MyXS64State {
    uint64_t s;
};

__device__ void init(MyXS64State& st, uint64_t seed, int tid) {
    st.s = seed ^ (uint64_t)tid;
}

__device__ uint64_t myxs64_step(uint64_t& x) {
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return x * 0x2545F4914F6CDD1DULL;
}

__global__ void generate(uint64_t* out, int n, uint64_t seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    MyXS64State st;
    init(st, seed, tid);

    int i = tid;
    int stride = blockDim.x * gridDim.x;

    while (i < n) {
        out[i] = myxs64_step(st.s);
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
    std::ofstream file("xorshift64.bin", std::ios::binary);
    file.write((char*)h_out, N * sizeof(uint64_t));
    file.close();

    cudaFree(d_out);
    delete[] h_out;

    std::cout << "File generato: xorshift64.bin (" << N << " uint64)\n";
}