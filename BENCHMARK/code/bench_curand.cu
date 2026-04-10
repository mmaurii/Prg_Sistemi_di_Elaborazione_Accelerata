#include <iostream>
#include <fstream>

// CUDA
#include <cuda.h>
#include <curand_kernel.h>

#define N (1<<26)

__global__ void generate(uint64_t* out, int n, uint64_t seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    int i = tid;
    int stride = blockDim.x * gridDim.x;

    while (i < n) {
        // Genera 4 numeri a 32 bit
        uint4 r = curand4(&state);

        // Combina in 2 uint64
        if (i < n) {
            out[i] = ((uint64_t)r.x << 32) | r.y;
            i += stride;
        }
        if (i < n) {
            out[i] = ((uint64_t)r.z << 32) | r.w;
            i += stride;
        }
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

    // Scrittura BINARIA
    std::ofstream file("curand.bin", std::ios::binary);
    file.write(reinterpret_cast<char*>(h_out), N * sizeof(uint64_t));
    file.close();

    cudaFree(d_out);
    delete[] h_out;

    std::cout << "File generato: curand.bin (" << N << " uint64)\n";
}