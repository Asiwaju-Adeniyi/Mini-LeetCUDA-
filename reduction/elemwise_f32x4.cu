#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>

#define FLOAT4(value)(reinterpret_cast<float4 *> ((&value)) [0])

__global__ void elemwise_f32x4(const float* a, const float* b, float* c, int N) {
    int idx = 4 * (threadIdx.x + blockDim.x * blockIdx.x);

    if (idx < N) {
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;

        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;

        FLOAT4(c[idx]) = reg_c;
    }
}

int main() {
    int N = 256 * 256;

    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c(N);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    size_t size = N * sizeof(float);

    int tpB = 256/4;
    int bpG = N/256;


}

