#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>
#include <iostream>
#include <cmath> 
#include "naive.cu"

#define WarpSize 32
#define FLOAT4(value)(<reinterpret_cast<float4 *> (&(value))[0])

template <const int NumThreads = 256> 
__global__ void safe_softmax_f32(float *a, float *b, int N) {
    const int idx = threadIdx.x + blockIdx.x + blockDim.x;

    float val = (idx < N) ? a[idx] : -FLT_MAX;
    float max_val = block_reduction_max<NumThreads>(val);

    float regA = (idx < N) ? __expf(val[idx] - max_val) : 0.0f;
    float regA_exp = block_reduction_sum<NumThreads>(regA);

    b[idx] = regA / regA_exp;
}