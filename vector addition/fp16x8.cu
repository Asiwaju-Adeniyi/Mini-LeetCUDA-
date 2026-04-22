#include <algorithm> 
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h> 
#include <vector>

#define HALF2(value) (reinterpret_cast<half2 *> (&value)[0])

__global__ void elemwise_fp16x8(half *a, half *b, half *c, int N) {
    int tid = 8 * (threadIdx.x + blockIdx.x * blockDim.x);

    half2 reg_a0 = HALF2(a[idx]);
    half2 reg_a1 = HALF2(a[idx + 2]);
    half2 reg_a2 = HALF2(a[idx + 4]);
    half2 reg_a3 = HALF2(a[idx + 6]);

    half2 reg_b0 = HALF2(b[idx]);
    half2 reg_b1 = HALF2(b[idx + 2]);
    half2 reg_b2 = HALF2(b[idx + 4]);
    half2 reg_b3 = HALF2(b[idx + 6]);

    half2 reg_c0, reg_c1, reg_c2, reg_c3;

    reg_c0.x = __halfadd(reg_a0.x, reg_b0.x);
    reg_c0.y = __halfadd(reg_a0.y, reg_b0.y);
    reg_c1.x = __halfadd(reg_a1.x, reg_b1.x);
    reg_c1.y = __halfadd(reg_a1.y, reg_b1.y);
    reg_c2.x = __halfadd(reg_a2.x, reg_b2.x);
    reg_c2.y = __halfadd(reg_a2.y, reg_b2.y);
    reg_c3.x = __halfadd(reg_a3.x, reg_b3.x);
    reg_c3.y = __halfadd(reg_a3.y, reg_b3.y);

    
   if (idx < N) {
    HALF2 (c[idx]) = reg_c0;
   }

   if ((idx + 2) < N) {
    HALF2 (c[idx + 2]) = reg_c1;
   }

   if ((idx + 4) < N) {
    HALF2 (c[idx + 4]) = reg_c2;
   }

   if ((idx + 6) < N) {
    HALF2 (c[idx + 6]) = reg_c3;
   }


}