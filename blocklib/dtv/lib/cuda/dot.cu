__global__ void dot8(float* a, float* b, float* c)
{
    static const int N = 8;
    __shared__ float temp[N];
    temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
    __syncthreads();
    
    if (threadIdx.x == 0) {
        float sum = 0;
        for (int i = 0; i < N; i++)
            sum += temp[i];
        *c = sum;
    }
}
