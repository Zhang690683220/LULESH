__global__ void assign(double *ptr, int size, int var_idx)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx<size) {
        ptr[idx] = idx + 0.01*var_idx;
    }
}

cudaError_t cuda_assign_double(int dev_rank, double *ptr, int size, int var_idx)
{
    cudaError_t cuda_status;
    cudaDeviceProp dev_prop;
    cuda_status = cudaGetDeviceProperties(&dev_prop,dev_rank);

    int threadsPerBlock = dev_prop.maxThreadsPerBlock;
    int numBlocks = (size + threadsPerBlock) / threadsPerBlock;

    assign<<<numBlocks, threadsPerBlock>>>(ptr, size, var_idx);

    return cuda_status;
}