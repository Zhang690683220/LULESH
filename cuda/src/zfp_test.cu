#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include "zfp.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

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

int zfp_compress(double* array, size_t nx, size_t ny, size_t nz) {
    int status = 0;
    double rate = 8.0;
    zfp_type type;     /* array scalar type */
    zfp_field* field;  /* array meta data */
    zfp_stream* zfp;   /* compressed stream */
    void* buffer;      /* storage for compressed stream */
    size_t bufsize;    /* byte size of compressed buffer */
    bitstream* stream; /* bit stream to write to or read from */
    size_t zfpsize;    /* byte size of compressed stream */

    /* allocate meta data for the 3D array a[nz][ny][nx] */
    type = zfp_type_double;
    field = zfp_field_3d(array, type, nx, ny, nz);

    /* allocate meta data for a compressed stream */
    zfp = zfp_stream_open(NULL);

    zfp_stream_set_rate(zfp, rate, type, 3, false);

    /* allocate buffer for compressed data */
    bufsize = zfp_stream_maximum_size(zfp, field);
    cudaMalloc(&buffer, bufsize);

    /* associate bit stream with allocated buffer */
    stream = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, stream);
    zfp_stream_rewind(zfp);

    if (zfp_stream_set_execution(zfp, zfp_exec_cuda)) {
        zfpsize = zfp_compress(zfp, field);
        if (!zfpsize) {
            fprintf(stderr, "compression failed\n");
            status = -1;
        }
    }

    /* clean up */
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(stream);
    cudaFree(buffer);

    return status;
}

int main(int argc, char** argv)
{
    int nprocs, rank;
    MPI_Comm gcomm;
    // Using SPMD style programming
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);

    gcomm = MPI_COMM_WORLD;

    int color = 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &gcomm);

    uint64_t nx = 512;
    uint64_t ny = 512;
    uint64_t nz = 512;

    uint64_t grid_size = nx * ny * nz;

    int dev_num, dev_rank;
    cudaError_t cuda_status;
    cuda_status = cudaGetDeviceCount(&dev_num);
    if(cuda_status != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaGetDeviceCount() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(cuda_status));
        return -1;
    }
    dev_rank = rank%dev_num;
    cuda_status = cudaSetDevice(dev_rank);
    if(cuda_status != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaSetDevice() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(cuda_status));
        return -1;
    }

    double *data;
    cudaMalloc((void**) &data, sizeof(double)*grid_size);

    cuda_assign_double(dev_rank, data, grid_size, 1);

    cuda_status = cudaDeviceSynchronize();
    if(cuda_status != cudaSuccess) {
        fprintf(stderr, "ERROR: (%s): cudaDeviceSynchronize() failed, Err Code: (%s)\n", __func__, cudaGetErrorString(cuda_status));
        return -1;
    }

    zfp_compress(data, nx, ny, nz);

    cudaFree(data);
    
    MPI_Finalize();

    return 0;
}