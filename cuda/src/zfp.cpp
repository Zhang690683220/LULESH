#include <stdint.h>
#include <stdlib.h>
#include "zfp.h"
#include "dspaces.h"

int zfp_compress(double* array, size_t nx, size_t ny, size_t nz) {
    int status = 0;
    double rate = 8.0
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

    if (zfp_stream_set_execution(stream, zfp_exec_cuda)) {
        zfpsize = zfp_compress(stream, field);
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
    CLI::App app{"ZFP for LULESH"};

    std::vector<int> np;
    std::vector<uint64_t> sp;
    int ndims = 3;
    int interval = 1;
    bool terminate = false;
    app.add_option("--np", np, "the number of processes in the ith dimension. The product of np[0],"
                    "...,np[dim-1] must be the number of MPI ranks")->expected(1, 8);
    app.add_option("--sp", sp, "the per-process data size in the ith dimension")->expected(1, 8);
    app.add_option("--interval", interval, "Output timestep interval. Default to 1", true);
    app.add_flag("-k", terminate, "send server kill signal after reading is complete");

    CLI11_PARSE(app, argc, argv);

    int npapp = 1;             // number of application processes
    for(int i = 0; i < ndims; i++) {
        npapp *= np[i];
    }

    int nprocs, rank, dspaces_rank;
    MPI_Comm gcomm;
    // Using SPMD style programming
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Barrier(MPI_COMM_WORLD);

    gcomm = MPI_COMM_WORLD;

    int color = 1;
    MPI_Comm_split(MPI_COMM_WORLD, color, rank, &gcomm);

    if(npapp != nprocs) {
        std::cerr<<"Product of np[i] args must equal number of MPI processes!"<<std::endl;
        print_usage();
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm_rank(gcomm, &dspaces_rank);
    dspaces_client_t dspaces_client = dspaces_CLIENT_NULL;
    dspaces_init(dspaces_rank, &dspaces_client);

    uint64_t grid_size = 1;
    for(int i=0; i<ndims; i++) {
        grid_size *= sp[i];
    }

    uint64_t lb[3] = {0};
    uint64_t ub[3] = {0};
    uint64_t off[3] = {0};

    // get the lb & ub for each rank
    for(int i=0; i<ndims; i++) {
        int tmp = rank;
        for(int j=0; j<i; j++)
            tmp /= np[j];
        off[i] = tmp % np[i] * sp[i];
        lb[i] = off[i];
        ub[i] = off[i] + sp[i] - 1;
    }

    double *d_energy, *d_pressure, *d_mass
    cudaMalloc((void**) &d_energy, sizeof(double)*grid_size);
    cudaMalloc((void**) &d_pressure, sizeof(double)*grid_size);
    cudaMalloc((void**) &d_mass, sizeof(double)*grid_size);

    for(int its=1; its<interval*10+1; its++) {
        if(its%interval == 0) {
            double time_copy, time_transfer;
            dspaces_cuda_get(ndcl, "energy", its, sizeof(double), 3, lb, ub, d_energy, -1,
                             &time_transfer, &time_copy);
            dspaces_cuda_get(ndcl, "pressure", its, sizeof(double), 3, lb, ub, d_pressure, -1,
                             &time_transfer, &time_copy);
            dspaces_cuda_get(ndcl, "mass", its, sizeof(double), 3, lb, ub, d_mass, -1,
                             &time_transfer, &time_copy);

            zfp_compress(d_energy, sp[0], sp[1], sp[2]);
            
        }
    }

    cudaFree(d_energy);
    cudaFree(d_pressure);
    cudaFree(d_mass);

    free(off);
    free(lb);
    free(ub);

    dspaces_fini(ndcl);
    
    MPI_Finalize();

    return 0;
}