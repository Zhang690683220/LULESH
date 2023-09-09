#include <stdint.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "zfp.h"
#include "dspaces.h"
#include "timer.hpp"
#include "CLI11.hpp"

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
    CLI::App app{"ZFP for LULESH"};

    std::vector<int> np;
    std::vector<uint64_t> sp;
    int ndims = 3;
    int input_step = 20;
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
        //print_usage();
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

    double *d_energy, *d_pressure, *d_mass;
    cudaMalloc((void**) &d_energy, sizeof(double)*grid_size);
    cudaMalloc((void**) &d_pressure, sizeof(double)*grid_size);
    cudaMalloc((void**) &d_mass, sizeof(double)*grid_size);

    std::ofstream log;
    double* avg_get = nullptr;
    double total_avg = 0;

    int input_count = 0;

    if(rank == 0) {
        avg_get = (double*) malloc(sizeof(double)*input_step);
        for(int i=0; i<input_step; i++) {
            avg_get[i] = 0.0;
        }
        log.open("zfp_dspaces.log", std::ofstream::out | std::ofstream::trunc);
        log << "step,get_ms" << std::endl;
    }

    for(int its=1; its<interval*input_step+1; its++) {
        if(its%interval == 0) {
            unsigned int dspaces_iter = its / interval;
            double time_copy, time_transfer;
            Timer timer_get;
            timer_get.start();
            dspaces_cuda_get(dspaces_client, "energy", its, sizeof(double), 3, lb, ub, d_energy, -1,
                             &time_transfer, &time_copy);
            dspaces_cuda_get(dspaces_client, "pressure", its, sizeof(double), 3, lb, ub, d_pressure, -1,
                             &time_transfer, &time_copy);
            dspaces_cuda_get(dspaces_client, "mass", its, sizeof(double), 3, lb, ub, d_mass, -1,
                             &time_transfer, &time_copy);
            double time_get = timer_get.stop();
            input_count ++;

            zfp_compress(d_energy, sp[0], sp[1], sp[2]);

            double *avg_time_get = nullptr;
            if(rank == 0) {
                avg_time_get = (double*) malloc(sizeof(double)*nprocs);
            }
            MPI_Gather(&time_get, 1, MPI_DOUBLE, avg_time_get, 1, MPI_DOUBLE, 0, gcomm);

            if(rank == 0) {
                for(int i=0; i<nprocs; i++) {
                    avg_get[input_count-1] += avg_time_get[i];
                }
                avg_get[input_count-1] /= nprocs;
                log << its << "," << avg_get[input_count-1] << std::endl;
                total_avg += avg_get[input_count-1];
                free(avg_time_get);
            }
            
        }
    }

    cudaFree(d_energy);
    cudaFree(d_pressure);
    cudaFree(d_mass);

    if(rank == 0) {
        total_avg /= input_step;
        log << "Average" << "," << total_avg << std::endl;
        log.close();
        std::cout<<"Writer sending kill signal to server."<<std::endl;
        dspaces_kill(dspaces_client);
    }

    dspaces_fini(dspaces_client);

    MPI_Finalize();

    return 0;
}