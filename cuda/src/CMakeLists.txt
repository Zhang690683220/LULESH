find_package(MPI)
if(MPI_FOUND)
    add_definitions(-DUSE_MPI)
endif()

find_package(ZFP REQUIRED)

add_executable(lulesh allocator.cu lulesh.cu lulesh-comms.cu lulesh-comms-gpu.cu)
target_include_directories(lulesh PRIVATE ${INCLUDE_PATH})
target_link_libraries(lulesh PUBLIC MPI::MPI_CXX DataSpaces::DataSpaces)

add_executable(zfp_cuda_test zfp_test.cpp generate_value.cu)
set_target_properties(zfp_cuda_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(zfp_cuda_test PRIVATE ${INCLUDE_PATH})
target_link_libraries(zfp_cuda_test PUBLIC MPI::MPI_CXX ZFP::zfp)

add_executable(zfp_cuda zfp.cpp)
target_include_directories(zfp_cuda PRIVATE ${INCLUDE_PATH})
target_link_libraries(zfp_cuda PUBLIC MPI::MPI_CXX DataSpaces::DataSpaces ZFP::zfp)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    target_include_directories(lulesh PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(lulesh PUBLIC CUDA::cudart CUDA::cuda_driver)

    target_include_directories(zfp_cuda_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(zfp_cuda_test PUBLIC CUDA::cudart CUDA::cuda_driver)

    target_include_directories(zfp_cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(zfp_cuda PUBLIC CUDA::cudart CUDA::cuda_driver)
else()
    target_include_directories(lulesh PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(lulesh PUBLIC ${CUDA_LIBRARIES})

    target_include_directories(zfp_cuda_test PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(zfp_cuda_test PUBLIC ${CUDA_LIBRARIES})

    target_include_directories(zfp_cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(zfp_cuda PUBLIC CUDA::cudart CUDA::cuda_driver)
endif()