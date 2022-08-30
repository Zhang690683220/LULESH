find_package(MPI)
if(MPI_FOUND)
    add_definitions(-DUSE_MPI)
endif()

add_executable(lulesh allocator.cu lulesh.cu lulesh-comms.cu lulesh-comms-gpu.cu)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.17)
    target_include_directories(lulesh PRIVATE ${INCLUDE_PATH} ${MPI_C_INCLUDE_PATH}
                                ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(lulesh PUBLIC MPI::MPI_CXX CUDA::cudart CUDA::cuda_driver)
else()
    target_include_directories(lulesh PRIVATE ${INCLUDE_PATH} ${MPI_C_INCLUDE_PATH}
                                ${CUDA_INCLUDE_DIRS})
    target_link_libraries(lulesh PUBLIC MPI::MPI_CXX ${CUDA_LIBRARIES})
endif()