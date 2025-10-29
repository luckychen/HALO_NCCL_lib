#include <mpi.h>
#include <iostream>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

#include "halo_lib.hpp"

typedef float DataType;

// ============================================================================
// KERNEL FUNCTIONS (from kernel.cu)
// ============================================================================

// 3x3 Gaussian kernel (normalized)
__constant__ float gaussianKernel3x3[9] = {
    0.0625f, 0.125f, 0.0625f,
    0.125f,  0.25f,  0.125f,
    0.0625f, 0.125f, 0.0625f
};

// 5x5 Gaussian kernel (normalized)
__constant__ float gaussianKernel5x5[25] = {
    1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f,
    4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
    7.0f/273.0f, 26.0f/273.0f, 41.0f/273.0f, 26.0f/273.0f, 7.0f/273.0f,
    4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
    1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f
};

// GPU kernel for 3x3 Gaussian convolution
template <typename T>
__global__ void gaussianKernel3x3_kernel(const T* input, T* output,
                                         int innerWidth, int innerHeight,
                                         int haloWidth, int totalWidth)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < innerWidth && y < innerHeight) {
        // Map to position in localData (accounting for halo)
        int outX = x + haloWidth;
        int outY = y + haloWidth;

        T sum = 0;

        // Apply 3x3 kernel
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int inY = outY + ky;
                int inX = outX + kx;
                int kernelIdx = (ky + 1) * 3 + (kx + 1);
                sum += input[inY * totalWidth + inX] * gaussianKernel3x3[kernelIdx];
            }
        }

        output[outY * totalWidth + outX] = sum;
    }
}

// GPU kernel for 5x5 Gaussian convolution
template <typename T>
__global__ void gaussianKernel5x5_kernel(const T* input, T* output,
                                         int innerWidth, int innerHeight,
                                         int haloWidth, int totalWidth)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < innerHeight && col < innerWidth) {
        int outCol = col + haloWidth;
        int outRow = row + haloWidth;

        T sum = 0;

        // Apply 5x5 kernel
        for (int krow = -2; krow <= 2; krow++) {
            for (int kcol = -2; kcol <= 2; kcol++) {
                int inRow = outRow + krow;
                int inCol = outCol + kcol;
                int kernelIdx = (krow + 2) * 5 + (kcol + 2);
                sum += input[inRow * (innerWidth + 2*haloWidth) + inCol] * gaussianKernel5x5[kernelIdx];
            }
        }

        output[outRow * (innerWidth + 2*haloWidth) + outCol] = sum;
    }
}

// Host function to launch appropriate kernel
template <typename T>
void applyGaussianKernel(T* data,
		T* buffer,
		int innerWidth,
		int innerHeight,
		int haloWidth,
		int totalWidth,
		cudaStream_t stream)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((innerWidth + blockSize.x - 1) / blockSize.x,
                  (innerHeight + blockSize.y - 1) / blockSize.y);

    if (haloWidth == 1) {
        gaussianKernel3x3_kernel<<<gridSize, blockSize, 0, stream>>>(
            data, buffer, innerWidth, innerHeight, haloWidth, totalWidth);
    } else if (haloWidth == 2) {
        gaussianKernel5x5_kernel<<<gridSize, blockSize, 0, stream>>>(
            data, buffer, innerWidth, innerHeight, haloWidth, totalWidth);
    }

    // Copy result back (only inner region)
    cudaMemcpyAsync(data, buffer,
                    (innerWidth + 2*haloWidth) * (innerHeight + 2 * haloWidth) * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
	cudaStreamSynchronize(stream);
}

// CPU reference implementation
template <typename T>
void applyGaussianKernelCPU(T* data,
		T* buffer,
		int width,
		int height,
		int haloWidth)
{
    // Copy original data
    memcpy(buffer, data, width * height * sizeof(T));

    if (haloWidth == 1) {
        // 3x3 Gaussian
        float kernel[9] = {
            0.0625f, 0.125f, 0.0625f,
            0.125f,  0.25f,  0.125f,
            0.0625f, 0.125f, 0.0625f
        };

        for (int y = haloWidth; y < height - haloWidth; y++) {
            for (int x = haloWidth; x < width - haloWidth; x++) {
                T sum = 0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int kernelIdx = (ky + 1) * 3 + (kx + 1);
                        sum += buffer[(y + ky) * width + (x + kx)] * kernel[kernelIdx];
                    }
                }
                data[y * width + x] = sum;
            }
        }
    } else if (haloWidth == 2) {
        // 5x5 Gaussian
        float kernel[25] = {
            1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f,
            4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
            7.0f/273.0f, 26.0f/273.0f, 41.0f/273.0f, 26.0f/273.0f, 7.0f/273.0f,
            4.0f/273.0f, 16.0f/273.0f, 26.0f/273.0f, 16.0f/273.0f, 4.0f/273.0f,
            1.0f/273.0f,  4.0f/273.0f,  7.0f/273.0f,  4.0f/273.0f, 1.0f/273.0f
        };

        for (int y = 0 ; y < height; y++) {
            for (int x = 0; x < width; x++) {
                T sum = 0;
                for (int ky = -2; ky <= 2; ky++) {
                    for (int kx = -2; kx <= 2; kx++) {
                        int kernelIdx = (ky + 2) * 5 + (kx + 2);
						if((y+ky) >=0 && (y + ky) < height && (x+kx) >=0 && (x + kx) < width)
	                        sum += buffer[(y + ky) * width + (x + kx)] * kernel[kernelIdx];
                    }
                }
                data[y * width + x] = sum;
            }
        }
    }
}

// Initialize data with pattern
template <typename T>
void initializeData(T* data, int width, int height, int seed)
{
    srand(seed);
    for (int i = 0; i < width * height; i++) {
        data[i] = static_cast<T>(rand() % 256);
    }
}

// Compare results
template <typename T>
bool compareResults(const T* data1, const T* data2, int size, T tolerance)
{
    int errors = 0;
    T maxDiff = 0;

    for (int i = 0; i < size; i++) {
        T diff = std::abs(data1[i] - data2[i]);
        if (diff > tolerance) {
            errors++;
            if (errors < 100) { // Print first 100 errors
                printf("Mismatch at %d: %f vs %f (diff: %f)\n",
                       i, (float)data1[i], (float)data2[i], (float)diff);
            }
        }
        maxDiff = std::max(maxDiff, diff);
    }

    printf("Max difference: %f, Errors: %d / %d\n", (float)maxDiff, errors, size);
    return errors == 0;
}

int main(int argc, char* argv[])
{
    // Parse command line arguments
    if (argc < 6) {
        printf("Usage: %s <totalWidth> <totalHeight> <gridWidth> <gridHeight> <haloWidth> [numIterations]\n",
               argv[0]);
        return 1;
    }

    int totalWidth = atoi(argv[1]);
    int totalHeight = atoi(argv[2]);
    int gridWidth = atoi(argv[3]);
    int gridHeight = atoi(argv[4]);
    int haloWidth = atoi(argv[5]);
    int numIterations = (argc > 6) ? atoi(argv[6]) : 1;

    // Initialize MPI
    int rank, numRanks;
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

    // Validate parameters
    if (gridWidth * gridHeight != numRanks) {
        if (rank == 0) {
            printf("Error: gridWidth * gridHeight (%d * %d = %d) must equal number of ranks (%d)\n",
                   gridWidth, gridHeight, gridWidth * gridHeight, numRanks);
        }
        MPI_Finalize();
        return 1;
    }

    // Calculate inner dimensions
    int innerWidth = totalWidth / gridWidth;
    int innerHeight = totalHeight / gridHeight;

    if (rank == 0) {
        printf("Configuration:\n");
        printf("  Total size: %d x %d\n", totalWidth, totalHeight);
        printf("  Grid: %d x %d (%d ranks)\n", gridWidth, gridHeight, numRanks);
        printf("  Inner size per rank: %d x %d\n", innerWidth, innerHeight);
        printf("  Halo width: %d\n", haloWidth);
        printf("  Iterations: %d\n", numIterations);
    }

    // Set device for this rank
    int deviceId = getDeviceId(rank, numRanks);
    CUDACHECK(cudaSetDevice(deviceId));
    printf("Rank %d using GPU %d\n", rank, deviceId);

    // Initialize NCCL
    ncclUniqueId ncclId;
    if (rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&ncclId));
    }
    MPICHECK(MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t ncclComm;
    NCCLCHECK(ncclCommInitRank(&ncclComm, numRanks, ncclId, rank));

    // Create CUDA stream
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    // Allocate CPU buffers
    int localWidth = innerWidth + 2 * haloWidth;
    int localHeight = innerHeight + 2 * haloWidth;

    DataType* h_localData = nullptr;

    DataType* h_globalData = nullptr;
    DataType* h_gpuResult = nullptr;

    // Step 1: Initialize MPI descriptors
    descrMPI<DataType>* descrArray = nullptr;

    if (rank == 0) {
        // Rank 0: Create descriptor array and initialize all descriptors
        descrArray = new descrMPI<DataType>[numRanks];

        // Initialize all descriptors in the array
        initialMPIDscrArray<DataType>(descrArray, numRanks,
                                     totalWidth, totalHeight,
                                     innerWidth, innerHeight,
                                     gridWidth, gridHeight, haloWidth);

        // Initialize global data
        h_globalData = descrArray[0].h_globalData;
        initializeData<DataType>(h_globalData, totalWidth, totalHeight, 42);
        printf("Rank 0: Initialized global data (%d x %d)\n", totalWidth, totalHeight);

        h_gpuResult = new DataType[totalWidth * totalHeight];
        h_localData = descrArray[0].h_localData;
    } else {
        // Non-rank-0: Create single descriptor with full buffer allocation
        descrArray = new descrMPI<DataType>[1];
        descrArray[0] = descrMPI<DataType>(rank, numRanks, totalWidth, totalHeight,
                                          innerWidth, innerHeight,
                                          gridWidth, gridHeight, haloWidth,
                                          false);  // false = allocate buffers for non-rank-0
        h_localData = descrArray[0].h_localData;
    }

    // Step 2: Distribute initial data
    distributeInitialData<DataType>(descrArray, h_localData, MPI_COMM_WORLD);

    printf("Rank %d: Received initial data\n", rank);

    MPI_Barrier(MPI_COMM_WORLD);

    // Step 3: Initialize GPU exchange descriptor and copy data to GPU
    descrHalo<DataType> exchangeDescr(rank, innerWidth, innerHeight,
                                     gridWidth, gridHeight, haloWidth);

    printf("Rank %d: Copying data to GPU (%zu bytes)\n", rank,
           localWidth * localHeight * sizeof(DataType));

    CHECK_CUDA_MEMCPY(exchangeDescr.localData, h_localData,
                     localWidth * localHeight * sizeof(DataType),
                     cudaMemcpyHostToDevice);

    printf("Rank %d: Initialized exchange descriptor and copied data to GPU\n", rank);

    // Step 4: Prepare buffer for kernel execution
    DataType* cudaBuffer;
    allocateDeviceMemory<DataType>(&cudaBuffer,
                                   (innerWidth + 2 * haloWidth) * (innerHeight + 2 * haloWidth));
    CUDACHECK(cudaMemcpy(cudaBuffer, exchangeDescr.localData,
                        (innerWidth + 2 * haloWidth) * (innerHeight + 2 * haloWidth) * sizeof(DataType),
                        cudaMemcpyDeviceToDevice));

    // Step 5: Perform iterations
    for (int iter = 0; iter < numIterations; iter++) {
        // Apply convolution kernel
        applyGaussianKernel<DataType>(exchangeDescr.localData, cudaBuffer,
                                     innerWidth, innerHeight,
                                     haloWidth, localWidth, stream);

        CUDACHECK(cudaStreamSynchronize(stream));
        printf("Rank %d: Completed iteration %d convolution\n", rank, iter);

        // Exchange halo data (except for last iteration)
        if (iter < numIterations - 1) {
            exchangeHalo<DataType>(exchangeDescr, ncclComm, stream);
            CUDACHECK(cudaStreamSynchronize(stream));
            printf("Rank %d: Completed iteration %d halo exchange\n", rank, iter);
        }
    }

    CUDACHECK(cudaFree(cudaBuffer));

    

    // Step 6: Copy results back to host
    CUDACHECK(cudaMemcpy(h_localData, exchangeDescr.localData,
                        localWidth * localHeight * sizeof(DataType),
                        cudaMemcpyDeviceToHost));

    // Step 7: Gather GPU results to rank 0
    if (rank == 0) {
        gatherResults<DataType>(descrArray, descrArray[0].h_localData, h_gpuResult, MPI_COMM_WORLD);
    } else {
        gatherResults<DataType>(descrArray, h_localData, nullptr, MPI_COMM_WORLD);
    }

	// Step 8: CPU reference implementation (rank 0 only)
    if (rank == 0) {
        DataType* cpuBuffer = new DataType[totalHeight * totalWidth];
        printf("Rank 0: Running CPU reference...\n");
        for (int iter = 0; iter < numIterations; iter++) {
            applyGaussianKernelCPU<DataType>(h_globalData, cpuBuffer, totalWidth, totalHeight, haloWidth);
            printf("Rank 0: Completed CPU iteration %d\n", iter);
        }
        delete[] cpuBuffer;
    }

    // Step 9: Compare results (rank 0 only)
    if (rank == 0) {
        printf("\n=== Comparing Results ===\n");

        bool match = compareResults<DataType>(h_globalData, h_gpuResult,
                                            totalWidth * totalHeight,
                                            static_cast<DataType>(0.01));

        if (match) {
            printf("\n*** TEST PASSED: Results match! ***\n");
        } else {
            printf("\n*** TEST FAILED: Results do not match! ***\n");
        }
    }

    // Cleanup
    // Delete h_gpuResult (allocated separately, only on rank 0)
    if (rank == 0 && h_gpuResult) {
        delete[] h_gpuResult;
        h_gpuResult = nullptr;
    }

    // Delete descriptor array (this also deletes all buffers managed by descriptors)
    // Note: h_globalData and h_localData are managed by descriptors, don't delete them separately
    if (descrArray) {
        delete[] descrArray;
        descrArray = nullptr;
    }

    CUDACHECK(cudaStreamDestroy(stream));
    NCCLCHECK(ncclCommDestroy(ncclComm));
    MPICHECK(MPI_Finalize());

    if (rank == 0) {
        printf("\nTest completed successfully!\n");
    }

    return 0;
}
