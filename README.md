# HALO Pattern GPU Data Exchange Library

A **header-only C++ library** for efficient halo-pattern data exchange between GPUs in a distributed computing environment using **MPI** and **NCCL**.

## Overview

This library provides a complete framework for:
1. **CPU-side MPI data distribution** - Initialize and distribute data across ranks
2. **GPU-side halo exchange** - Efficient neighbor communication using NCCL

Designed for stencil-based computations, convolutional operations, and other applications requiring periodic boundary communication in distributed GPU clusters.

## Features

- **Header-only library** - Easy integration, no separate compilation required
- **Template-based design** - Generic implementation for any numeric data type
- **Move semantics** - Proper C++11 memory ownership handling
- **2D grid topology** - Flexible rank arrangement in 2D grids
- **Comprehensive error checking** - MPI, CUDA, and NCCL error macros
- **Production-ready** - Tested on 4-GPU PowerPC configuration

## Architecture

### Part 1: MPI Data Distribution

The `descrMPI<T>` class manages CPU host memory for each rank:

```cpp
descrMPI<float> descriptor(rank, numRanks, totalWidth, totalHeight,
                          innerWidth, innerHeight,
                          gridWidth, gridHeight, haloWidth);
```

**Rank 0 (index 0) allocates:**
- `h_globalData`: Full domain
- `h_sendBuffer`: MPI send buffer
- `h_localData`: Local domain with halo padding

**Non-rank-0 descriptors allocate:**
- `h_sendBuffer`: For sending results back to rank 0
- `h_recvBuffer`: For receiving initial data from rank 0
- `h_localData`: Local domain with halo padding

### Part 2: GPU Halo Exchange

The `descrHalo<T>` class manages GPU device memory:

```cpp
descrHalo<float> gpuDescr(rank, innerWidth, innerHeight,
                         gridWidth, gridHeight, haloWidth);
```

**Features:**
- Automatic neighbor rank computation based on grid topology
- Send/receive buffers for 4 cardinal + 4 diagonal directions
- NCCL-based all-to-all neighbor communication

## Usage Example

```cpp
#include <mpi.h>
#include "halo_lib.hpp"

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // Configuration
    int totalWidth = 1024, totalHeight = 1024;
    int gridWidth = 2, gridHeight = 2;  // 2x2 grid
    int haloWidth = 2;
    int innerWidth = totalWidth / gridWidth;
    int innerHeight = totalHeight / gridHeight;

    // Step 1: Initialize MPI descriptors
    descrMPI<float>* descrArray = nullptr;
    if (rank == 0) {
        descrArray = new descrMPI<float>[numRanks];
        initialMPIDscrArray<float>(descrArray, numRanks,
                                  totalWidth, totalHeight,
                                  innerWidth, innerHeight,
                                  gridWidth, gridHeight, haloWidth);
        // Initialize global data...
    } else {
        descrArray = new descrMPI<float>[1];
        descrArray[0] = descrMPI<float>(rank, numRanks, totalWidth, totalHeight,
                                      innerWidth, innerHeight,
                                      gridWidth, gridHeight, haloWidth, false);
    }

    // Step 2: Distribute initial data
    distributeInitialData<float>(descrArray, h_localData, MPI_COMM_WORLD);

    // Step 3: GPU setup
    descrHalo<float> gpuDescr(rank, innerWidth, innerHeight,
                             gridWidth, gridHeight, haloWidth);

    cudaMemcpy(gpuDescr.localData, h_localData,
              (innerWidth + 2*haloWidth) * (innerHeight + 2*haloWidth) * sizeof(float),
              cudaMemcpyHostToDevice);

    // Step 4: GPU computation and halo exchange
    for (int iter = 0; iter < numIterations; iter++) {
        // Your computation kernel here...
        applyKernel<<<...>>>(gpuDescr.localData, ...);

        // Exchange halo data
        if (iter < numIterations - 1) {
            exchangeHalo<float>(gpuDescr, ncclComm, stream);
        }
    }

    // Step 5: Gather results
    cudaMemcpy(h_localData, gpuDescr.localData, ..., cudaMemcpyDeviceToHost);
    gatherResults<float>(descrArray, h_localData, h_globalData, MPI_COMM_WORLD);

    // Cleanup
    delete[] descrArray;
    MPI_Finalize();
    return 0;
}
```

## Example Application

The `main.cu` file in this project demonstrates how to utilize the HALO library with a convolutional kernel function. The example compares the results between CPU and GPU implementations to validate the correctness of the distributed GPU computation:

- **CPU Implementation**: Serial computation for reference and verification
- **GPU Implementation**: Parallel computation with halo exchange using NCCL
- **Comparison**: Results from both implementations are compared to ensure numerical accuracy

This practical example showcases the complete workflow: initializing the halo pattern infrastructure, performing distributed GPU computations with a convolutional kernel, and verifying results against a CPU baseline.

## Building

### Requirements
- CUDA Toolkit (7.0 or higher)
- MPI implementation (OpenMPI, MPICH, etc.)
- NCCL library
- CMake 3.18+

### Configuration

Edit `CMakeLists.txt`:
```cmake
set(CONDA_ROOT "/path/to/conda/env")
set(CUDA_ARCH "70")  # Adjust for your GPU
```

### Build
```bash
mkdir build && cd build
cmake ..
make
```

## Testing

Run the included test with 4 ranks on a 2×2 GPU grid:

```bash
mpirun -n 4 ./halo_test 1024 1024 2 2 2 3
```

### Command Arguments

The `mpirun` command executes the test program with the following parameters:

- **`-n 4`**: Run 4 GPU processes (one per GPU in a 2×2 grid)
- **`1024 1024`**: Original data size (1024 × 1024 pixels)
- **`2 2`**: 2×2 GPU grid topology (each GPU handles a 512×512 local domain)
- **`2`**: Halo width = 2 (represents a 5×5 convolution kernel: center + 2 pixels in each direction)
- **`3`**: Number of iterations (convolution operations performed 3 times sequentially, with output of one iteration becoming input to the next)

### Computation Details

The test performs the same **5×5 convolutional operation** on the 1024×1024 data across two different compute paths:

1. **Distributed GPU Computation**:
   - Data is decomposed across 4 GPUs in a 2×2 grid
   - Each GPU computes its local 512×512 domain plus halo regions
   - Halo data is exchanged between neighboring GPUs using NCCL
   - 3 sequential iterations of convolution with result feedback

2. **Serial CPU Computation**:
   - Same convolution operation executed serially on CPU
   - Results from one iteration feed into the next

3. **Verification**: Results from both distributed GPU and serial CPU computations are compared to ensure numerical accuracy and correctness of the halo exchange mechanism

Expected output:
```
Configuration:
  Total size: 1024 x 1024
  Grid: 2 x 2 (4 ranks)
  Inner size per rank: 512 x 512
  Halo width: 2
  Iterations: 3

...computation output...

=== Comparing Results ===
*** TEST PASSED: Results match! ***

Test completed successfully!
```

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Project overview, architecture, and design decisions
- **[BUG_FIX.md](BUG_FIX.md)** - Detailed description of bugs found and fixes applied

## Key Classes and Functions

### `descrMPI<T>`
MPI descriptor for CPU host memory management
- **Constructor variants** for different initialization patterns
- **Move semantics** for safe ownership transfer
- **Automatic memory cleanup** in destructor

### `descrHalo<T>`
GPU halo descriptor for device memory management
- Neighbor rank computation
- Send/receive buffers for 8 directions
- NCCL communication setup

### MPI Functions
- `initialMPIDscrArray<T>()` - Initialize descriptor array for rank 0
- `distributeInitialData<T>()` - MPI distribution from rank 0 to all ranks
- `gatherResults<T>()` - MPI gathering from all ranks to rank 0

### GPU Functions
- `exchangeHalo<T>()` - Perform NCCL-based halo exchange
- `allocateDeviceMemory<T>()` - Wrapper for CUDA memory allocation

## Platform Support

Currently compiled for and tested on:
- **PowerPC (ppc64le)** with V100 GPUs
- CUDA 11.x
- OpenMPI 4.x

For other platforms, adjust CMakeLists.txt accordingly.

## Known Limitations

- 2D grid topology only (3D grids would require extension)
- Equal domain decomposition across all ranks required
- Halo width must be consistent across all ranks
- Assumes MPI_Init has been called by the user

## Bug Fixes

This project includes fixes for three critical issues:

1. **Missing send buffer in non-rank-0 descriptors** - Now properly allocated
2. **Missing move semantics** - Added move constructor and assignment operator
3. **Double deletion in cleanup** - Fixed duplicate deletions

See [BUG_FIX.md](BUG_FIX.md) for detailed information.

## Performance Notes

- MPI communication happens once per iteration
- NCCL uses grouped send/receive for efficiency
- GPU memory layout optimized for 2D stencil operations
- Halo data preserved between iterations

## Future Improvements

- [ ] Support for 3D grid topology
- [ ] Overlap computation and communication
- [ ] CUDA graph support
- [ ] Automatic load balancing
- [ ] Support for irregular grids

## License

This project is provided as-is for research and educational purposes.

## References

- NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/
- MPI Standard: https://www.mpi-forum.org/
- Halo Exchange Pattern: https://en.wikipedia.org/wiki/Halo_exchange

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and create a pull request

---

**Author**: Generated with Claude Code
**Date**: 2024
**Status**: Production-ready (tested on PowerPC 4-GPU configuration)
