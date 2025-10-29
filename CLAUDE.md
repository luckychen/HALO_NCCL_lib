# HALO Pattern Data Transfer Library - Header-Only Implementation

## Project Overview

This project is a **header-only library** (`halo_lib.hpp`) that implements a Halo-pattern data exchange framework for distributed GPU computing using **MPI** and **NCCL**. The library enables efficient data transfer between GPUs arranged in a 2D grid topology, with applications to scientific computing, stencil operations, and convolutional processing.

The project is derived from a successfully tested implementation in the parent directory (`../`) and has been refactored into a reusable header-only library.

## Purpose

The library serves two main functions:

1. **CPU-side MPI Data Distribution**: Initialize data on rank 0 and distribute it to all other ranks using MPI
2. **GPU-side NCCL Halo Exchange**: Perform efficient halo-pattern data exchange between GPUs using NCCL for overlapping boundary regions

## Architecture

### Part 1: CPU Initialization and MPI Data Distribution

The library handles CPU-side operations to prepare data for GPU processing:

- **`descrMPI<T>` class**: Template class representing MPI descriptor for CPU host memory management
  - **Rank 0 (index 0)**: Allocates three buffers:
    - `h_globalData`: Full global data array (distributed to all ranks)
    - `h_sendBuffer`: Continuous memory buffer for sending data to specific ranks
    - `h_localData`: Local data segment for rank 0 (with halo padding)

  - **Non-rank-0 descriptors**: Boundary information only (no buffer allocation for other indices in rank 0's array)

  - **Other ranks**: Allocate:
    - `h_recvBuffer`: Continuous memory buffer for receiving data from rank 0
    - `h_localData`: Local data segment with halo padding

- **`initialMPIDscrArray<T>()`**: Creates descriptor array for all ranks (called by rank 0)
  - Index 0: Full descriptor with buffer allocation
  - Other indices: Boundary information descriptors

- **`distributeInitialData<T>()`**: Distributes initial data from rank 0 to all ranks
  - Rank 0 copies its own data and sends slices to other ranks via MPI

- **`gatherResults<T>()`**: Gathers computed results from all ranks back to rank 0
  - Non-rank-0 ranks send their inner region data to rank 0
  - Rank 0 reconstructs the global result array

### Part 2: GPU Data Exchange (NCCL-based Halo Exchange)

The library handles GPU-side halo exchange operations:

- **`descrHalo<T>` class**: Template class for GPU halo exchange descriptor
  - Allocates GPU device memory for:
    - `localData`: Local data with halo padding
    - Send/receive buffers for 4 cardinal directions (left, right, up, down)
    - Send/receive buffers for 4 diagonal directions (corners)

  - Computes neighbor ranks based on 2D grid topology

- **`exchangeHalo<T>()`**: Performs halo exchange using NCCL
  - Prepares send buffers by copying from localData
  - Uses NCCL for all-to-all neighbor communication
  - Updates localData with received data

- **GPU Kernels**: Gaussian convolution kernels (3×3 and 5×5) for testing
  - Applied to the inner region while preserving halo data

## Data Flow

```
Global Data (Rank 0)
    ↓
distributeInitialData() → CPU MPI send/receive
    ↓
Each Rank: h_localData (CPU) → GPU exchangeDescr.localData
    ↓
GPU Computation: applyGaussianKernel() + exchangeHalo()
    ↓
Result: GPU exchangeDescr.localData → h_localData (CPU)
    ↓
gatherResults() → CPU MPI send/receive → h_gpuResult (Rank 0)
```

## Key Components

### Files

- **`halo_lib.hpp`**: Header-only library containing all MPI and GPU halo exchange logic
- **`main.cu`**: Test application demonstrating library usage with Gaussian convolution
- **`CMakeLists.txt`**: Build configuration for PowerPC platform with CUDA, MPI, and NCCL

### Memory Layout

Each rank manages local data with halo padding:
- **Local width**: `innerWidth + 2 * haloWidth`
- **Local height**: `innerHeight + 2 * haloWidth`
- **Boundary regions**: Automatically calculated based on rank position in grid

### Configuration Parameters

- `totalWidth`, `totalHeight`: Global domain size
- `gridWidth`, `gridHeight`: Number of ranks in each dimension
- `innerWidth`, `innerHeight`: Size per rank (calculated as total / grid dimensions)
- `haloWidth`: Boundary padding width for neighbor communication

## Usage Pattern

### For Rank 0:

```cpp
// Create descriptor array
descrMPI<T>* descrArray = new descrMPI<T>[numRanks];

// Initialize all descriptors (rank 0 handles this)
initialMPIDscrArray<T>(descrArray, numRanks,
                       totalWidth, totalHeight,
                       innerWidth, innerHeight,
                       gridWidth, gridHeight, haloWidth);

// Initialize global data
initializeData(descrArray[0].h_globalData, ...);

// Distribute to all ranks
distributeInitialData<T>(descrArray, descrArray[0].h_localData, MPI_COMM_WORLD);

// GPU processing (copy to GPU, compute, halo exchange)
// ...

// Gather results
gatherResults<T>(descrArray, descrArray[0].h_localData, h_gpuResult, MPI_COMM_WORLD);
```

### For Non-Rank-0 Ranks:

```cpp
// Create single descriptor with buffer allocation
descrArray = new descrMPI<T>[1];
descrArray[0] = descrMPI<T>(rank, numRanks, totalWidth, totalHeight,
                           innerWidth, innerHeight,
                           gridWidth, gridHeight, haloWidth,
                           false);  // false = allocate buffers

// Receive initial data
distributeInitialData<T>(descrArray, descrArray[0].h_localData, MPI_COMM_WORLD);

// GPU processing...

// Send results to rank 0
gatherResults<T>(descrArray, descrArray[0].h_localData, nullptr, MPI_COMM_WORLD);
```

## Compilation

The project compiles for **PowerPC (ppc)** platform using:
- Cross-compiled CUDA toolkit
- MPI from conda environment
- NCCL for GPU communication

Build command:
```bash
cd build
cmake ..
make
```

## Testing

The included `main.cu` tests the library with:
- **4-GPU configuration** (2×2 grid)
- **Gaussian convolution** (3×3 or 5×5 kernel)
- **Multiple iterations** with halo exchange between iterations
- **CPU reference implementation** for result verification

Test execution (on PowerPC platform):
```bash
mpirun --mca psec ^munge -n 4 --oversubscribe ./halo_test 1024 1024 2 2 2 3
```
Arguments: `<totalWidth> <totalHeight> <gridWidth> <gridHeight> <haloWidth> [numIterations]`

## Key Design Decisions

1. **Header-Only Library**: Allows users to include the library without separate compilation
2. **Template-Based**: Generic implementation supports any numeric data type
3. **MPI-First Initialization**: MPI must be initialized by the calling code
4. **Descriptor Pattern**: Separate descriptors for MPI (CPU) and Halo (GPU) operations for clarity
5. **NCCL Communication**: Grouped send/receive operations for efficiency
6. **Gaussian Kernels**: Included as test case to demonstrate GPU computation integration

## Error Handling

The library provides macros for error checking:
- `MPICHECK()`: MPI error checking
- `CUDACHECK()`: CUDA error checking
- `NCCLCHECK()`: NCCL error checking
- `CHECK_CUDA_MEMCPY()`: Enhanced memcpy validation

## Limitations

- Compiled for PowerPC platform only (different cross-compilation required for other platforms)
- Assumes equal domain decomposition across ranks
- 2D grid topology only
- Halo width must be consistent across all ranks
