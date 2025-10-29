/*---------------------------
HALO Pattern Data Transfer Library (Header-Only)

This is a header-only library for Halo-pattern data exchange
between GPUs using NCCL and MPI.

Two main components:
1. CPU initialization and data distribution (MPI-based)
2. GPU data exchange (NCCL-based)

The library assumes MPI_Init has been called by the user.
-----------------------*/

#ifndef HALO_LIB_HPP
#define HALO_LIB_HPP

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <functional>

// ============================================================================
// ERROR CHECKING MACROS
// ============================================================================

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);                      \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(call)                                                    \
    do {                                                                    \
        ncclResult_t res = call;                                           \
        if (res != ncclSuccess) {                                          \
            fprintf(stderr, "[%s:%d] NCCL error %s\n", __FILE__, __LINE__,\
                    ncclGetErrorString(res));                              \
            MPI_Abort(MPI_COMM_WORLD, -1);                                 \
        }                                                                   \
    } while (0)

#define CHECK_CUDA_MEMCPY(dst, src, bytes, kind)                     \
    do {                                                            \
        if ((dst) == nullptr) {                                     \
            fprintf(stderr, "cudaMemcpy error: dst is NULL\n");      \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
        if ((src) == nullptr) {                                     \
            fprintf(stderr, "cudaMemcpy error: src is NULL\n");      \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
        if ((bytes) == 0) {                                         \
            fprintf(stderr, "cudaMemcpy error: byte count is 0\n"); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
        CUDACHECK( cudaMemcpy((dst), (src), (bytes), (kind)) );      \
} while (0)

// ============================================================================
// PART 1: CPU INITIALIZATION AND MPI DATA DISTRIBUTION
// ============================================================================

// Helper function to compute 2D rank position
inline void getRankPosition(int rank, int gridWidth, int gridHeight, int& row, int& col) {
    row = rank / gridWidth;
    col = rank % gridWidth;
}

// Helper function to get device ID for a rank
inline int getDeviceId(int rank, int numRanks) {
    int numDevices;
    CUDACHECK(cudaGetDeviceCount(&numDevices));
    return rank % numDevices;
}

// MPI descriptor for CPU host memory management
template <typename T>
class descrMPI {
public:
    // Configuration parameters
    int rank;
    int numRanks;
    int totalWidth;
    int totalHeight;
    int innerWidth;
    int innerHeight;
    int gridWidth;
    int gridHeight;
    int haloWidth;

    // Boundary and position information
    int globalStartRow;
    int globalEndRow;
    int globalStartCol;
    int globalEndCol;

    int localStartRow;
    int localStartCol;
    int localEndRow;
    int localEndCol;

    // Host data buffers (only allocated if needed)
    T* h_globalData;    // Only for rank 0, index 0
    T* h_sendBuffer;    // Only for rank 0, index 0
    T* h_recvBuffer;    // Only for non-rank-0 ranks
    T* h_localData;     // For all ranks (with halo)

    int localWidth;
    int localHeight;

private:
    // Helper function to calculate boundary positions
    void calculateBoundaryPositions() {
        int row = rank / gridWidth;
        int col = rank % gridWidth;

        if (row > 0) {
            globalStartRow = row * innerHeight - haloWidth;
            localStartRow = 0;
        } else {
            globalStartRow = 0;
            localStartRow = haloWidth;
        }

        if (row < gridHeight - 1) {
            globalEndRow = row * innerHeight + innerHeight + haloWidth;
            localEndRow = innerHeight + 2 * haloWidth;
        } else {
            globalEndRow = row * innerHeight + innerHeight;
            localEndRow = innerHeight + haloWidth;
        }

        if (col > 0) {
            globalStartCol = col * innerWidth - haloWidth;
            localStartCol = 0;
        } else {
            globalStartCol = 0;
            localStartCol = haloWidth;
        }

        if (col < gridWidth - 1) {
            globalEndCol = col * innerWidth + innerWidth + haloWidth;
            localEndCol = innerWidth + 2 * haloWidth;
        } else {
            globalEndCol = col * innerWidth + innerWidth;
            localEndCol = innerWidth + haloWidth;
        }
    }

public:
    // Constructor for rank 0 (index 0) - allocates all buffers
    descrMPI(int rank, int numRanks, int totalWidth, int totalHeight,
             int innerWidth, int innerHeight,
             int gridWidth, int gridHeight, int haloWidth)
        : rank(rank), numRanks(numRanks), totalWidth(totalWidth),
          totalHeight(totalHeight), innerWidth(innerWidth),
          innerHeight(innerHeight), gridWidth(gridWidth),
          gridHeight(gridHeight), haloWidth(haloWidth),
          globalStartRow(0), globalEndRow(0),
          globalStartCol(0), globalEndCol(0),
          localStartRow(0), localStartCol(0),
          localEndRow(0), localEndCol(0),
          h_globalData(nullptr), h_sendBuffer(nullptr), h_recvBuffer(nullptr),
          h_localData(nullptr), localWidth(0), localHeight(0)
    {
        if (rank != 0) {
            fprintf(stderr, "Error: Constructor with full parameters should only be called for rank 0\n");
            exit(EXIT_FAILURE);
        }

        localWidth = innerWidth + 2 * haloWidth;
        localHeight = innerHeight + 2 * haloWidth;

        // Calculate boundary positions
        calculateBoundaryPositions();

        // Allocate global data (only rank 0 needs this)
        h_globalData = new T[totalWidth * totalHeight];
        if (!h_globalData) {
            fprintf(stderr, "Failed to allocate h_globalData\n");
            exit(EXIT_FAILURE);
        }

        // Allocate send buffer for MPI continuous memory
        h_sendBuffer = new T[localWidth * localHeight];
        if (!h_sendBuffer) {
            fprintf(stderr, "Failed to allocate h_sendBuffer\n");
            exit(EXIT_FAILURE);
        }

        // Allocate local data for rank 0
        h_localData = new T[localWidth * localHeight];
        if (!h_localData) {
            fprintf(stderr, "Failed to allocate h_localData\n");
            exit(EXIT_FAILURE);
        }

        memset(h_localData, 0, localWidth * localHeight * sizeof(T));
    }

    // Constructor for non-rank-0 descriptors (full descriptor with buffer allocation)
    descrMPI(int rank, int numRanks, int totalWidth, int totalHeight,
             int innerWidth, int innerHeight,
             int gridWidth, int gridHeight, int haloWidth,
             bool isRank0Descriptor)  // false indicates non-rank-0
        : rank(rank), numRanks(numRanks), totalWidth(totalWidth),
          totalHeight(totalHeight), innerWidth(innerWidth),
          innerHeight(innerHeight), gridWidth(gridWidth),
          gridHeight(gridHeight), haloWidth(haloWidth),
          globalStartRow(0), globalEndRow(0),
          globalStartCol(0), globalEndCol(0),
          localStartRow(0), localStartCol(0),
          localEndRow(0), localEndCol(0),
          h_globalData(nullptr), h_sendBuffer(nullptr), h_recvBuffer(nullptr),
          h_localData(nullptr), localWidth(0), localHeight(0)
    {
        if (rank == 0 && !isRank0Descriptor) {
            fprintf(stderr, "Error: Non-rank-0 descriptor constructor used for rank 0\n");
            exit(EXIT_FAILURE);
        }

        if (!isRank0Descriptor) {
            // Non-rank-0 rank initialization
            localWidth = innerWidth + 2 * haloWidth;
            localHeight = innerHeight + 2 * haloWidth;

            // Calculate boundary positions
            calculateBoundaryPositions();

            // Allocate send buffer for MPI continuous memory (for sending results to rank 0)
            h_sendBuffer = new T[innerWidth * innerHeight];
            if (!h_sendBuffer) {
                fprintf(stderr, "Failed to allocate h_sendBuffer for rank %d\n", rank);
                exit(EXIT_FAILURE);
            }

            // Allocate receive buffer for MPI continuous memory
            h_recvBuffer = new T[localWidth * localHeight];
            if (!h_recvBuffer) {
                fprintf(stderr, "Failed to allocate h_recvBuffer for rank %d\n", rank);
                exit(EXIT_FAILURE);
            }

            // Allocate local data for this rank (with halo)
            h_localData = new T[localWidth * localHeight];
            if (!h_localData) {
                fprintf(stderr, "Failed to allocate h_localData for rank %d\n", rank);
                exit(EXIT_FAILURE);
            }

            memset(h_localData, 0, localWidth * localHeight * sizeof(T));
        }
    }

    // Constructor for descriptors in rank 0's array (boundary info only, no allocation)
    descrMPI(int rank, int numRanks, int totalWidth, int totalHeight,
             int innerWidth, int innerHeight,
             int gridWidth, int gridHeight, int haloWidth,
             bool isRank0Descriptor, bool allocateBuffers)  // both false = boundary only
        : rank(rank), numRanks(numRanks), totalWidth(totalWidth),
          totalHeight(totalHeight), innerWidth(innerWidth),
          innerHeight(innerHeight), gridWidth(gridWidth),
          gridHeight(gridHeight), haloWidth(haloWidth),
          globalStartRow(0), globalEndRow(0),
          globalStartCol(0), globalEndCol(0),
          localStartRow(0), localStartCol(0),
          localEndRow(0), localEndCol(0),
          h_globalData(nullptr), h_sendBuffer(nullptr), h_recvBuffer(nullptr),
          h_localData(nullptr), localWidth(0), localHeight(0)
    {
        // Only calculate boundary positions, no buffer allocation
        calculateBoundaryPositions();
    }

    // Default constructor
    descrMPI()
        : rank(-1), numRanks(0), totalWidth(0), totalHeight(0),
          innerWidth(0), innerHeight(0), gridWidth(0), gridHeight(0),
          haloWidth(0), globalStartRow(0), globalEndRow(0),
          globalStartCol(0), globalEndCol(0),
          localStartRow(0), localStartCol(0),
          localEndRow(0), localEndCol(0),
          h_globalData(nullptr), h_sendBuffer(nullptr),
          h_recvBuffer(nullptr), h_localData(nullptr),
          localWidth(0), localHeight(0)
    {
    }

    // Move constructor
    descrMPI(descrMPI&& other) noexcept
        : rank(other.rank), numRanks(other.numRanks),
          totalWidth(other.totalWidth), totalHeight(other.totalHeight),
          innerWidth(other.innerWidth), innerHeight(other.innerHeight),
          gridWidth(other.gridWidth), gridHeight(other.gridHeight),
          haloWidth(other.haloWidth),
          globalStartRow(other.globalStartRow), globalEndRow(other.globalEndRow),
          globalStartCol(other.globalStartCol), globalEndCol(other.globalEndCol),
          localStartRow(other.localStartRow), localStartCol(other.localStartCol),
          localEndRow(other.localEndRow), localEndCol(other.localEndCol),
          h_globalData(other.h_globalData), h_sendBuffer(other.h_sendBuffer),
          h_recvBuffer(other.h_recvBuffer), h_localData(other.h_localData),
          localWidth(other.localWidth), localHeight(other.localHeight)
    {
        // Clear the other's pointers to prevent double deletion
        other.h_globalData = nullptr;
        other.h_sendBuffer = nullptr;
        other.h_recvBuffer = nullptr;
        other.h_localData = nullptr;
    }

    // Move assignment operator
    descrMPI& operator=(descrMPI&& other) noexcept {
        if (this != &other) {
            // Clean up our existing resources
            if (h_globalData) delete[] h_globalData;
            if (h_sendBuffer) delete[] h_sendBuffer;
            if (h_recvBuffer) delete[] h_recvBuffer;
            if (h_localData) delete[] h_localData;

            // Move data from other
            rank = other.rank;
            numRanks = other.numRanks;
            totalWidth = other.totalWidth;
            totalHeight = other.totalHeight;
            innerWidth = other.innerWidth;
            innerHeight = other.innerHeight;
            gridWidth = other.gridWidth;
            gridHeight = other.gridHeight;
            haloWidth = other.haloWidth;
            globalStartRow = other.globalStartRow;
            globalEndRow = other.globalEndRow;
            globalStartCol = other.globalStartCol;
            globalEndCol = other.globalEndCol;
            localStartRow = other.localStartRow;
            localStartCol = other.localStartCol;
            localEndRow = other.localEndRow;
            localEndCol = other.localEndCol;
            h_globalData = other.h_globalData;
            h_sendBuffer = other.h_sendBuffer;
            h_recvBuffer = other.h_recvBuffer;
            h_localData = other.h_localData;
            localWidth = other.localWidth;
            localHeight = other.localHeight;

            // Clear the other's pointers to prevent double deletion
            other.h_globalData = nullptr;
            other.h_sendBuffer = nullptr;
            other.h_recvBuffer = nullptr;
            other.h_localData = nullptr;
        }
        return *this;
    }

    // Delete copy constructor and copy assignment (prevent unintended copies)
    descrMPI(const descrMPI&) = delete;
    descrMPI& operator=(const descrMPI&) = delete;

    // Destructor
    ~descrMPI() {
        if (h_globalData) delete[] h_globalData;
        if (h_sendBuffer) delete[] h_sendBuffer;
        if (h_recvBuffer) delete[] h_recvBuffer;
        if (h_localData) delete[] h_localData;
    }
};

// Initialize descriptor array for rank 0 (creates all descriptors with boundary positions)
template <typename T>
void initialMPIDscrArray(descrMPI<T>* array, int numRanks,
                         int totalWidth, int totalHeight,
                         int innerWidth, int innerHeight,
                         int gridWidth, int gridHeight,
                         int haloWidth)
{
    // Initialize rank 0 descriptor (index 0) with full buffers
    array[0] = descrMPI<T>(0, numRanks, totalWidth, totalHeight,
                          innerWidth, innerHeight,
                          gridWidth, gridHeight, haloWidth);

    // Initialize other descriptors in the array (boundary info only)
    for (int i = 1; i < numRanks; ++i) {
        array[i] = descrMPI<T>(i, numRanks, totalWidth, totalHeight,
                              innerWidth, innerHeight,
                              gridWidth, gridHeight, haloWidth,
                              false, false);  // boundary info only, no buffer allocation
    }
}

// Distribute initial data from rank 0 to all ranks
template <typename T>
void distributeInitialData(descrMPI<T>* descrArray, T* localData, MPI_Comm comm)
{
    int rank = descrArray[0].rank;
    int numRanks = descrArray[0].numRanks;
    int totalWidth = descrArray[0].totalWidth;
    int innerWidth = descrArray[0].innerWidth;
    int haloWidth = descrArray[0].haloWidth;

    int localWidth = innerWidth + 2 * haloWidth;
    int localHeight = descrArray[0].innerHeight + 2 * haloWidth;

    if (rank == 0) {
        // Rank 0: copy its own data and send to others
        T* globalData = descrArray[0].h_globalData;
        int lineLen = descrArray[0].localEndCol - descrArray[0].localStartCol;

        // Copy rank 0's data
        for (int row = 0; row < descrArray[0].globalEndRow; row++) {
            memcpy(&localData[row * localWidth + haloWidth + haloWidth * localWidth],
                   &globalData[row * totalWidth],
                   lineLen * sizeof(T));
        }

        // Send data to other ranks - use the descriptor's sendBuffer
        T* sendBuffer = descrArray[0].h_sendBuffer;
        for (int sendRank = 1; sendRank < numRanks; ++sendRank) {
            int globalToSendBias = descrArray[sendRank].globalStartRow * totalWidth +
                                   descrArray[sendRank].globalStartCol;
            lineLen = descrArray[sendRank].localEndCol - descrArray[sendRank].localStartCol;
            int sendBufBias = 0;

            for (int row = descrArray[sendRank].globalStartRow;
                 row < descrArray[sendRank].globalEndRow; row++) {
                memcpy(&sendBuffer[sendBufBias],
                       &globalData[globalToSendBias],
                       lineLen * sizeof(T));
                sendBufBias += lineLen;
                globalToSendBias += totalWidth;
            }

            MPI_Send(sendBuffer, sendBufBias * sizeof(T), MPI_CHAR,
                    sendRank, 0, comm);
        }
    } else {
        // Non-rank-0: receive data
        T* recvBuffer = descrArray[0].h_recvBuffer;
        MPI_Status status;

        int lineLen = descrArray[0].localEndCol - descrArray[0].localStartCol;
        int recvSize = lineLen * (descrArray[0].localEndRow - descrArray[0].localStartRow);

        MPI_Recv(recvBuffer, recvSize * sizeof(T), MPI_CHAR, 0, 0, comm, &status);

        int recvToLocalBias = descrArray[0].localStartRow * localWidth +
                             descrArray[0].localStartCol;
        int recvBufBias = 0;

        for (int row = descrArray[0].localStartRow;
             row < descrArray[0].localEndRow; row++) {
            memcpy(&localData[recvToLocalBias],
                   &recvBuffer[recvBufBias],
                   lineLen * sizeof(T));
            recvToLocalBias += localWidth;
            recvBufBias += lineLen;
        }
    }

    MPI_Barrier(comm);
}

// Gather results from all ranks to rank 0
template <typename T>
void gatherResults(descrMPI<T>* descrArray, T* localData, T* globalData, MPI_Comm comm)
{
    int rank = descrArray[0].rank;
    int numRanks = descrArray[0].numRanks;
    int totalWidth = descrArray[0].totalWidth;
    int innerWidth = descrArray[0].innerWidth;
    int innerHeight = descrArray[0].innerHeight;
    int gridWidth = descrArray[0].gridWidth;
    int haloWidth = descrArray[0].haloWidth;

    if (rank > 0) {
        // Non-rank-0: send data to rank 0
        T* sendBuffer = descrArray[0].h_sendBuffer;
        for (int row = 0; row < innerHeight; ++row) {
            int localToSendBias = haloWidth * (innerWidth + 2 * haloWidth) +
                                  row * (innerWidth + 2 * haloWidth) + haloWidth;
            memcpy(&sendBuffer[row * innerWidth],
                   &localData[localToSendBias],
                   innerWidth * sizeof(T));
        }

        MPI_Send(sendBuffer, innerHeight * innerWidth * sizeof(T), MPI_CHAR, 0, 0, comm);
    }

    if (rank == 0) {
        // Rank 0: receive from all ranks
        T* recvBuffer = descrArray[0].h_sendBuffer;  // Reuse sendBuffer for receiving
        for (int r = 0; r < numRanks; ++r) {
            if (r > 0) {
                MPI_Recv(recvBuffer, innerHeight * innerWidth * sizeof(T),
                        MPI_CHAR, r, 0, comm, MPI_STATUS_IGNORE);

                int col_global = (r % gridWidth) * innerWidth;
                int row_global = r / gridWidth * innerHeight;
                int recvToGlobalBias = row_global * totalWidth + col_global;

                for (int line = 0; line < innerHeight; ++line) {
                    memcpy(&globalData[recvToGlobalBias],
                           &recvBuffer[line * innerWidth],
                           innerWidth * sizeof(T));
                    recvToGlobalBias += totalWidth;
                }
            } else {
                // Copy rank 0's own data
                int recvToGlobalBias = 0;
                for (int line = 0; line < innerHeight; ++line) {
                    memcpy(&globalData[recvToGlobalBias],
                           &localData[(line + haloWidth) * (innerWidth + 2 * haloWidth) + haloWidth],
                           innerWidth * sizeof(T));
                    recvToGlobalBias += totalWidth;
                }
            }
        }
    }

    MPI_Barrier(comm);
}

// ============================================================================
// PART 2: GPU DATA EXCHANGE (NCCL-based Halo Exchange)
// ============================================================================

template <typename T>
void allocateDeviceMemory(T** devPtr, size_t count) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));

    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Memory Allocation Error: %s\n", cudaGetErrorString(err));
        return;
    }

    *devPtr = static_cast<T*>(ptr);
}

// GPU Halo descriptor class
template <typename T>
class descrHalo {
public:
    // Data buffers (on GPU device memory)
    T* localData;           // Local data with halo

    // Send/Receive buffers for halo exchange
    T* sendBufferLeft;
    T* recvBufferLeft;
    T* sendBufferRight;
    T* recvBufferRight;
    T* sendBufferUp;
    T* recvBufferUp;
    T* sendBufferDown;
    T* recvBufferDown;

    T* sendBufferUpLeft;
    T* recvBufferUpLeft;
    T* sendBufferUpRight;
    T* recvBufferUpRight;
    T* sendBufferDownLeft;
    T* recvBufferDownLeft;
    T* sendBufferDownRight;
    T* recvBufferDownRight;

    // Descriptor information
    int haloWidth;
    int innerWidth;
    int innerHeight;
    int commRank;
    int chunkRow;
    int chunkCol;
    int gridWidth;
    int gridHeight;

    // Neighbor ranks
    int leftRank;
    int rightRank;
    int upRank;
    int downRank;
    int upLeftRank;
    int upRightRank;
    int downLeftRank;
    int downRightRank;

    // Lambda for computing neighbor ranks
    std::function<int(int, int)> nbr;

    // Constructor
    descrHalo(int rank, int innerWidth, int innerHeight,
             int gridWidth, int gridHeight, int haloWidth)
        : localData(nullptr),
          sendBufferLeft(nullptr), recvBufferLeft(nullptr),
          sendBufferRight(nullptr), recvBufferRight(nullptr),
          sendBufferUp(nullptr), recvBufferUp(nullptr),
          sendBufferDown(nullptr), recvBufferDown(nullptr),
          sendBufferUpLeft(nullptr), recvBufferUpLeft(nullptr),
          sendBufferUpRight(nullptr), recvBufferUpRight(nullptr),
          sendBufferDownLeft(nullptr), recvBufferDownLeft(nullptr),
          sendBufferDownRight(nullptr), recvBufferDownRight(nullptr),
          commRank(rank), haloWidth(haloWidth),
          innerWidth(innerWidth), innerHeight(innerHeight),
          gridWidth(gridWidth), gridHeight(gridHeight)
    {
        // Calculate chunk position
        chunkRow = commRank / gridWidth;
        chunkCol = commRank % gridWidth;

        // Define neighbor function
        nbr = [this](int ny, int nx) -> int {
            if (nx < 0 || nx >= this->gridWidth || ny < 0 || ny >= this->gridHeight) return -1;
            return nx + ny * this->gridWidth;
        };

        // Calculate neighbor ranks
        leftRank = nbr(chunkRow, chunkCol - 1);
        rightRank = nbr(chunkRow, chunkCol + 1);
        upRank = nbr(chunkRow - 1, chunkCol);
        downRank = nbr(chunkRow + 1, chunkCol);
        upLeftRank = nbr(chunkRow - 1, chunkCol - 1);
        upRightRank = nbr(chunkRow - 1, chunkCol + 1);
        downLeftRank = nbr(chunkRow + 1, chunkCol - 1);
        downRightRank = nbr(chunkRow + 1, chunkCol + 1);

        // Allocate device memory for local data
        allocateDeviceMemory<T>(&this->localData,
                               (innerHeight + 2 * haloWidth) * (innerWidth + 2 * haloWidth));

        // Allocate buffers for each direction
        allocateDeviceMemory<T>(&this->sendBufferLeft, innerHeight * haloWidth);
        allocateDeviceMemory<T>(&this->recvBufferLeft, innerHeight * haloWidth);

        allocateDeviceMemory<T>(&this->sendBufferRight, innerHeight * haloWidth);
        allocateDeviceMemory<T>(&this->recvBufferRight, innerHeight * haloWidth);

        allocateDeviceMemory<T>(&this->sendBufferUp, innerWidth * haloWidth);
        allocateDeviceMemory<T>(&this->recvBufferUp, innerWidth * haloWidth);

        allocateDeviceMemory<T>(&this->sendBufferDown, innerWidth * haloWidth);
        allocateDeviceMemory<T>(&this->recvBufferDown, innerWidth * haloWidth);

        // Allocate buffers for corners
        allocateDeviceMemory<T>(&this->sendBufferUpLeft, haloWidth * haloWidth);
        allocateDeviceMemory<T>(&this->recvBufferUpLeft, haloWidth * haloWidth);

        allocateDeviceMemory<T>(&this->sendBufferUpRight, haloWidth * haloWidth);
        allocateDeviceMemory<T>(&this->recvBufferUpRight, haloWidth * haloWidth);

        allocateDeviceMemory<T>(&this->sendBufferDownLeft, haloWidth * haloWidth);
        allocateDeviceMemory<T>(&this->recvBufferDownLeft, haloWidth * haloWidth);

        allocateDeviceMemory<T>(&this->sendBufferDownRight, haloWidth * haloWidth);
        allocateDeviceMemory<T>(&this->recvBufferDownRight, haloWidth * haloWidth);
    }

    // Destructor
    ~descrHalo() {
        if (localData) cudaFree(localData);

        if (sendBufferLeft) cudaFree(sendBufferLeft);
        if (recvBufferLeft) cudaFree(recvBufferLeft);

        if (sendBufferRight) cudaFree(sendBufferRight);
        if (recvBufferRight) cudaFree(recvBufferRight);

        if (sendBufferUp) cudaFree(sendBufferUp);
        if (recvBufferUp) cudaFree(recvBufferUp);

        if (sendBufferDown) cudaFree(sendBufferDown);
        if (recvBufferDown) cudaFree(recvBufferDown);

        if (sendBufferUpLeft) cudaFree(sendBufferUpLeft);
        if (recvBufferUpLeft) cudaFree(recvBufferUpLeft);

        if (sendBufferUpRight) cudaFree(sendBufferUpRight);
        if (recvBufferUpRight) cudaFree(recvBufferUpRight);

        if (sendBufferDownLeft) cudaFree(sendBufferDownLeft);
        if (recvBufferDownLeft) cudaFree(recvBufferDownLeft);

        if (sendBufferDownRight) cudaFree(sendBufferDownRight);
        if (recvBufferDownRight) cudaFree(recvBufferDownRight);
    }
};

// Halo exchange function (NCCL-based GPU data transfer)
template <typename T>
void exchangeHalo(descrHalo<T>& exchangeDescr, ncclComm_t comm, cudaStream_t stream) {
    int haloWidth = exchangeDescr.haloWidth;
    int innerWidth = exchangeDescr.innerWidth;
    int innerHeight = exchangeDescr.innerHeight;
    int localStride = innerWidth + 2 * haloWidth;

    int leftRank = exchangeDescr.leftRank;
    int rightRank = exchangeDescr.rightRank;
    int upRank = exchangeDescr.upRank;
    int downRank = exchangeDescr.downRank;
    int upLeftRank = exchangeDescr.upLeftRank;
    int upRightRank = exchangeDescr.upRightRank;
    int downLeftRank = exchangeDescr.downLeftRank;
    int downRightRank = exchangeDescr.downRightRank;

    // =====================================================================
    // PREPARE BUFFERS: Copy from localData to send buffers
    // =====================================================================

    if (leftRank >= 0) {
        int leftSendBias = haloWidth * localStride + haloWidth;
        cudaMemcpy2D(exchangeDescr.sendBufferLeft, haloWidth * sizeof(T),
                    exchangeDescr.localData + leftSendBias, localStride * sizeof(T),
                    haloWidth * sizeof(T), innerHeight, cudaMemcpyDeviceToDevice);
    }

    if (rightRank >= 0) {
        int rightSendBias = haloWidth * localStride + innerWidth;
        cudaMemcpy2D(exchangeDescr.sendBufferRight, haloWidth * sizeof(T),
                    exchangeDescr.localData + rightSendBias, localStride * sizeof(T),
                    haloWidth * sizeof(T), innerHeight, cudaMemcpyDeviceToDevice);
    }

    if (upRank >= 0) {
        int upSendBias = haloWidth * localStride + haloWidth;
        cudaMemcpy2D(exchangeDescr.sendBufferUp, innerWidth * sizeof(T),
                    exchangeDescr.localData + upSendBias, localStride * sizeof(T),
                    innerWidth * sizeof(T), haloWidth, cudaMemcpyDeviceToDevice);
    }

    if (downRank >= 0) {
        int downSendBias = innerHeight * localStride + haloWidth;
        cudaMemcpy2D(exchangeDescr.sendBufferDown, innerWidth * sizeof(T),
                    exchangeDescr.localData + downSendBias, localStride * sizeof(T),
                    innerWidth * sizeof(T), haloWidth, cudaMemcpyDeviceToDevice);
    }

    if (upLeftRank >= 0) {
        int upLeftSendBias = haloWidth * localStride + haloWidth;
        cudaMemcpy2D(exchangeDescr.sendBufferUpLeft, haloWidth * sizeof(T),
                    exchangeDescr.localData + upLeftSendBias, localStride * sizeof(T),
                    haloWidth * sizeof(T), haloWidth, cudaMemcpyDeviceToDevice);
    }

    if (upRightRank >= 0) {
        int upRightSendBias = haloWidth * localStride + innerWidth;
        cudaMemcpy2D(exchangeDescr.sendBufferUpRight, haloWidth * sizeof(T),
                    exchangeDescr.localData + upRightSendBias, localStride * sizeof(T),
                    haloWidth * sizeof(T), haloWidth, cudaMemcpyDeviceToDevice);
    }

    if (downLeftRank >= 0) {
        int downLeftSendBias = innerHeight * localStride + haloWidth;
        cudaMemcpy2D(exchangeDescr.sendBufferDownLeft, haloWidth * sizeof(T),
                    exchangeDescr.localData + downLeftSendBias, localStride * sizeof(T),
                    haloWidth * sizeof(T), haloWidth, cudaMemcpyDeviceToDevice);
    }

    if (downRightRank >= 0) {
        int downRightSendBias = innerHeight * localStride + innerWidth;
        cudaMemcpy2D(exchangeDescr.sendBufferDownRight, haloWidth * sizeof(T),
                    exchangeDescr.localData + downRightSendBias, localStride * sizeof(T),
                    haloWidth * sizeof(T), haloWidth, cudaMemcpyDeviceToDevice);
    }

    // =====================================================================
    // NCCL COMMUNICATION: Send and receive data
    // =====================================================================

    NCCLCHECK(ncclGroupStart());

    if (leftRank >= 0) {
        NCCLCHECK(ncclSend(exchangeDescr.sendBufferLeft, haloWidth * innerHeight * sizeof(T),
                          ncclChar, leftRank, comm, stream));
        NCCLCHECK(ncclRecv(exchangeDescr.recvBufferLeft, haloWidth * innerHeight * sizeof(T),
                          ncclChar, leftRank, comm, stream));
    }

    if (rightRank >= 0) {
        NCCLCHECK(ncclSend(exchangeDescr.sendBufferRight, haloWidth * innerHeight * sizeof(T),
                          ncclChar, rightRank, comm, stream));
        NCCLCHECK(ncclRecv(exchangeDescr.recvBufferRight, haloWidth * innerHeight * sizeof(T),
                          ncclChar, rightRank, comm, stream));
    }

    if (upRank >= 0) {
        NCCLCHECK(ncclSend(exchangeDescr.sendBufferUp, haloWidth * innerWidth * sizeof(T),
                          ncclChar, upRank, comm, stream));
        NCCLCHECK(ncclRecv(exchangeDescr.recvBufferUp, haloWidth * innerWidth * sizeof(T),
                          ncclChar, upRank, comm, stream));
    }

    if (downRank >= 0) {
        NCCLCHECK(ncclSend(exchangeDescr.sendBufferDown, haloWidth * innerWidth * sizeof(T),
                          ncclChar, downRank, comm, stream));
        NCCLCHECK(ncclRecv(exchangeDescr.recvBufferDown, haloWidth * innerWidth * sizeof(T),
                          ncclChar, downRank, comm, stream));
    }

    if (upLeftRank >= 0) {
        NCCLCHECK(ncclSend(exchangeDescr.sendBufferUpLeft, haloWidth * haloWidth * sizeof(T),
                          ncclChar, upLeftRank, comm, stream));
        NCCLCHECK(ncclRecv(exchangeDescr.recvBufferUpLeft, haloWidth * haloWidth * sizeof(T),
                          ncclChar, upLeftRank, comm, stream));
    }

    if (upRightRank >= 0) {
        NCCLCHECK(ncclSend(exchangeDescr.sendBufferUpRight, haloWidth * haloWidth * sizeof(T),
                          ncclChar, upRightRank, comm, stream));
        NCCLCHECK(ncclRecv(exchangeDescr.recvBufferUpRight, haloWidth * haloWidth * sizeof(T),
                          ncclChar, upRightRank, comm, stream));
    }

    if (downLeftRank >= 0) {
        NCCLCHECK(ncclSend(exchangeDescr.sendBufferDownLeft, haloWidth * haloWidth * sizeof(T),
                          ncclChar, downLeftRank, comm, stream));
        NCCLCHECK(ncclRecv(exchangeDescr.recvBufferDownLeft, haloWidth * haloWidth * sizeof(T),
                          ncclChar, downLeftRank, comm, stream));
    }

    if (downRightRank >= 0) {
        NCCLCHECK(ncclSend(exchangeDescr.sendBufferDownRight, haloWidth * haloWidth * sizeof(T),
                          ncclChar, downRightRank, comm, stream));
        NCCLCHECK(ncclRecv(exchangeDescr.recvBufferDownRight, haloWidth * haloWidth * sizeof(T),
                          ncclChar, downRightRank, comm, stream));
    }

    NCCLCHECK(ncclGroupEnd());

    // =====================================================================
    // UPDATE LOCAL DATA: Copy from receive buffers back to localData
    // =====================================================================

    if (leftRank >= 0) {
        int leftRecvBias = haloWidth * localStride;
        cudaMemcpy2D(exchangeDescr.localData + leftRecvBias, localStride * sizeof(T),
                    exchangeDescr.recvBufferLeft, haloWidth * sizeof(T),
                    haloWidth * sizeof(T), innerHeight, cudaMemcpyDeviceToDevice);
    }

    if (rightRank >= 0) {
        int rightRecvBias = haloWidth * localStride + innerWidth + haloWidth;
        cudaMemcpy2D(exchangeDescr.localData + rightRecvBias, localStride * sizeof(T),
                    exchangeDescr.recvBufferRight, haloWidth * sizeof(T),
                    haloWidth * sizeof(T), innerHeight, cudaMemcpyDeviceToDevice);
    }

    if (upRank >= 0) {
        int upRecvBias = haloWidth;
        cudaMemcpy2D(exchangeDescr.localData + upRecvBias, localStride * sizeof(T),
                    exchangeDescr.recvBufferUp, innerWidth * sizeof(T),
                    innerWidth * sizeof(T), haloWidth, cudaMemcpyDeviceToDevice);
    }

    if (downRank >= 0) {
        int downRecvBias = (innerHeight + haloWidth) * localStride + haloWidth;
        cudaMemcpy2D(exchangeDescr.localData + downRecvBias, localStride * sizeof(T),
                    exchangeDescr.recvBufferDown, innerWidth * sizeof(T),
                    innerWidth * sizeof(T), haloWidth, cudaMemcpyDeviceToDevice);
    }

    if (upLeftRank >= 0) {
        int upLeftRecvBias = 0;
        cudaMemcpy2D(exchangeDescr.localData + upLeftRecvBias, localStride * sizeof(T),
                    exchangeDescr.recvBufferUpLeft, haloWidth * sizeof(T),
                    haloWidth * sizeof(T), haloWidth, cudaMemcpyDeviceToDevice);
    }

    if (upRightRank >= 0) {
        int upRightRecvBias = innerWidth + haloWidth;
        cudaMemcpy2D(exchangeDescr.localData + upRightRecvBias, localStride * sizeof(T),
                    exchangeDescr.recvBufferUpRight, haloWidth * sizeof(T),
                    haloWidth * sizeof(T), haloWidth, cudaMemcpyDeviceToDevice);
    }

    if (downLeftRank >= 0) {
        int downLeftRecvBias = (innerHeight + haloWidth) * localStride;
        cudaMemcpy2D(exchangeDescr.localData + downLeftRecvBias, localStride * sizeof(T),
                    exchangeDescr.recvBufferDownLeft, haloWidth * sizeof(T),
                    haloWidth * sizeof(T), haloWidth, cudaMemcpyDeviceToDevice);
    }

    if (downRightRank >= 0) {
        int downRightRecvBias = (innerHeight + haloWidth) * localStride + innerWidth + haloWidth;
        cudaMemcpy2D(exchangeDescr.localData + downRightRecvBias, localStride * sizeof(T),
                    exchangeDescr.recvBufferDownRight, haloWidth * sizeof(T),
                    haloWidth * sizeof(T), haloWidth, cudaMemcpyDeviceToDevice);
    }
}

#endif // HALO_LIB_HPP
