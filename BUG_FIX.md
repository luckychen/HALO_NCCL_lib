# Bug Fix: Segmentation Fault in gatherResults()

## Issue Description

When running the halo_test program with the command:
```bash
mpirun --mca psec ^munge -n 4 --oversubscribe ./halo_test 1024 1024 2 2 2 3
```

The program would complete all GPU computations and halo exchanges successfully but crash with a **segmentation fault** during cleanup (after iteration 2), affecting all ranks.

### Error Output:
```
Signal: Segmentation fault (11)
Signal code: Address not mapped (1)
Failing at address: (nil)
```

## Root Cause

The bug was in the `gatherResults()` function in `halo_lib.hpp` (line 416):

```cpp
if (rank > 0) {
    // Non-rank-0: send data to rank 0
    T* sendBuffer = descrArray[0].h_sendBuffer;  // <-- BUG: h_sendBuffer is NULL!
    // ... copy data to sendBuffer ...
    MPI_Send(sendBuffer, ...);
}
```

### Why This Happens:

1. **Rank 0 descriptor** (index 0) allocates three buffers:
   - `h_globalData` (global data)
   - `h_sendBuffer` (for sending data)
   - `h_localData` (local rank 0 data)

2. **Non-rank-0 descriptors** only allocate:
   - `h_recvBuffer` (for receiving from rank 0)
   - `h_localData` (local data with halo)
   - **NO `h_sendBuffer`**

3. When non-rank-0 ranks call `gatherResults()`, they try to use `descrArray[0].h_sendBuffer` which is **nullptr**, causing:
   - Undefined behavior in `memcpy()` operations (lines 420-422)
   - Crash when `MPI_Send()` tries to send from null pointer
   - Segmentation fault during MPI finalization when cleaning up corrupted state

## Solution

Modified the **`descrMPI` class non-rank-0 constructor** to allocate a **send buffer** in addition to the receive buffer:

```cpp
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
```

This ensures that:
1. Rank 0's descriptor (index 0) has all three buffers: `h_globalData`, `h_sendBuffer`, `h_localData`
2. Non-rank-0 descriptors have: `h_sendBuffer`, `h_recvBuffer`, `h_localData`
3. The `gatherResults()` function can safely use `descrArray[0].h_sendBuffer` for all ranks

## Impact

- **Memory overhead**: Buffer allocation of size `innerWidth * innerHeight * sizeof(T)` per non-rank-0 rank at initialization (only once)
- **Performance**: Negligible impact; allocations happen once at initialization, before computation
- **Design**: Proper separation of concerns - descriptors manage their own memory, not functions
- **Correctness**: Fixes the segmentation fault and allows the program to complete successfully

## Files Modified

- `halo_lib.hpp`:
  1. Modified `descrMPI` class non-rank-0 constructor (lines 247-252)
     - Added `h_sendBuffer` allocation for non-rank-0 ranks
  2. Added move semantics to `descrMPI` class (lines 306-373)
     - Move constructor and move assignment operator
     - Deleted copy constructor and copy assignment operator

## Additional Fix: Double Free/Corruption Error (Part 1)

After fixing the initial segmentation fault, a **double free or corruption** error occurred during program cleanup.

### Root Cause #1: Temporary Assignment

In `main.cu` line 310, descriptor assignment was done with a temporary:
```cpp
descrArray[0] = descrMPI<DataType>(rank, ...);  // Temporary object!
```

This caused:
1. A temporary `descrMPI` object was created with allocated buffers
2. The temporary was **copy-assigned** to `descrArray[0]`
3. The temporary's destructor ran immediately and deleted all buffers
4. Later, `descrArray[0]` tried to delete already-freed memory → double free error

### Solution #1: Move Semantics

Added **move semantics** to the `descrMPI` class:
- **Move constructor**: Transfers ownership of buffers from temporary to target
- **Move assignment operator**: Properly transfers memory ownership and clears source pointers
- **Deleted copy operations**: Prevents accidental copying

This ensures buffers are only deleted once when the last owner is destroyed.

## Bug Fix #3: Double Deletion of h_gpuResult

After fixing issues 1 and 2, another **double free** occurred in the final cleanup phase.

### Root Cause

`h_gpuResult` was being deleted twice:
1. Line 404 in the comparison section: `delete[] h_gpuResult;`
2. Lines 409-411 in the cleanup section: `delete[] h_gpuResult;` (again!)

### Solution

Removed the first deletion (line 404) and kept only the one in the cleanup section. This ensures `h_gpuResult` is deleted exactly once.

**Files Modified:**
- `main.cu`: Lines 390-410 - removed duplicate deletion of `h_gpuResult`

## Testing

To verify both fixes, rebuild on the PowerPC platform and run:
```bash
cd /home/ceoas/chenchon/FEZ/HALO_lib_header/build
cmake ..
make
mpirun --mca psec ^munge -n 4 --oversubscribe ./halo_test 1024 1024 2 2 2 3
```

The program should now complete successfully with the message:
```
Test completed successfully!
```
