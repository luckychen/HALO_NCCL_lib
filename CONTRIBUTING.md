# Contributing to HALO Library

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork: `git clone https://github.com/yourusername/HALO_lib_header.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites
- CUDA Toolkit 7.0+
- MPI implementation (OpenMPI or MPICH)
- NCCL library
- CMake 3.18+
- Git

### Building for Development

```bash
mkdir build && cd build
cmake ..
make
```

### Running Tests

```bash
# 4-rank test on 2x2 GPU grid
mpirun -n 4 ./halo_test 1024 1024 2 2 2 3

# Expected output should show:
# - All iterations completing successfully
# - "TEST PASSED: Results match!" message
# - "Test completed successfully!" at the end
```

## Code Style

### C++ Standards
- Use C++14 features (compatible with PowerPC platform)
- Use meaningful variable names
- Add comments for complex logic
- Use const correctness

### Header-Only Library Guidelines
- All implementations in headers (no .cpp files for library code)
- Use inline functions where appropriate
- Provide clear template instantiation points
- Document template parameters

### Error Handling
- Use existing macros: `MPICHECK()`, `CUDACHECK()`, `NCCLCHECK()`
- Always check return values
- Provide meaningful error messages

## Pull Request Process

1. **Before submitting:**
   - Ensure your code builds without warnings: `make clean && make`
   - Run tests and verify they pass
   - Update documentation if needed
   - Add comments for non-obvious changes

2. **Create pull request with:**
   - Clear description of changes
   - Reference to any related issues
   - Test results showing your changes work
   - Updated CLAUDE.md or README.md if appropriate

3. **PR Template:**
   ```markdown
   ## Description
   Brief description of what this PR does

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Performance improvement
   - [ ] Documentation update

   ## Testing
   - [ ] I have tested this on PowerPC with CUDA
   - [ ] I have tested this with 4+ GPU ranks
   - [ ] Tests pass and produce expected output

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] I have commented complex logic
   - [ ] I have updated documentation
   - [ ] No new compiler warnings
   ```

## Reporting Issues

### Bug Reports
Include:
- Clear title describing the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (GPU, CUDA version, MPI, NCCL versions)
- Error messages and stack traces (if available)
- Platform (PowerPC/x86, number of GPUs)

### Feature Requests
Include:
- Clear description of the feature
- Motivation and use cases
- Possible implementation approach
- Impact on existing code

## Documentation Guidelines

- **README.md**: High-level overview and quick start
- **CLAUDE.md**: Detailed architecture and design
- **BUG_FIX.md**: Bug tracking and fixes
- **Code comments**: Non-obvious implementation details

### Documentation Standards
- Use clear, concise language
- Include code examples where helpful
- Update docs when changing functionality
- Include platform-specific notes

## Performance Considerations

When contributing performance-related changes:
- Provide benchmarks showing improvements
- Include test configuration (GPU type, problem size, rank count)
- Document any trade-offs
- Ensure backward compatibility

## Memory Management Guidelines

This library uses explicit memory management. When contributing:
- Always use `new`/`delete` for consistency
- Implement move semantics for resource-owning types
- Use RAII principles (Resource Acquisition Is Initialization)
- Clean up resources in destructors
- Test for memory leaks

## NCCL Communication Guidelines

When modifying NCCL code:
- Use `ncclGroupStart()/ncclGroupEnd()` for efficiency
- Handle all 8 neighbor directions (4 cardinal + 4 diagonal)
- Consider bandwidth and latency
- Document neighborhood patterns

## Testing Guidelines

- Test on multiple GPU configurations (2×2, 2×3, 3×3 grids)
- Test with different data types (float, double)
- Verify numerical accuracy (use compareResults function)
- Test error conditions and edge cases

## Commit Guidelines

- Write clear, descriptive commit messages
- Use imperative mood: "Add feature" not "Added feature"
- Reference issue numbers: "Fix #123"
- Keep commits focused on single changes
- Squash related commits before submitting PR

Example commit message:
```
Add support for custom halo patterns

- Allow variable halo regions per direction
- Update exchangeHalo() function signature
- Add examples in main.cu
- Fixes #45
```

## Code Review Process

- Expect constructive feedback
- Be open to suggestions
- Respond to review comments
- Make requested changes in new commits (don't force-push)

## Development Workflow

```
feature/your-feature
        ↓
   (develop & test)
        ↓
   Create PR
        ↓
   Code review
        ↓
   Merge to master
```

## Questions?

- Check existing issues and documentation first
- Open a GitHub issue with your question
- Include relevant code and configuration

## Recognition

Contributors will be:
- Listed in commit messages
- Acknowledged in documentation
- Credited for significant contributions

---

Thank you for helping improve the HALO library!
