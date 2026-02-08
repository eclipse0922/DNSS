# DNSS / NSS

Clean C++ implementation of:

- NSS: Normal-Space Sampling for ICP point selection.
- DNSS: Dual-Normal-Space Sampling from
  Tsz-Ho Kwok, "DNSS: Dual-Normal-Space Sampling for 3-D ICP Registration," IEEE T-ASE, 2018.

## What This Version Fixes

- Rewrites legacy code into readable, modular C++17.
- Adds missing method implementations (`setInputCloud`) and input validation.
- Fixes rotational-return math to use radians consistently.
- Uses efficient lazy removal (active-mask + heap cleanup) instead of repeated `erase` on vectors.
- Adds optional CUDA acceleration for DNSS rotational feature computation.

## Algorithm Notes

DNSS follows the paper's main idea:

1. Center and scale points by `Lmax`.
2. Build translational normal space from original normals.
3. Build rotational normal space from `nr = p x n`.
4. Compute rotational return and use it as the rotational-bucket constraint increment.
5. Iteratively pick points from the currently least-constrained bucket across both spaces.

Default bucket settings:

- Translational space: `12 x 6`
- Rotational space: `6 x 6`
- Return angle `theta`: `pi / 4`

## Build

CPU only:

```bash
cmake -S . -B build
cmake --build build
```

With CUDA (if CUDA toolkit is installed):

```bash
cmake -S . -B build -DDNSS_ENABLE_CUDA=ON
cmake --build build
```

When CUDA is enabled, call `DNSS::setUseCuda(true)` to request GPU acceleration. If GPU execution fails at runtime, the implementation falls back to the CPU path.

## References

- DNSS paper (2018): https://ieeexplore.ieee.org/document/8375750
- DNSS open-access repository page: https://spectrum.library.concordia.ca/id/eprint/984398/
- NSS/ICP variant context: Rusinkiewicz and Levoy, "Efficient Variants of the ICP Algorithm," 3DIM 2001.
