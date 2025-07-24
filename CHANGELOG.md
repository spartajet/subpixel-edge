# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-07-24

### Added
- Initial release of subpixel-edge library
- High-performance Canny-based subpixel edge detection algorithm
- Parallel Sobel gradient computation using rayon
- Non-maximum suppression for edge thinning
- Hysteresis thresholding with parallel optimization
- Subpixel edge refinement using parabolic fitting
- Bilinear interpolation for subpixel accuracy
- Edge visualization utilities
- Optional logger feature for debug output
- Comprehensive documentation with examples
- MIT/Apache-2.0 dual licensing

### Features
- `canny_based_subpixel_edges_optimized` - Main edge detection function
- `parallel_sobel_gradients` - Parallel gradient computation
- `visualize_edges` - Edge visualization on original image
- Optional `logger` feature for debugging and performance monitoring

### Performance
- Multi-core parallel processing with rayon
- Memory-efficient algorithms with pre-allocated buffers
- Optimized for modern CPU architectures
- Typical 2-4x speedup on multi-core systems

### Documentation
- Complete API documentation with rustdoc
- Comprehensive README with usage examples
- Parameter tuning guides for different image types
- Performance characteristics and benchmarks
- Development setup instructions

[Unreleased]: https://github.com/yourusername/subpixel-edge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/subpixel-edge/releases/tag/v0.1.0