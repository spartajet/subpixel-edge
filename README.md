# Subpixel Edge Detection Library

A high-performance Rust library for subpixel edge detection using optimized Canny algorithm with parallel processing capabilities.

## Overview

This library provides state-of-the-art subpixel edge detection with significant performance improvements through parallel processing using [rayon](https://github.com/rayon-rs/rayon). The implementation combines the classical Canny edge detection algorithm with advanced subpixel refinement techniques to achieve both speed and accuracy.

## Features

- **🚀 High Performance**: Parallel processing with rayon for multi-core optimization
- **🎯 Subpixel Accuracy**: Advanced parabolic fitting for precise edge localization
- **🔧 Optimized Algorithms**: Custom parallel Sobel operators and hysteresis thresholding
- **📊 Comprehensive Pipeline**: Complete Canny edge detection with subpixel refinement
- **🎨 Visualization Tools**: Built-in edge visualization capabilities
- **📈 Memory Efficient**: Optimized memory usage with pre-allocated buffers

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
subpixel-edge = "0.1.0"
image = "0.25"
```

## Quick Start

```rust
use image::open;
use subpixel_edge::{canny_based_subpixel_edges_optimized, visualize_edges};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load image
    let image = open("input.png")?.to_luma8();
    
    // Detect subpixel edges
    let edges = canny_based_subpixel_edges_optimized(
        &image,
        20.0,  // low_threshold
        80.0,  // high_threshold
        0.6    // edge_point_threshold
    );
    
    // Visualize results
    let visualization = visualize_edges(&image, &edges);
    visualization.save("edges_output.png")?;
    
    println!("Detected {} subpixel edge points", edges.len());
    Ok(())
}
```

## Algorithm Pipeline

### 1. Parallel Sobel Gradient Computation
- Custom parallel implementation of Sobel operators
- Optimized for multi-core systems using rayon
- Computes horizontal (Gx) and vertical (Gy) gradients simultaneously

### 2. Gradient Magnitude Calculation
- Parallel computation of gradient magnitude using L2 norm
- Essential for edge strength determination

### 3. Non-Maximum Suppression
- Thins edges to single-pixel width
- Directional comparison along gradient direction
- Preserves only local maxima in gradient magnitude

### 4. Hysteresis Thresholding
- Two-threshold approach for robust edge connectivity
- Parallel-optimized implementation with connected component analysis
- Strong edges (high threshold) anchor weak edges (low threshold)

### 5. Subpixel Refinement
- Parabolic fitting for subpixel accuracy
- Bilinear interpolation for smooth gradient sampling
- Quality validation and outlier rejection

## API Reference

### Core Functions

#### `canny_based_subpixel_edges_optimized`

```rust
pub fn canny_based_subpixel_edges_optimized(
    image: &GrayImage,
    low_threshold: f32,
    high_threshold: f32,
    edge_point_threshold: f32,
) -> Vec<(f32, f32)>
```

Main function for subpixel edge detection.

**Parameters:**
- `image`: Input grayscale image
- `low_threshold`: Lower threshold for hysteresis (10.0-50.0)
- `high_threshold`: Upper threshold for hysteresis (50.0-150.0)
- `edge_point_threshold`: Maximum subpixel offset (0.5-1.0)

**Returns:** Vector of subpixel edge coordinates (x, y)

#### `parallel_sobel_gradients`

```rust
pub fn parallel_sobel_gradients(image: &GrayImage) -> (Vec<f32>, Vec<f32>)
```

Computes Sobel gradients in parallel.

**Parameters:**
- `image`: Input grayscale image

**Returns:** Tuple of (horizontal_gradients, vertical_gradients)

#### `visualize_edges`

```rust
pub fn visualize_edges(image: &GrayImage, edge_points: &[(f32, f32)]) -> RgbImage
```

Creates visualization of detected edges.

**Parameters:**
- `image`: Original grayscale image
- `edge_points`: Detected subpixel edge coordinates

**Returns:** RGB image with edges highlighted in red

## Parameter Tuning Guide

### Threshold Selection

**Low Threshold (10.0 - 50.0)**
- Lower values: More edge pixels, higher noise sensitivity
- Higher values: Fewer edge pixels, more noise suppression
- Recommended starting point: 20.0

**High Threshold (50.0 - 150.0)**
- Should be 2-3x the low threshold
- Controls strong edge detection
- Recommended starting point: 80.0

**Edge Point Threshold (0.3 - 1.0)**
- Maximum allowed subpixel offset
- Lower values: More conservative, fewer edge points
- Higher values: More permissive, potential noise
- Recommended starting point: 0.6

### Image-Specific Tuning

**High Noise Images:**
```rust
let edges = canny_based_subpixel_edges_optimized(&image, 30.0, 90.0, 0.5);
```

**Clean Images:**
```rust
let edges = canny_based_subpixel_edges_optimized(&image, 15.0, 60.0, 0.8);
```

**Fine Detail Detection:**
```rust
let edges = canny_based_subpixel_edges_optimized(&image, 10.0, 40.0, 0.7);
```

## Performance Characteristics

### Benchmarks

On a typical 2800x1600 image with modern multi-core CPU:
- **Total Processing Time**: ~1.5 seconds
- **Speedup vs Serial**: 2-4x depending on core count
- **Memory Usage**: Efficient with pre-allocated buffers
- **Subpixel Accuracy**: ±0.1 pixel typical precision

### Optimization Features

- **Parallel Processing**: Utilizes all available CPU cores
- **Memory Efficiency**: Minimal allocations during processing
- **Cache Optimization**: Locality-aware algorithms
- **SIMD Potential**: Ready for future vectorization

## Examples

### Basic Edge Detection

```rust
use image::open;
use subpixel_edge::canny_based_subpixel_edges_optimized;

let image = open("photo.jpg")?.to_luma8();
let edges = canny_based_subpixel_edges_optimized(&image, 20.0, 80.0, 0.6);

for (x, y) in edges.iter().take(10) {
    println!("Edge at: ({:.3}, {:.3})", x, y);
}
```

### Batch Processing

```rust
use std::fs;
use image::open;
use subpixel_edge::canny_based_subpixel_edges_optimized;

fn process_directory(input_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(input_dir)? {
        let path = entry?.path();
        if let Some(ext) = path.extension() {
            if ext == "png" || ext == "jpg" {
                let image = open(&path)?.to_luma8();
                let edges = canny_based_subpixel_edges_optimized(&image, 20.0, 80.0, 0.6);
                println!("{}: {} edges", path.display(), edges.len());
            }
        }
    }
    Ok(())
}
```

### Custom Visualization

```rust
use image::{Rgb, RgbImage};
use subpixel_edge::{canny_based_subpixel_edges_optimized, visualize_edges};

fn custom_edge_overlay(
    image: &image::GrayImage, 
    edges: &[(f32, f32)]
) -> RgbImage {
    let mut canvas = visualize_edges(image, edges);
    
    // Add additional markers for strong edges
    for &(x, y) in edges.iter() {
        let ix = x as u32;
        let iy = y as u32;
        if ix > 0 && iy > 0 && ix < canvas.width()-1 && iy < canvas.height()-1 {
            // Add cross marker
            canvas.put_pixel(ix-1, iy, Rgb([255, 255, 0]));
            canvas.put_pixel(ix+1, iy, Rgb([255, 255, 0]));
            canvas.put_pixel(ix, iy-1, Rgb([255, 255, 0]));
            canvas.put_pixel(ix, iy+1, Rgb([255, 255, 0]));
        }
    }
    
    canvas
}
```

## Dependencies

- **image** (0.25): Image processing and I/O
- **imageproc** (0.25): Additional image processing utilities
- **rayon** (1.10): Data parallelism library
- **log** (0.4): Logging framework

## Development Dependencies

- **env_logger** (0.11): Environment-based logger for examples

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
git clone https://github.com/your-username/subpixel-edge.git
cd subpixel-edge
cargo build
cargo test
cargo run --example subpixel_edge_detection
```

## Acknowledgments

- Based on the Canny edge detection algorithm
- Inspired by classical computer vision literature
- Optimized using modern Rust parallel programming techniques

## Changelog

### v0.1.0
- Initial release
- Parallel Sobel gradient computation
- Optimized hysteresis thresholding
- Subpixel edge refinement
- Comprehensive documentation
- Example applications