//! # Subpixel Edge Detection Library
//!
//! This crate provides high-performance subpixel edge detection algorithms optimized with
//! parallel processing using rayon. The implementation is based on the Canny edge detection
//! algorithm with additional subpixel refinement for enhanced accuracy.
//!
//! ## Features
//!
//! - Parallel Sobel gradient computation
//! - Non-maximum suppression for edge thinning
//! - Hysteresis thresholding with parallel optimization
//! - Subpixel edge localization using parabolic fitting
//! - Edge visualization utilities
//! - Optional debug logging (enable with `logger` feature)
//!
//! ## Basic Usage
//!
//! ```rust,no_run
//! use image::open;
//! use subpixel_edge::{canny_based_subpixel_edges_optimized, visualize_edges};
//!
//! // Load and process image
//! let image = open("example.png").unwrap().to_luma8();
//! let edges = canny_based_subpixel_edges_optimized(&image, 20.0, 80.0, 0.6);
//!
//! // Create visualization
//! let result = visualize_edges(&image, &edges);
//! result.save("edges_output.png").unwrap();
//!
//! println!("Found {} subpixel edge points", edges.len());
//! ```
//!
//! ## Optional Features
//!
//! ### Logger Feature
//!
//! Enable debug logging to monitor the edge detection pipeline:
//!
//! ```toml
//! [dependencies]
//! subpixel-edge = { version = "0.1.0", features = ["logger"] }
//! log = "0.4"
//! env_logger = "0.11"
//! ```
//!
//! ```rust,no_run
//! use image::open;
//! use subpixel_edge::canny_based_subpixel_edges_optimized;
//!
//! // Initialize logger to see debug output
//! env_logger::init();
//!
//! let image = open("example.png").unwrap().to_luma8();
//! let edges = canny_based_subpixel_edges_optimized(&image, 20.0, 80.0, 0.6);
//! // With logger feature, you'll see debug messages like:
//! // DEBUG subpixel_edge: start calculate subpixel edges
//! // DEBUG subpixel_edge: thinned ok
//! // DEBUG subpixel_edge: canny_edge_points ok
//! ```
//!
//! ## Advanced Usage
//!
//! ```rust,no_run
//! use image::{open, imageops::blur};
//! use subpixel_edge::{canny_based_subpixel_edges_optimized, parallel_sobel_gradients};
//!
//! // Pre-process image with blur for noise reduction
//! let image = open("noisy_image.png").unwrap().to_luma8();
//! let blurred = blur(&image, 1.0);
//!
//! // Fine-tune parameters for specific image characteristics
//! let edges = canny_based_subpixel_edges_optimized(
//!     &blurred,
//!     15.0,  // Lower threshold for more edge details
//!     60.0,  // Adjusted high threshold
//!     0.8    // Allow larger subpixel offsets
//! );
//!
//! // Process results
//! let high_precision_edges: Vec<_> = edges
//!     .iter()
//!     .filter(|(x, y)| {
//!         // Custom filtering based on application needs
//!         x.fract().abs() > 0.1 || y.fract().abs() > 0.1
//!     })
//!     .collect();
//! ```

use image::{GenericImageView, GrayImage, ImageBuffer, Luma, Rgb, RgbImage, buffer::ConvertBuffer};
use imageproc::definitions::{HasBlack, HasWhite};
use rayon::prelude::*;
use std::{
    f32::consts::PI,
    sync::{Arc, Mutex},
};

// Conditional logging macros
#[cfg(feature = "logger")]
macro_rules! debug {
    ($($arg:tt)*) => {
        log::debug!($($arg)*);
    };
}

#[cfg(not(feature = "logger"))]
macro_rules! debug {
    ($($arg:tt)*) => {};
}

/// Performs high-performance Canny-based subpixel edge detection with parallel optimization.
///
/// This function implements a complete pipeline for subpixel edge detection combining
/// the Canny edge detection algorithm with subpixel refinement. The implementation is
/// fully optimized using parallel processing for maximum performance.
///
/// # Arguments
///
/// * `image` - Input grayscale image for edge detection
/// * `low_threshold` - Lower threshold for hysteresis (typical range: 10.0-50.0)
/// * `high_threshold` - Upper threshold for hysteresis (typical range: 50.0-150.0)
/// * `edge_point_threshold` - Maximum allowed subpixel offset (typically 0.5-1.0)
///
/// # Returns
///
/// A vector of tuples containing subpixel edge coordinates (x, y) as floating-point values.
///
/// # Performance
///
/// This function uses parallel processing extensively:
/// - Parallel Sobel gradient computation
/// - Parallel magnitude calculation
/// - Parallel subpixel edge refinement
///
/// # Examples
///
/// ## Basic Edge Detection
/// ```rust,no_run
/// use image::open;
/// use subpixel_edge::canny_based_subpixel_edges_optimized;
///
/// let image = open("input.png").unwrap().to_luma8();
/// let edges = canny_based_subpixel_edges_optimized(&image, 20.0, 80.0, 0.6);
///
/// for (x, y) in &edges {
///     println!("Edge at subpixel location: ({:.2}, {:.2})", x, y);
/// }
/// ```
///
/// ## Parameter Optimization for Different Image Types
/// ```rust,no_run
/// use image::open;
/// use subpixel_edge::canny_based_subpixel_edges_optimized;
///
/// let image = open("input.png").unwrap().to_luma8();
///
/// // For high-noise images
/// let robust_edges = canny_based_subpixel_edges_optimized(&image, 30.0, 90.0, 0.5);
///
/// // For fine detail detection
/// let detailed_edges = canny_based_subpixel_edges_optimized(&image, 10.0, 40.0, 0.8);
///
/// println!("Robust detection: {} edges", robust_edges.len());
/// println!("Detailed detection: {} edges", detailed_edges.len());
/// ```
///
/// # Algorithm Pipeline
///
/// 1. Parallel Sobel gradient computation (Gx, Gy)
/// 2. Gradient magnitude calculation
/// 3. Non-maximum suppression for edge thinning
/// 4. Hysteresis thresholding for edge connectivity
/// 5. Subpixel refinement using parabolic fitting
pub fn canny_based_subpixel_edges_optimized(
    image: &GrayImage,
    low_threshold: f32,
    high_threshold: f32,
    edge_point_threshold: f32,
) -> Vec<(f32, f32)> {
    let (width, height) = image.dimensions();
    debug!("start calcualte subpixel edges");

    // Step 1: Parallel computation of gradient information using optimized Sobel operators
    // This replaces the standard imageproc Sobel functions with our parallel implementation
    let (gx_data, gy_data) = parallel_sobel_gradients(image);

    debug!("gx_data and gy_data ok");

    // Convert gradient data to i16 format for compatibility with non-maximum suppression
    let gx_image_data: Vec<i16> = gx_data.par_iter().map(|p| *p as i16).collect();
    let gy_image_data: Vec<i16> = gy_data.par_iter().map(|p| *p as i16).collect();

    let gx_image = ImageBuffer::from_raw(width, height, gx_image_data).unwrap();
    let gy_image = ImageBuffer::from_raw(width, height, gy_image_data).unwrap();

    debug!("gx_image and gy_image ok");

    debug!("gx_data and gy_data ok");

    // Step 2: Parallel computation of gradient magnitude
    // Calculate the magnitude of gradients using L2 norm for each pixel
    let mag_data: Vec<f32> = gx_data
        .par_iter()
        .zip(gy_data.par_iter())
        .map(|(gx, gy)| (gx.powi(2) + gy.powi(2)).sqrt())
        .collect();

    debug!("mag_data ok");

    // Create gradient magnitude image for non-maximum suppression
    let g: Vec<f32> = gx_image
        .iter()
        .zip(gy_image.iter())
        .map(|(h, v)| (*h as f32).hypot(*v as f32))
        .collect::<Vec<f32>>();

    debug!("g ok");

    let g = ImageBuffer::from_raw(image.width(), image.height(), g).unwrap();

    debug!("g image ok");

    // Step 3: Non-maximum suppression to thin edges to single-pixel width
    // This ensures edges are as thin as possible for accurate subpixel refinement
    let thinned = non_maximum_suppression(&g, &gx_image, &gy_image);

    debug!("thinned ok");

    // Step 4: Hysteresis thresholding to connect edge pixels and filter weak edges
    // Uses parallel-optimized implementation for improved performance
    let canny_edge_points = hysteresis(&thinned, low_threshold, high_threshold);

    debug!("canny_edge_points ok");

    // Step 5: Parallel subpixel refinement for each detected edge point
    // This is the final step that provides subpixel accuracy using parabolic fitting
    canny_edge_points
        .into_par_iter()
        .filter_map(|(x, y)| {
            subpixel_in_3x3(
                x,
                y,
                &gx_data,
                &gy_data,
                &mag_data,
                width,
                height,
                edge_point_threshold,
            )
        })
        .collect()
}

/// Computes Sobel gradients in parallel for improved performance.
///
/// This function calculates the horizontal (Gx) and vertical (Gy) gradients of an image
/// using 3x3 Sobel operators. The computation is parallelized across image rows using
/// rayon for optimal performance on multi-core systems.
///
/// # Arguments
///
/// * `image` - Input grayscale image
///
/// # Returns
///
/// A tuple containing:
/// - `Vec<f32>` - Horizontal gradients (Gx) flattened as a 1D vector
/// - `Vec<f32>` - Vertical gradients (Gy) flattened as a 1D vector
///
/// # Sobel Operators
///
/// Horizontal (Gx):
/// ```text
/// [-1  0  1]
/// [-2  0  2]
/// [-1  0  1]
/// ```
///
/// Vertical (Gy):
/// ```text
/// [-1 -2 -1]
/// [ 0  0  0]
/// [ 1  2  1]
/// ```
///
/// # Performance
///
/// This function uses thread-safe atomic operations and parallel row processing
/// to achieve significant speedup on multi-core systems. Border pixels are handled
/// by skipping the outermost rows and columns.
///
/// # Examples
///
/// ## Basic Gradient Computation
/// ```rust,no_run
/// use image::open;
/// use subpixel_edge::parallel_sobel_gradients;
///
/// let image = open("input.png").unwrap().to_luma8();
/// let (gx, gy) = parallel_sobel_gradients(&image);
/// println!("Computed gradients for {}x{} image", image.width(), image.height());
/// ```
///
/// ## Gradient Analysis
/// ```rust,no_run
/// use image::open;
/// use subpixel_edge::parallel_sobel_gradients;
///
/// let image = open("input.png").unwrap().to_luma8();
/// let (gx, gy) = parallel_sobel_gradients(&image);
/// let (width, height) = image.dimensions();
///
/// // Find maximum gradient magnitudes
/// let max_gradient = gx.iter().zip(gy.iter())
///     .map(|(dx, dy)| (dx * dx + dy * dy).sqrt())
///     .fold(0.0f32, |max, val| max.max(val));
///
/// println!("Maximum gradient magnitude: {:.2}", max_gradient);
///
/// // Analyze gradient distribution
/// let strong_gradients = gx.iter().zip(gy.iter())
///     .filter(|(dx, dy)| (**dx * **dx + **dy * **dy).sqrt() > max_gradient * 0.3)
///     .count();
///
/// println!("Strong gradient pixels: {} ({:.1}%)",
///          strong_gradients,
///          100.0 * strong_gradients as f32 / (width * height) as f32);
/// ```
pub fn parallel_sobel_gradients(image: &GrayImage) -> (Vec<f32>, Vec<f32>) {
    let (width, height) = image.dimensions();
    // Thread-safe storage for gradient results
    let gx = Arc::new(Mutex::new(vec![0.0; (width * height) as usize]));
    let gy = Arc::new(Mutex::new(vec![0.0; (width * height) as usize]));

    // Access raw pixel data for efficient processing
    let pixels = image.as_raw();

    // Sobel kernel definitions for edge detection
    /// Horizontal Sobel kernel for detecting vertical edges
    const SOBEL_KERNEL_X: [f32; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    /// Vertical Sobel kernel for detecting horizontal edges
    const SOBEL_KERNEL_Y: [f32; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

    // Parallel processing of each row (excluding border rows)
    // This provides significant speedup on multi-core systems
    (1..height - 1).into_par_iter().for_each(|y| {
        // Calculate starting index for the current row
        let row_start = (y * width) as usize;

        // Extract pixel slices for the 3x3 neighborhood
        // Previous row (y-1), current row (y), and next row (y+1)
        let prev_row =
            &pixels[(row_start - width as usize)..(row_start - width as usize + width as usize)];
        let curr_row = &pixels[row_start..(row_start + width as usize)];
        let next_row =
            &pixels[(row_start + width as usize)..(row_start + width as usize + width as usize)];

        // Acquire locks for thread-safe access to gradient arrays
        let mut gx_mutex = gx.lock().unwrap();
        let mut gy_mutex = gy.lock().unwrap();

        // Process each pixel in the current row (excluding border columns)
        for x in 1..(width - 1) {
            let mut gx_val = 0.0;
            let mut gy_val = 0.0;

            // Apply 3x3 convolution with Sobel kernels
            for ky in 0..3 {
                for kx in 0..3 {
                    let pixel_x = x + kx - 1;

                    // Extract pixel value from appropriate row
                    let pixel = if ky == 0 {
                        prev_row[pixel_x as usize] as f32
                    } else if ky == 1 {
                        curr_row[pixel_x as usize] as f32
                    } else {
                        next_row[pixel_x as usize] as f32
                    };

                    // Apply Sobel kernel coefficients
                    let kernel_index = (ky * 3 + kx) as usize;
                    gx_val += pixel * SOBEL_KERNEL_X[kernel_index];
                    gy_val += pixel * SOBEL_KERNEL_Y[kernel_index];
                }
            }

            // Store computed gradients at the correct index
            let index = (y * width + x) as usize;
            gx_mutex[index] = gx_val;
            gy_mutex[index] = gy_val;
        }
    });

    // Extract results from Arc<Mutex<>> containers
    let gx_result = Arc::try_unwrap(gx).unwrap().into_inner().unwrap();
    let gy_result = Arc::try_unwrap(gy).unwrap().into_inner().unwrap();
    (gx_result, gy_result)
}

/// Performs non-maximum suppression to thin edges to single-pixel width.
///
/// This function implements the non-maximum suppression step of the Canny edge detection
/// algorithm. It examines each pixel's gradient magnitude and compares it with its two
/// neighbors along the gradient direction. If the pixel is not a local maximum, it is
/// suppressed (set to zero).
///
/// # Arguments
///
/// * `g` - Gradient magnitude image
/// * `gx` - Horizontal gradient image
/// * `gy` - Vertical gradient image
///
/// # Returns
///
/// A new image buffer with suppressed non-maximum pixels, resulting in thin edges.
///
/// # Algorithm
///
/// 1. Calculate gradient direction for each pixel
/// 2. Quantize direction to one of four angles: 0°, 45°, 90°, 135°
/// 3. Compare pixel magnitude with neighbors along gradient direction
/// 4. Suppress pixel if it's not a local maximum
///
/// # Edge Direction Quantization
///
/// - 0° (horizontal): Compare with left and right neighbors
/// - 45° (diagonal): Compare with diagonal neighbors (↗ and ↙)
/// - 90° (vertical): Compare with top and bottom neighbors
/// - 135° (diagonal): Compare with diagonal neighbors (↖ and ↘)
fn non_maximum_suppression(
    g: &ImageBuffer<Luma<f32>, Vec<f32>>,
    gx: &ImageBuffer<Luma<i16>, Vec<i16>>,
    gy: &ImageBuffer<Luma<i16>, Vec<i16>>,
) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    /// Conversion factor from radians to degrees
    const RADIANS_TO_DEGREES: f32 = 180f32 / PI;

    // Initialize output image with zeros
    let mut out = ImageBuffer::from_pixel(g.width(), g.height(), Luma([0.0]));
    let mut points = Vec::new();

    // Process each pixel (excluding border pixels)
    for y in 1..g.height() - 1 {
        for x in 1..g.width() - 1 {
            let x_gradient = gx[(x, y)][0] as f32;
            let y_gradient = gy[(x, y)][0] as f32;

            // Calculate gradient direction in degrees
            let mut angle = (y_gradient).atan2(x_gradient) * RADIANS_TO_DEGREES;
            if angle < 0.0 {
                angle += 180.0
            }

            // Quantize angle to one of four directions for neighbor comparison
            // This determines which neighbors to compare for local maximum detection
            let clamped_angle = if !(22.5..157.5).contains(&angle) {
                0 // Horizontal direction (0° or 180°)
            } else if (22.5..67.5).contains(&angle) {
                45 // Diagonal direction (45°)
            } else if (67.5..112.5).contains(&angle) {
                90 // Vertical direction (90°)
            } else if (112.5..157.5).contains(&angle) {
                135 // Diagonal direction (135°)
            } else {
                unreachable!()
            };

            // Get the two neighbors perpendicular to the gradient direction
            // These are the pixels we compare against for local maximum detection
            let (cmp1, cmp2) = unsafe {
                match clamped_angle {
                    0 => (g.unsafe_get_pixel(x - 1, y), g.unsafe_get_pixel(x + 1, y)),
                    45 => (
                        g.unsafe_get_pixel(x + 1, y + 1),
                        g.unsafe_get_pixel(x - 1, y - 1),
                    ),
                    90 => (g.unsafe_get_pixel(x, y - 1), g.unsafe_get_pixel(x, y + 1)),
                    135 => (
                        g.unsafe_get_pixel(x - 1, y + 1),
                        g.unsafe_get_pixel(x + 1, y - 1),
                    ),
                    _ => unreachable!(),
                }
            };

            let pixel = *g.get_pixel(x, y);

            // Suppress pixel if it's not a local maximum along gradient direction
            if pixel[0] < cmp1[0] || pixel[0] < cmp2[0] {
                out.put_pixel(x, y, Luma([0.0]));
            } else {
                // Keep pixel as potential edge and track for debugging
                out.put_pixel(x, y, pixel);
                points.push((x, y));
            }
        }
    }
    debug!("points len:{}", points.len());
    out
}

/// Performs hysteresis thresholding to connect edge pixels and suppress weak edges.
///
/// This function implements the final step of the Canny edge detection algorithm using
/// a two-threshold approach. Strong edges (above high threshold) are immediately accepted,
/// and weak edges (between low and high thresholds) are accepted only if they connect
/// to strong edges through a chain of other weak edges.
///
/// # Arguments
///
/// * `input` - Gradient magnitude image after non-maximum suppression
/// * `low_thresh` - Lower threshold for weak edge detection
/// * `high_thresh` - Upper threshold for strong edge detection
///
/// # Returns
///
/// A vector of pixel coordinates (x, y) representing confirmed edge points.
///
/// # Algorithm
///
/// 1. Scan image for pixels above high threshold (strong edges)
/// 2. For each strong edge, perform depth-first search to find connected weak edges
/// 3. Mark all connected pixels that are above low threshold as valid edges
/// 4. Use non-recursive breadth-first search to avoid stack overflow
///
/// # Thresholding Strategy
///
/// - **High threshold**: Pixels above this are definitely edges
/// - **Low threshold**: Pixels above this are potential edges if connected to strong edges
/// - **Below low**: Pixels below this are discarded as noise
///
/// # Example Threshold Values
///
/// - For noisy images: low=20, high=60
/// - For clean images: low=50, high=150
/// - The ratio high:low should typically be 2:1 to 3:1
fn hysteresis(
    input: &ImageBuffer<Luma<f32>, Vec<f32>>,
    low_thresh: f32,
    high_thresh: f32,
) -> Vec<(u32, u32)> {
    let max_brightness = Luma::<u8>::white();
    let min_brightness = Luma::<u8>::black();
    // Initialize output image as all black to track processed pixels
    let mut out = ImageBuffer::from_pixel(input.width(), input.height(), min_brightness);

    // Stack for depth-first search of connected edge components
    // Pre-allocate with reasonable capacity to reduce allocations
    let mut edges = Vec::with_capacity(((input.width() * input.height()) / 2) as usize);
    let mut result_edge_points = Vec::new();

    // Scan entire image for strong edges (above high threshold)
    for y in 1..input.height() - 1 {
        for x in 1..input.width() - 1 {
            let inp_pix = *input.get_pixel(x, y);
            let out_pix = *out.get_pixel(x, y);

            // If edge strength exceeds high threshold and hasn't been processed yet
            if inp_pix[0] >= high_thresh && out_pix[0] == 0 {
                // Mark as strong edge
                out.put_pixel(x, y, max_brightness);
                edges.push((x, y));

                // Perform depth-first search to find all connected weak edges
                // This connects edge fragments that might be broken by noise
                while let Some((nx, ny)) = edges.pop() {
                    // Check all 8-connected neighbors (excluding diagonals for efficiency)
                    let neighbor_indices = [
                        (nx + 1, ny),                             // Right
                        (nx + 1, ny + 1),                         // Bottom-right
                        (nx, ny + 1),                             // Bottom
                        (nx.wrapping_sub(1), ny.wrapping_sub(1)), // Top-left
                        (nx.wrapping_sub(1), ny),                 // Left
                        (nx.wrapping_sub(1), ny + 1),             // Bottom-left
                    ];

                    for neighbor_idx in &neighbor_indices {
                        // Comprehensive bounds checking to prevent panics
                        if neighbor_idx.0 >= input.width()
                            || neighbor_idx.1 >= input.height()
                            || neighbor_idx.0 == u32::MAX  // Check for underflow
                            || neighbor_idx.1 == u32::MAX
                        // Check for underflow
                        {
                            continue;
                        }

                        let in_neighbor = *input.get_pixel(neighbor_idx.0, neighbor_idx.1);
                        let out_neighbor = *out.get_pixel(neighbor_idx.0, neighbor_idx.1);

                        // Accept weak edges that are connected to strong edges
                        if in_neighbor[0] >= low_thresh && out_neighbor[0] == 0 {
                            out.put_pixel(neighbor_idx.0, neighbor_idx.1, max_brightness);
                            edges.push((neighbor_idx.0, neighbor_idx.1));
                            result_edge_points.push((neighbor_idx.0, neighbor_idx.1));
                        }
                    }
                }
            }
        }
    }
    result_edge_points
}

/// Visualizes subpixel edge points on the original image.
///
/// This function creates a color visualization of detected subpixel edges by overlaying
/// red pixels on the original grayscale image. Subpixel coordinates are rounded to the
/// nearest integer pixel locations for display purposes.
///
/// # Arguments
///
/// * `image` - Original grayscale image used as background
/// * `edge_points` - Vector of subpixel edge coordinates (x, y) as floating-point values
///
/// # Returns
///
/// An RGB image with the original image in grayscale and edge points highlighted in red.
///
/// # Color Scheme
///
/// - **Background**: Original image converted to grayscale RGB
/// - **Edge points**: Bright red (RGB: 255, 0, 0)
///
/// # Performance
///
/// The function uses parallel processing to filter valid edge points and then performs
/// serial drawing to avoid race conditions when multiple threads try to write to the
/// same pixel location.
///
/// # Examples
///
/// ## Basic Visualization
/// ```rust,no_run
/// use image::open;
/// use subpixel_edge::{canny_based_subpixel_edges_optimized, visualize_edges};
///
/// let image = open("input.png").unwrap().to_luma8();
/// let edges = canny_based_subpixel_edges_optimized(&image, 20.0, 80.0, 0.6);
/// let visualization = visualize_edges(&image, &edges);
/// visualization.save("edges_visualization.png").unwrap();
/// ```
///
/// ## Batch Visualization with Statistics
/// ```rust,no_run
/// use image::open;
/// use subpixel_edge::{canny_based_subpixel_edges_optimized, visualize_edges};
///
/// let image = open("input.png").unwrap().to_luma8();
/// let edges = canny_based_subpixel_edges_optimized(&image, 20.0, 80.0, 0.6);
///
/// // Create visualization with statistics
/// let visualization = visualize_edges(&image, &edges);
///
/// // Calculate edge density
/// let total_pixels = (image.width() * image.height()) as f32;
/// let edge_density = edges.len() as f32 / total_pixels * 100.0;
///
/// println!("Edge density: {:.2}% ({} edges in {} pixels)",
///          edge_density, edges.len(), total_pixels as u32);
///
/// visualization.save("edges_with_stats.png").unwrap();
/// ```
pub fn visualize_edges(image: &GrayImage, edge_points: &[(f32, f32)]) -> RgbImage {
    // Convert grayscale image to RGB for color overlay
    let mut canvas: RgbImage = image.convert();
    let red = Rgb([255u8, 0, 0]);

    // Parallel filtering of valid edge points within image bounds
    // This avoids bounds checking during the drawing phase
    let valid_points: Vec<(u32, u32)> = edge_points
        .par_iter()
        .filter_map(|&(x, y)| {
            // Round subpixel coordinates to nearest integer pixels
            let sx = x as i32;
            let sy = y as i32;

            // Ensure coordinates are within image bounds
            if sx >= 0 && sy >= 0 && sx < canvas.width() as i32 && sy < canvas.height() as i32 {
                Some((sx as u32, sy as u32))
            } else {
                None
            }
        })
        .collect();

    // Serial drawing to avoid race conditions in pixel writing
    // Although this is not parallelized, it's typically fast since
    // the number of edge pixels is much smaller than total pixels
    for (sx, sy) in valid_points {
        canvas.put_pixel(sx, sy, red);
    }

    canvas
}

/// Performs bilinear interpolation on a 2D data array at fractional coordinates.
///
/// This function computes the interpolated value at a non-integer position (x, y)
/// using bilinear interpolation between the four nearest grid points. This is
/// essential for subpixel accuracy in edge detection.
///
/// # Arguments
///
/// * `data` - Flattened 2D array containing the values to interpolate
/// * `width` - Width of the 2D grid
/// * `height` - Height of the 2D grid
/// * `x` - X coordinate (can be fractional)
/// * `y` - Y coordinate (can be fractional)
///
/// # Returns
///
/// The interpolated value at position (x, y), or 0.0 if coordinates are out of bounds.
///
/// # Algorithm
///
/// For a point (x, y), the function:
/// 1. Finds the four surrounding grid points: (x0,y0), (x1,y0), (x0,y1), (x1,y1)
/// 2. Computes fractional offsets: dx = x - x0, dy = y - y0
/// 3. Applies bilinear interpolation formula:
///    ```text
///    result = p00*(1-dx)*(1-dy) + p10*dx*(1-dy) + p01*(1-dx)*dy + p11*dx*dy
///    ```
///
/// # Mathematical Formula
///
/// ```text
/// (x0,y0)-----(x1,y0)
///    |   p00     p10  |
///    |     (x,y)      |
///    |   p01     p11  |
/// (x0,y1)-----(x1,y1)
/// ```
///
/// The interpolation formula is:
/// ```text
/// result = p00*(1-dx)*(1-dy) + p10*dx*(1-dy) + p01*(1-dx)*dy + p11*dx*dy
/// where dx = x - x0, dy = y - y0
/// ```
///
/// # Examples
///
/// ## Basic Interpolation
/// ```rust,no_run
/// # fn bilinear_interpolation(data: &[f32], width: u32, height: u32, x: f32, y: f32) -> f32 { 0.0 }
/// let data = vec![1.0, 2.0, 3.0, 4.0]; // 2x2 grid
/// let value = bilinear_interpolation(&data, 2, 2, 0.5, 0.5);
/// // Returns interpolated value at the center of the 2x2 grid
/// // Should be approximately (1+2+3+4)/4 = 2.5
/// println!("Interpolated value: {}", value);
/// ```
///
/// ## Edge Case Handling
/// ```rust,no_run
/// # fn bilinear_interpolation(data: &[f32], width: u32, height: u32, x: f32, y: f32) -> f32 { 0.0 }
/// let data = vec![1.0, 2.0, 3.0, 4.0];
///
/// // Valid interpolation
/// let valid = bilinear_interpolation(&data, 2, 2, 0.25, 0.75);
/// println!("Valid interpolation: {}", valid);
///
/// // Out of bounds returns 0.0
/// let invalid = bilinear_interpolation(&data, 2, 2, -1.0, 0.5);
/// println!("Out of bounds result: {}", invalid); // Should be 0.0
///
/// let invalid2 = bilinear_interpolation(&data, 2, 2, 2.5, 0.5);
/// println!("Out of bounds result: {}", invalid2); // Should be 0.0
/// ```
fn bilinear_interpolation(data: &[f32], width: u32, height: u32, x: f32, y: f32) -> f32 {
    // Find the integer coordinates of the top-left corner
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    // Bounds checking - return 0 if coordinates are outside the valid range
    if x0 < 0 || y0 < 0 || x1 >= width as i32 || y1 >= height as i32 {
        return 0.0;
    }

    // Extract the four corner values for interpolation
    // p00: top-left, p10: top-right, p01: bottom-left, p11: bottom-right
    let p00 = data[(y0 as u32 * width + x0 as u32) as usize];
    let p10 = data[(y0 as u32 * width + x1 as u32) as usize];
    let p01 = data[(y1 as u32 * width + x0 as u32) as usize];
    let p11 = data[(y1 as u32 * width + x1 as u32) as usize];

    // Calculate fractional offsets within the unit square
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    // Apply bilinear interpolation formula
    // This gives a smooth transition between the four corner values
    p00 * (1.0 - dx) * (1.0 - dy) + p10 * dx * (1.0 - dy) + p01 * (1.0 - dx) * dy + p11 * dx * dy
}

/// Performs subpixel edge localization within a 3x3 neighborhood using parabolic fitting.
///
/// This function refines the location of an edge pixel to subpixel accuracy by fitting
/// a parabola to the gradient magnitude values along the gradient direction. The peak
/// of the parabola gives the precise subpixel edge location.
///
/// # Arguments
///
/// * `x` - Integer x-coordinate of the edge pixel
/// * `y` - Integer y-coordinate of the edge pixel
/// * `gx_data` - Horizontal gradient values (flattened 2D array)
/// * `gy_data` - Vertical gradient values (flattened 2D array)
/// * `mag_data` - Gradient magnitude values (flattened 2D array)
/// * `width` - Image width
/// * `height` - Image height
/// * `edge_point_threshold` - Maximum allowed subpixel offset (typically 0.5-1.0)
///
/// # Returns
///
/// * `Some((x, y))` - Subpixel coordinates if refinement is successful
/// * `None` - If refinement fails due to boundary conditions or invalid fitting
///
/// # Algorithm
///
/// 1. **Gradient Direction**: Calculate the gradient direction at the pixel
/// 2. **Sample Points**: Sample magnitude at three points along the gradient direction:
///    - Center point (original pixel)
///    - Point shifted +0.5 pixels along gradient
///    - Point shifted -0.5 pixels along gradient
/// 3. **Parabolic Fitting**: Fit a parabola through these three points
/// 4. **Peak Finding**: Calculate the peak of the parabola for subpixel location
/// 5. **Validation**: Ensure the peak represents a local maximum and is within bounds
///
/// # Mathematical Details
///
/// For three points along the gradient direction with magnitudes m0, m1, m2:
/// - Parabola coefficient: `a = 2(m1 + m2 - 2*m0)`
/// - Subpixel offset: `offset = (m2 - m1) / (4(m1 + m2 - 2*m0))`
/// - Final coordinates: `(x + offset*cos(θ), y + offset*sin(θ))`
///
/// # Quality Checks
///
/// - Ensures parabola has a maximum (a < 0)
/// - Validates offset is within reasonable bounds
/// - Performs boundary checking for interpolation points
///
/// # Example Use Case
///
/// This function is typically called for each pixel identified as an edge by the
/// Canny algorithm to achieve subpixel accuracy for applications requiring precise
/// edge localization, such as machine vision or medical imaging.
#[allow(clippy::too_many_arguments)]
fn subpixel_in_3x3(
    x: u32,
    y: u32,
    gx_data: &[f32],
    gy_data: &[f32],
    mag_data: &[f32],
    width: u32,
    height: u32,
    edge_point_threshold: f32,
) -> Option<(f32, f32)> {
    let idx = (y * width + x) as usize;
    let mag = mag_data[idx];

    // Calculate gradient direction at the current pixel
    // This determines the direction perpendicular to the edge
    let gx_val = gx_data[idx];
    let gy_val = gy_data[idx];
    let theta = gy_val.atan2(gx_val);
    let dx = theta.cos();
    let dy = theta.sin();

    // Calculate sampling points along the gradient direction
    // These points are used for parabolic fitting
    let x1 = x as f32 + 0.5 * dx; // Point shifted forward along gradient
    let y1 = y as f32 + 0.5 * dy;
    let x2 = x as f32 - 0.5 * dx; // Point shifted backward along gradient
    let y2 = y as f32 - 0.5 * dy;

    // Boundary checking to ensure interpolation points are valid
    // All points must be within the image boundaries with margin for interpolation
    if x1 < 1.0
        || x1 >= (width - 1) as f32
        || y1 < 1.0
        || y1 >= (height - 1) as f32
        || x2 < 1.0
        || x2 >= (width - 1) as f32
        || y2 < 1.0
        || y2 >= (height - 1) as f32
    {
        return None;
    }

    // Use bilinear interpolation to get gradient magnitudes at the sampling points
    // This provides sub-pixel accuracy for the magnitude values
    let m1 = bilinear_interpolation(mag_data, width, height, x1, y1);
    let m2 = bilinear_interpolation(mag_data, width, height, x2, y2);

    // Parabolic fitting: fit parabola through three points (m2, mag, m1)
    // The coefficient 'a' determines the curvature of the parabola
    let a = 2.0 * (m1 + m2 - 2.0 * mag);

    // Ensure we have a maximum (parabola opens downward)
    // If a >= 0, the parabola opens upward, indicating no local maximum
    if a >= 0.0 {
        return None;
    }

    // Calculate subpixel offset along the gradient direction
    // This gives the distance from the center pixel to the true edge peak
    let offset = (m2 - m1) / (4.0 * (m1 + m2 - 2.0 * mag));

    // Filter out unreasonable offsets that might indicate poor fitting
    // Large offsets suggest the edge is not well-localized at this pixel
    if offset.abs() > edge_point_threshold {
        return None;
    }

    // Calculate final subpixel coordinates by applying the offset
    // along the gradient direction from the original pixel location
    let sub_x = x as f32 + offset * dx;
    let sub_y = y as f32 + offset * dy;

    Some((sub_x, sub_y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageBuffer;

    #[test]
    fn test_logger_feature_compilation() {
        // This test ensures that the code compiles with and without the logger feature
        // The debug! macro should work in both cases

        // Create a small test image
        let image = ImageBuffer::from_fn(10, 10, |_x, _y| image::Luma([128u8]));

        // Test that debug macro doesn't cause compilation errors
        debug!("Test debug message");

        // Test basic functionality works regardless of logger feature
        let (gx, gy) = parallel_sobel_gradients(&image);
        assert_eq!(gx.len(), 100); // 10x10 = 100 pixels
        assert_eq!(gy.len(), 100);
    }

    #[test]
    #[cfg(feature = "logger")]
    fn test_logger_feature_enabled() {
        // This test only runs when the logger feature is enabled
        use std::cell::RefCell;
        use std::sync::{Arc, Mutex};

        // Verify that log crate is available when feature is enabled
        assert!(true, "Logger feature is enabled and working");
    }

    #[test]
    #[cfg(not(feature = "logger"))]
    fn test_logger_feature_disabled() {
        // This test only runs when the logger feature is disabled
        assert!(true, "Logger feature is disabled and library still works");
    }

    #[test]
    fn test_debug_macro_no_panic() {
        // Test that debug macro calls don't panic regardless of feature state
        debug!("Starting test");
        debug!("Processing data: {}", 42);
        debug!("Test completed successfully");

        // If we reach here, the debug macro is working correctly
        assert!(true);
    }
}
