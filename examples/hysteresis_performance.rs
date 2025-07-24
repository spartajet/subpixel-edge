use env_logger::Builder;
use image::{ImageBuffer, Luma, imageops::blur, open};
use log::info;

use subpixel_edge::compare_hysteresis_performance;

fn main() {
    Builder::from_default_env().format_timestamp_nanos().init();

    // Load test image
    let img = open("../test_image/edge.png").unwrap().to_luma8();
    let blur_img = blur(&img, 1.0);

    info!("Loading image: {}x{}", img.width(), img.height());

    // Create synthetic magnitude data with strong edges for better performance testing
    let magnitude: ImageBuffer<Luma<f32>, Vec<f32>> =
        ImageBuffer::from_fn(blur_img.width(), blur_img.height(), |x, y| {
            let center_x = blur_img.width() / 2;
            let center_y = blur_img.height() / 2;

            // Create several edge patterns for more realistic testing
            let mut mag: f32 = 0.0;

            // Horizontal edge
            if (y as i32 - center_y as i32).abs() < 3 {
                mag = mag.max(100.0);
            }

            // Vertical edge
            if (x as i32 - center_x as i32).abs() < 3 {
                mag = mag.max(90.0);
            }

            // Diagonal edges
            if ((x as i32 - y as i32).abs() < 5)
                || ((x as i32 + y as i32 - blur_img.width() as i32).abs() < 5)
            {
                mag = mag.max(70.0);
            }

            // Some noise and weaker edges
            if (x + y) % 10 == 0 {
                mag = mag.max(30.0);
            }

            // Add some randomness based on original image
            let pixel_val = blur_img.get_pixel(x, y)[0] as f32;
            if pixel_val > 128.0 {
                mag = mag.max(50.0);
            }

            Luma([mag])
        });

    info!("Starting hysteresis performance comparison...");

    // Parameters for hysteresis - adjusted for synthetic data
    let low_thresh = 25.0;
    let high_thresh = 60.0;

    // Run performance comparison multiple times for better accuracy
    let mut original_times = Vec::new();
    let mut optimized_times = Vec::new();
    let runs = 5;

    for i in 0..runs {
        info!("Run {}/{}", i + 1, runs);

        let (original_time, optimized_time, original_result, optimized_result) =
            compare_hysteresis_performance(&magnitude, low_thresh, high_thresh);

        original_times.push(original_time);
        optimized_times.push(optimized_time);

        info!(
            "  Original: {:?}, Optimized: {:?}, Speedup: {:.2}x",
            original_time,
            optimized_time,
            original_time.as_secs_f64() / optimized_time.as_secs_f64()
        );

        info!(
            "  Edge points - Original: {}, Optimized: {}",
            original_result.len(),
            optimized_result.len()
        );

        // Verify results are similar (they might not be identical due to parallel processing)
        let diff = (original_result.len() as i32 - optimized_result.len() as i32).abs();
        let diff_percentage = (diff as f64 / original_result.len() as f64) * 100.0;
        info!(
            "  Result difference: {} points ({:.2}%)",
            diff, diff_percentage
        );
    }

    // Calculate averages
    let avg_original = original_times.iter().sum::<std::time::Duration>() / runs as u32;
    let avg_optimized = optimized_times.iter().sum::<std::time::Duration>() / runs as u32;
    let speedup = avg_original.as_secs_f64() / avg_optimized.as_secs_f64();

    info!("\n=== Performance Summary ===");
    info!("Average Original Time: {:?}", avg_original);
    info!("Average Optimized Time: {:?}", avg_optimized);
    info!("Average Speedup: {:.2}x", speedup);

    if speedup > 1.0 {
        info!(
            "✅ Optimization successful! {:.1}% faster",
            (speedup - 1.0) * 100.0
        );
    } else {
        info!("⚠️  Optimization shows no improvement");
    }
}
