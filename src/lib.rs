use std::{
    f32::consts::PI,
    sync::{Arc, Mutex},
};

use image::{GenericImageView, GrayImage, ImageBuffer, Luma, Rgb, RgbImage, buffer::ConvertBuffer};
use imageproc::{
    definitions::{HasBlack, HasWhite},
    edges::canny,
    gradients::{horizontal_sobel, vertical_sobel},
};
use log::debug;
use rayon::prelude::*;

/// 双线性插值函数
fn bilinear_interpolation(data: &[f32], width: u32, height: u32, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    // 边界检查
    if x0 < 0 || y0 < 0 || x1 >= width as i32 || y1 >= height as i32 {
        return 0.0;
    }

    // 获取四个相邻点
    let p00 = data[(y0 as u32 * width + x0 as u32) as usize];
    let p10 = data[(y0 as u32 * width + x1 as u32) as usize];
    let p01 = data[(y1 as u32 * width + x0 as u32) as usize];
    let p11 = data[(y1 as u32 * width + x1 as u32) as usize];

    // 计算权重
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    // 双线性插值
    p00 * (1.0 - dx) * (1.0 - dy) + p10 * dx * (1.0 - dy) + p01 * (1.0 - dx) * dy + p11 * dx * dy
}

/// 在3x3区域内进行亚像素边缘定位
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

    // 计算梯度方向
    let gx_val = gx_data[idx];
    let gy_val = gy_data[idx];
    let theta = gy_val.atan2(gx_val);
    let dx = theta.cos();
    let dy = theta.sin();

    // 计算梯度方向上的偏移点
    let x1 = x as f32 + 0.5 * dx;
    let y1 = y as f32 + 0.5 * dy;
    let x2 = x as f32 - 0.5 * dx;
    let y2 = y as f32 - 0.5 * dy;

    // 边界检查
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

    // 双线性插值获取梯度幅值
    let m1 = bilinear_interpolation(mag_data, width, height, x1, y1);
    let m2 = bilinear_interpolation(mag_data, width, height, x2, y2);

    // 抛物线拟合参数
    let a = 2.0 * (m1 + m2 - 2.0 * mag);
    if a >= 0.0 {
        // 确保是极大值点
        return None;
    }

    // 计算亚像素偏移
    let offset = (m2 - m1) / (4.0 * (m1 + m2 - 2.0 * mag));

    // 过滤无效偏移
    if offset.abs() > edge_point_threshold {
        return None;
    }

    // 计算亚像素坐标
    let sub_x = x as f32 + offset * dx;
    let sub_y = y as f32 + offset * dy;

    Some((sub_x, sub_y))
}

// /// 基于Canny的亚像素边缘检测
// pub fn canny_based_subpixel_edges(
//     image: &GrayImage,
//     low_threshold: f32,
//     high_threshold: f32,
//     edge_point_threshold: f32,
// ) -> Vec<(f32, f32)> {
//     let (width, height) = image.dimensions();

//     // 步骤1：使用Canny检测像素级边缘
//     let canny_edges = canny(image, low_threshold, high_threshold);

//     // 步骤2：计算梯度信息
//     // let (gx_image, gy_image) = sobel_gradients(image);
//     let gx_image = horizontal_sobel(image);
//     let gy_image = vertical_sobel(image);

//     let gx_data: Vec<f32> = gx_image.pixels().map(|p| p[0] as f32).collect();
//     let gy_data: Vec<f32> = gy_image.pixels().map(|p| p[0] as f32).collect();

//     // 计算梯度幅值 - 并行化
//     let mag_data: Vec<f32> = gx_data
//         .par_iter()
//         .zip(gy_data.par_iter())
//         .map(|(gx, gy)| (gx.powi(2) + gy.powi(2)).sqrt())
//         .collect();

//     // 收集所有Canny边缘点 - 并行化
//     let canny_points: Vec<(u32, u32)> = (1..(height - 1))
//         .into_par_iter()
//         .flat_map(|y| {
//             let canny_edges = &canny_edges;
//             (1..(width - 1)).into_par_iter().filter_map(move |x| {
//                 if canny_edges.get_pixel(x, y)[0] > 0 {
//                     Some((x, y))
//                 } else {
//                     None
//                 }
//             })
//         })
//         .collect();

//     // 步骤3：在Canny边缘点上进行亚像素定位 - 并行化
//     canny_points
//         .into_par_iter()
//         .filter_map(|(x, y)| {
//             subpixel_in_3x3(
//                 x,
//                 y,
//                 &gx_data,
//                 &gy_data,
//                 &mag_data,
//                 width,
//                 height,
//                 edge_point_threshold,
//             )
//         })
//         .collect()
// }

// /// 并行版本的Canny亚像素边缘检测
// pub fn canny_based_subpixel_edges_parallel(
//     image: &GrayImage,
//     low_threshold: f32,
//     high_threshold: f32,
//     edge_point_threshold: f32,
// ) -> Vec<(f32, f32)> {
//     let (width, height) = image.dimensions();

//     // 步骤1：使用Canny检测像素级边缘
//     let canny_edges = canny(image, low_threshold, high_threshold);

//     // 步骤2：计算梯度信息
//     let gx_image = horizontal_sobel(image);
//     let gy_image = vertical_sobel(image);
//     // 并行化数据转换
//     let gx_data: Vec<f32> = gx_image
//         .pixels()
//         .collect::<Vec<_>>()
//         .into_par_iter()
//         .map(|p| p[0] as f32)
//         .collect();
//     let gy_data: Vec<f32> = gy_image
//         .pixels()
//         .collect::<Vec<_>>()
//         .into_par_iter()
//         .map(|p| p[0] as f32)
//         .collect();

//     // 计算梯度幅值 - 并行化
//     let mag_data: Vec<f32> = gx_data
//         .par_iter()
//         .zip(gy_data.par_iter())
//         .map(|(gx, gy)| (gx.powi(2) + gy.powi(2)).sqrt())
//         .collect();

//     // 收集所有Canny边缘点 - 并行化
//     let canny_points: Vec<(u32, u32)> = (1..(height - 1))
//         .into_par_iter()
//         .flat_map(|y| {
//             let canny_edges = &canny_edges;
//             (1..(width - 1)).into_par_iter().filter_map(move |x| {
//                 if canny_edges.get_pixel(x, y)[0] > 0 {
//                     Some((x, y))
//                 } else {
//                     None
//                 }
//             })
//         })
//         .collect();

//     // 并行处理每个边缘点
//     canny_points
//         .into_par_iter()
//         .filter_map(|(x, y)| {
//             subpixel_in_3x3(
//                 x,
//                 y,
//                 &gx_data,
//                 &gy_data,
//                 &mag_data,
//                 width,
//                 height,
//                 edge_point_threshold,
//             )
//         })
//         .collect()
// }

/// 可视化亚像素边缘
pub fn visualize_edges(image: &GrayImage, edge_points: &[(f32, f32)]) -> RgbImage {
    let mut canvas: RgbImage = image.convert();
    let red = Rgb([255u8, 0, 0]);

    // 并行化边缘点绘制（注意：这里需要同步写入，所以使用 for_each 而不是直接修改）
    // 为了线程安全，我们收集有效点然后串行绘制
    let valid_points: Vec<(u32, u32)> = edge_points
        .par_iter()
        .filter_map(|&(x, y)| {
            let sx = x as i32;
            let sy = y as i32;
            if sx >= 0 && sy >= 0 && sx < canvas.width() as i32 && sy < canvas.height() as i32 {
                Some((sx as u32, sy as u32))
            } else {
                None
            }
        })
        .collect();

    // 串行绘制（避免并发写入同一个像素的问题）
    for (sx, sy) in valid_points {
        canvas.put_pixel(sx, sy, red);
    }

    canvas
}

/// 高性能版本 - 添加一个新的完全优化的函数
pub fn canny_based_subpixel_edges_optimized(
    image: &GrayImage,
    low_threshold: f32,
    high_threshold: f32,
    edge_point_threshold: f32,
) -> Vec<(f32, f32)> {
    let (width, height) = image.dimensions();
    debug!("start calcualte subpixel edges");
    // 步骤2：并行计算梯度信息
    // let gx_image = horizontal_sobel(image);
    // let gy_image = vertical_sobel(image);

    let (gx_data, gy_data) = parallel_sobel_gradients(image);

    debug!("gx_data and gy_data ok");

    let gx_image_data: Vec<i16> = gx_data.par_iter().map(|p| *p as i16).collect();
    let gy_image_data: Vec<i16> = gy_data.par_iter().map(|p| *p as i16).collect();

    let gx_image = ImageBuffer::from_raw(width, height, gx_image_data).unwrap();
    let gy_image = ImageBuffer::from_raw(width, height, gy_image_data).unwrap();

    debug!("gx_image and gy_image ok");

    // 并行化所有数据转换和计算
    // let gx_data: Vec<f32> = gx_image.pixels().map(|p| p[0] as f32).collect();
    // let gy_data: Vec<f32> = gy_image.pixels().map(|p| p[0] as f32).collect();

    debug!("gx_data and gy_data ok");

    let mag_data: Vec<f32> = gx_data
        .par_iter()
        .zip(gy_data.par_iter())
        .map(|(gx, gy)| (gx.powi(2) + gy.powi(2)).sqrt())
        .collect();

    debug!("mag_data ok");

    let g: Vec<f32> = gx_image
        .iter()
        .zip(gy_image.iter())
        .map(|(h, v)| (*h as f32).hypot(*v as f32))
        .collect::<Vec<f32>>();

    debug!("g ok");

    let g = ImageBuffer::from_raw(image.width(), image.height(), g).unwrap();

    debug!("g image ok");

    // 3. Non-maximum-suppression (Make edges thinner)
    let thinned = non_maximum_suppression(&g, &gx_image, &gy_image);

    debug!("thinned ok");

    // 4. Hysteresis to filter out edges based on thresholds.
    let canny_edge_points = hysteresis_original(&thinned, low_threshold, high_threshold);

    debug!("canny_edge_points ok");

    // 并行处理每个边缘点
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

pub fn parallel_sobel_gradients(image: &GrayImage) -> (Vec<f32>, Vec<f32>) {
    let (width, height) = image.dimensions();
    let gx = Arc::new(Mutex::new(vec![0.0; (width * height) as usize]));
    let gy = Arc::new(Mutex::new(vec![0.0; (width * height) as usize]));

    // 获取原始像素数据
    let pixels = image.as_raw();

    // 定义Sobel算子内核
    const SOBEL_KERNEL_X: [f32; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    const SOBEL_KERNEL_Y: [f32; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

    // 使用Rayon并行处理每一行（跳过边界行）
    (1..height - 1).into_par_iter().for_each(|y| {
        // 计算当前行的起始索引
        let row_start = (y * width) as usize;

        // 获取上一行、当前行和下一行的像素切片
        let prev_row =
            &pixels[(row_start - width as usize)..(row_start - width as usize + width as usize)];
        let curr_row = &pixels[row_start..(row_start + width as usize)];
        let next_row =
            &pixels[(row_start + width as usize)..(row_start + width as usize + width as usize)];

        let mut gx_mutex = gx.lock().unwrap();
        let mut gy_mutex = gy.lock().unwrap();

        // 处理当前行的每个像素（跳过边界列）
        for x in 1..(width - 1) {
            let mut gx_val = 0.0;
            let mut gy_val = 0.0;

            // 3x3卷积核处理
            for ky in 0..3 {
                for kx in 0..3 {
                    let pixel_x = x + kx - 1;

                    // 获取像素值
                    let pixel = if ky == 0 {
                        prev_row[pixel_x as usize] as f32
                    } else if ky == 1 {
                        curr_row[pixel_x as usize] as f32
                    } else {
                        next_row[pixel_x as usize] as f32
                    };

                    // 应用Sobel核
                    let kernel_index = (ky * 3 + kx) as usize;
                    gx_val += pixel * SOBEL_KERNEL_X[kernel_index];
                    gy_val += pixel * SOBEL_KERNEL_Y[kernel_index];
                }
            }

            // 存储计算结果
            let index = (y * width + x) as usize;
            gx_mutex[index] = gx_val;
            gy_mutex[index] = gy_val;
        }
    });

    let gx_result = Arc::try_unwrap(gx).unwrap().into_inner().unwrap();
    let gy_result = Arc::try_unwrap(gy).unwrap().into_inner().unwrap();
    (gx_result, gy_result)
}

/// Finds local maxima to make the edges thinner.
/// Finds local maxima to make the edges thinner.
fn non_maximum_suppression(
    g: &ImageBuffer<Luma<f32>, Vec<f32>>,
    gx: &ImageBuffer<Luma<i16>, Vec<i16>>,
    gy: &ImageBuffer<Luma<i16>, Vec<i16>>,
) -> ImageBuffer<Luma<f32>, Vec<f32>> {
    const RADIANS_TO_DEGREES: f32 = 180f32 / PI;
    let mut out = ImageBuffer::from_pixel(g.width(), g.height(), Luma([0.0]));
    let mut points = Vec::new();
    for y in 1..g.height() - 1 {
        for x in 1..g.width() - 1 {
            let x_gradient = gx[(x, y)][0] as f32;
            let y_gradient = gy[(x, y)][0] as f32;
            let mut angle = (y_gradient).atan2(x_gradient) * RADIANS_TO_DEGREES;
            if angle < 0.0 {
                angle += 180.0
            }
            // Clamp angle.
            let clamped_angle = if !(22.5..157.5).contains(&angle) {
                0
            } else if (22.5..67.5).contains(&angle) {
                45
            } else if (67.5..112.5).contains(&angle) {
                90
            } else if (112.5..157.5).contains(&angle) {
                135
            } else {
                unreachable!()
            };

            // Get the two perpendicular neighbors.
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
            // If the pixel is not a local maximum, suppress it.
            if pixel[0] < cmp1[0] || pixel[0] < cmp2[0] {
                out.put_pixel(x, y, Luma([0.0]));
            } else {
                out.put_pixel(x, y, pixel);
                points.push((x, y));
            }
        }
    }
    debug!("points len:{}", points.len());
    out
}

/// Filter out edges with the thresholds.
/// Non-recursive breadth-first search.
fn hysteresis_original(
    input: &ImageBuffer<Luma<f32>, Vec<f32>>,
    low_thresh: f32,
    high_thresh: f32,
) -> Vec<(u32, u32)> {
    let max_brightness = Luma::<u8>::white();
    let min_brightness = Luma::<u8>::black();
    // Init output image as all black.
    let mut out = ImageBuffer::from_pixel(input.width(), input.height(), min_brightness);
    // Stack. Possible optimization: Use previously allocated memory, i.e. gx.
    let mut edges = Vec::with_capacity(((input.width() * input.height()) / 2) as usize);
    let mut result_edge_points = Vec::new();
    for y in 1..input.height() - 1 {
        for x in 1..input.width() - 1 {
            let inp_pix = *input.get_pixel(x, y);
            let out_pix = *out.get_pixel(x, y);
            // If the edge strength is higher than high_thresh, mark it as an edge.
            if inp_pix[0] >= high_thresh && out_pix[0] == 0 {
                out.put_pixel(x, y, max_brightness);
                edges.push((x, y));
                // Track neighbors until no neighbor is >= low_thresh.
                while let Some((nx, ny)) = edges.pop() {
                    let neighbor_indices = [
                        (nx + 1, ny),
                        (nx + 1, ny + 1),
                        (nx, ny + 1),
                        (nx.wrapping_sub(1), ny.wrapping_sub(1)),
                        (nx.wrapping_sub(1), ny),
                        (nx.wrapping_sub(1), ny + 1),
                    ];

                    for neighbor_idx in &neighbor_indices {
                        // Bounds checking
                        if neighbor_idx.0 >= input.width()
                            || neighbor_idx.1 >= input.height()
                            || neighbor_idx.0 == u32::MAX
                            || neighbor_idx.1 == u32::MAX
                        {
                            continue;
                        }

                        let in_neighbor = *input.get_pixel(neighbor_idx.0, neighbor_idx.1);
                        let out_neighbor = *out.get_pixel(neighbor_idx.0, neighbor_idx.1);
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

fn hysteresis(
    input: &ImageBuffer<Luma<f32>, Vec<f32>>,
    low_thresh: f32,
    high_thresh: f32,
) -> Vec<(u32, u32)> {
    let width = input.width();
    let height = input.height();

    // Use Arc<Mutex<>> for thread-safe access to the output image
    let out = Arc::new(Mutex::new(ImageBuffer::from_pixel(
        width,
        height,
        Luma::<u8>::black(),
    )));

    // Find all strong edge pixels (above high threshold) in parallel
    let strong_edges: Vec<(u32, u32)> = (1..height - 1)
        .into_par_iter()
        .flat_map(|y| {
            (1..width - 1).into_par_iter().filter_map(move |x| {
                let pixel_value = input.get_pixel(x, y)[0];
                if pixel_value >= high_thresh {
                    Some((x, y))
                } else {
                    None
                }
            })
        })
        .collect();

    // Mark strong edges and collect result points in parallel
    let result_edge_points = Arc::new(Mutex::new(Vec::new()));

    strong_edges.par_iter().for_each(|&(start_x, start_y)| {
        let mut local_edges = Vec::new();
        let mut local_stack = vec![(start_x, start_y)];

        // Check if this pixel is already processed by another thread
        {
            let mut out_guard = out.lock().unwrap();
            if out_guard.get_pixel(start_x, start_y)[0] != 0 {
                return; // Already processed
            }
            out_guard.put_pixel(start_x, start_y, Luma::<u8>::white());
        }

        // Process connected component using depth-first search
        while let Some((x, y)) = local_stack.pop() {
            local_edges.push((x, y));

            // Check all 8 neighbors with proper bounds checking
            let neighbors = [
                (x.wrapping_sub(1), y.wrapping_sub(1)),
                (x, y.wrapping_sub(1)),
                (x + 1, y.wrapping_sub(1)),
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x.wrapping_sub(1), y + 1),
                (x, y + 1),
                (x + 1, y + 1),
            ];

            for &(nx, ny) in &neighbors {
                // Bounds checking - check for overflow and out of bounds
                if nx >= width || ny >= height || nx == u32::MAX || ny == u32::MAX {
                    continue;
                }
                // Also skip border pixels
                if nx == 0 || ny == 0 || nx >= width - 1 || ny >= height - 1 {
                    continue;
                }

                // Check if neighbor meets low threshold and hasn't been processed
                let neighbor_value = input.get_pixel(nx, ny)[0];
                if neighbor_value >= low_thresh {
                    let mut out_guard = out.lock().unwrap();
                    if out_guard.get_pixel(nx, ny)[0] == 0 {
                        out_guard.put_pixel(nx, ny, Luma::<u8>::white());
                        local_stack.push((nx, ny));
                    }
                }
            }
        }

        // Add local results to global result
        if !local_edges.is_empty() {
            let mut result_guard = result_edge_points.lock().unwrap();
            result_guard.extend(local_edges);
        }
    });

    // Extract final result
    Arc::try_unwrap(result_edge_points)
        .unwrap()
        .into_inner()
        .unwrap()
}

/// Performance comparison function for hysteresis optimization
pub fn compare_hysteresis_performance(
    input: &ImageBuffer<Luma<f32>, Vec<f32>>,
    low_thresh: f32,
    high_thresh: f32,
) -> (
    std::time::Duration,
    std::time::Duration,
    Vec<(u32, u32)>,
    Vec<(u32, u32)>,
) {
    use std::time::Instant;

    // Test original implementation
    let start = Instant::now();
    let original_result = hysteresis_original(input, low_thresh, high_thresh);
    let original_time = start.elapsed();

    // Test optimized implementation
    let start = Instant::now();
    let optimized_result = hysteresis(input, low_thresh, high_thresh);
    let optimized_time = start.elapsed();

    (
        original_time,
        optimized_time,
        original_result,
        optimized_result,
    )
}
