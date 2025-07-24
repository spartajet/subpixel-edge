use image::{GrayImage, Rgb, RgbImage, buffer::ConvertBuffer};
use imageproc::{
    edges::canny,
    gradients::{horizontal_sobel, vertical_sobel},
};
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

/// 基于Canny的亚像素边缘检测
pub fn canny_based_subpixel_edges(
    image: &GrayImage,
    low_threshold: f32,
    high_threshold: f32,
    edge_point_threshold: f32,
) -> Vec<(f32, f32)> {
    let (width, height) = image.dimensions();

    // 步骤1：使用Canny检测像素级边缘
    let canny_edges = canny(image, low_threshold, high_threshold);

    // 步骤2：计算梯度信息
    // let (gx_image, gy_image) = sobel_gradients(image);
    let gx_image = horizontal_sobel(image);
    let gy_image = vertical_sobel(image);

    let gx_data: Vec<f32> = gx_image.pixels().map(|p| p[0] as f32).collect();
    let gy_data: Vec<f32> = gy_image.pixels().map(|p| p[0] as f32).collect();

    // 计算梯度幅值 - 并行化
    let mag_data: Vec<f32> = gx_data
        .par_iter()
        .zip(gy_data.par_iter())
        .map(|(gx, gy)| (gx.powi(2) + gy.powi(2)).sqrt())
        .collect();

    // 收集所有Canny边缘点 - 并行化
    let canny_points: Vec<(u32, u32)> = (1..(height - 1))
        .into_par_iter()
        .flat_map(|y| {
            let canny_edges = &canny_edges;
            (1..(width - 1)).into_par_iter().filter_map(move |x| {
                if canny_edges.get_pixel(x, y)[0] > 0 {
                    Some((x, y))
                } else {
                    None
                }
            })
        })
        .collect();

    // 步骤3：在Canny边缘点上进行亚像素定位 - 并行化
    canny_points
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

/// 并行版本的Canny亚像素边缘检测
pub fn canny_based_subpixel_edges_parallel(
    image: &GrayImage,
    low_threshold: f32,
    high_threshold: f32,
    edge_point_threshold: f32,
) -> Vec<(f32, f32)> {
    let (width, height) = image.dimensions();

    // 步骤1：使用Canny检测像素级边缘
    let canny_edges = canny(image, low_threshold, high_threshold);

    // 步骤2：计算梯度信息
    let gx_image = horizontal_sobel(image);
    let gy_image = vertical_sobel(image);
    // 并行化数据转换
    let gx_data: Vec<f32> = gx_image
        .pixels()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|p| p[0] as f32)
        .collect();
    let gy_data: Vec<f32> = gy_image
        .pixels()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|p| p[0] as f32)
        .collect();

    // 计算梯度幅值 - 并行化
    let mag_data: Vec<f32> = gx_data
        .par_iter()
        .zip(gy_data.par_iter())
        .map(|(gx, gy)| (gx.powi(2) + gy.powi(2)).sqrt())
        .collect();

    // 收集所有Canny边缘点 - 并行化
    let canny_points: Vec<(u32, u32)> = (1..(height - 1))
        .into_par_iter()
        .flat_map(|y| {
            let canny_edges = &canny_edges;
            (1..(width - 1)).into_par_iter().filter_map(move |x| {
                if canny_edges.get_pixel(x, y)[0] > 0 {
                    Some((x, y))
                } else {
                    None
                }
            })
        })
        .collect();

    // 并行处理每个边缘点
    canny_points
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

    // 步骤1：使用Canny检测像素级边缘
    let canny_edges = canny(image, low_threshold, high_threshold);

    // 步骤2：并行计算梯度信息
    let gx_image = horizontal_sobel(image);
    let gy_image = vertical_sobel(image);

    // 并行化所有数据转换和计算
    let gx_pixels: Vec<_> = gx_image.pixels().collect();
    let gy_pixels: Vec<_> = gy_image.pixels().collect();

    let tuples: Vec<(f32, f32, f32)> = gx_pixels
        .into_par_iter()
        .zip(gy_pixels.into_par_iter())
        .map(|(gx_pixel, gy_pixel)| {
            let gx = gx_pixel[0] as f32;
            let gy = gy_pixel[0] as f32;
            let mag = (gx.powi(2) + gy.powi(2)).sqrt();
            (gx, gy, mag)
        })
        .collect();

    let mut gx_data = Vec::with_capacity(tuples.len());
    let mut gy_data = Vec::with_capacity(tuples.len());
    let mut mag_data = Vec::with_capacity(tuples.len());

    for (gx, gy, mag) in tuples {
        gx_data.push(gx);
        gy_data.push(gy);
        mag_data.push(mag);
    }

    // 步骤3：并行收集和处理边缘点
    let canny_points: Vec<(u32, u32)> = (1..(height - 1))
        .into_par_iter()
        .flat_map(|y| {
            let canny_edges = &canny_edges;
            (1..(width - 1)).into_par_iter().filter_map(move |x| {
                if canny_edges.get_pixel(x, y)[0] > 0 {
                    Some((x, y))
                } else {
                    None
                }
            })
        })
        .collect();

    // 并行处理每个边缘点
    canny_points
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
