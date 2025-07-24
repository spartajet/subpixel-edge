use image::{GenericImageView, GrayImage, Luma, Pixel, Rgb, RgbImage, buffer::ConvertBuffer};
use imageproc::{
    definitions::Image,
    edges::canny,
    gradients::{horizontal_sobel, sobel_gradients, vertical_sobel},
};
use rayon::prelude::*;
use std::f32::consts::PI;

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
    let gx_image = horizontal_sobel(&image);
    let gy_image = vertical_sobel(&image);

    let gx_data: Vec<f32> = gx_image.pixels().map(|p| p[0] as f32).collect();
    let gy_data: Vec<f32> = gy_image.pixels().map(|p| p[0] as f32).collect();

    // 计算梯度幅值
    let mag_data: Vec<f32> = gx_data
        .iter()
        .zip(gy_data.iter())
        .map(|(gx, gy)| (gx.powi(2) + gy.powi(2)).sqrt())
        .collect();

    let mut edge_points = Vec::new();

    // 步骤3：在Canny边缘点上进行亚像素定位
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            // 只处理Canny检测到的边缘点
            if canny_edges.get_pixel(x, y)[0] > 0 {
                if let Some(point) = subpixel_in_3x3(
                    x,
                    y,
                    &gx_data,
                    &gy_data,
                    &mag_data,
                    width,
                    height,
                    edge_point_threshold,
                ) {
                    edge_points.push(point);
                }
            }
        }
    }

    edge_points
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
    let gx_image = horizontal_sobel(&image);
    let gy_image = vertical_sobel(&image);
    let gx_data: Vec<f32> = gx_image.pixels().map(|p| p[0] as f32).collect();
    let gy_data: Vec<f32> = gy_image.pixels().map(|p| p[0] as f32).collect();

    // 计算梯度幅值
    let mag_data: Vec<f32> = gx_data
        .iter()
        .zip(gy_data.iter())
        .map(|(gx, gy)| (gx.powi(2) + gy.powi(2)).sqrt())
        .collect();

    // 收集所有Canny边缘点
    let mut canny_points = Vec::new();
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            if canny_edges.get_pixel(x, y)[0] > 0 {
                canny_points.push((x, y));
            }
        }
    }

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

    for &(x, y) in edge_points {
        let sx = x as i32;
        let sy = y as i32;

        if sx >= 0 && sy >= 0 && sx < canvas.width() as i32 && sy < canvas.height() as i32 {
            canvas.put_pixel(sx as u32, sy as u32, red);
        }
    }
    canvas
}
