use image::{GrayImage, Luma, Pixel, Rgba, RgbaImage, buffer::ConvertBuffer};
use imageproc::{
    definitions::Image,
    gradients::{horizontal_sobel, sobel_gradient_map, sobel_gradients, vertical_sobel},
};

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
    let p00 = data[y0 as usize * width as usize + x0 as usize];
    let p10 = data[y0 as usize * width as usize + x1 as usize];
    let p01 = data[y1 as usize * width as usize + x0 as usize];
    let p11 = data[y1 as usize * width as usize + x1 as usize];

    // 计算权重
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    // 双线性插值
    p00 * (1.0 - dx) * (1.0 - dy) + p10 * dx * (1.0 - dy) + p01 * (1.0 - dx) * dy + p11 * dx * dy
}

/// 亚像素边缘检测
///
/// # 参数
/// - `image`: 输入灰度图像
/// - `gradient_threshold`: 梯度幅值阈值
/// - `edge_point_threshold`: 有效边缘点阈值
///
/// # 返回
/// 亚像素边缘点列表 `Vec<(f32, f32)>`
pub fn subpixel_edge_detection(
    image: &GrayImage,
    gradient_threshold: f32,
    edge_point_threshold: f32,
) -> Vec<(f32, f32)> {
    let (width, height) = image.dimensions();

    // 计算梯度
    // let (gx, gy) = sobel_gradient_map(image, |p| p as f32);
    // let gx_data = gx.as_raw();
    // let gy_data = gy.as_raw();
    //
    let gx = horizontal_sobel(image);
    let gy = vertical_sobel(image);

    // 将梯度转换为f32并存储为Vec
    let gx_data: Vec<f32> = gx.iter().map(|&p| p as f32).collect();
    let gy_data: Vec<f32> = gy.iter().map(|&p| p as f32).collect();

    // 计算梯度幅值
    let mag_data: Vec<f32> = gx_data
        .iter()
        .zip(gy_data.iter())
        .map(|(gx, gy)| (gx.powi(2) + gy.powi(2)).sqrt())
        .collect();

    let mut edge_points = Vec::new();

    // 遍历像素（跳过边界）
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let idx = (y * width + x) as usize;
            let mag = mag_data[idx];

            // 过滤低梯度点
            if mag < gradient_threshold {
                continue;
            }

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
                continue;
            }

            // 双线性插值获取梯度幅值
            let m1 = bilinear_interpolation(&mag_data, width, height, x1, y1);
            let m2 = bilinear_interpolation(&mag_data, width, height, x2, y2);

            // 抛物线拟合参数
            let a = 2.0 * (m1 + m2 - 2.0 * mag);
            if a >= 0.0 {
                // 确保是极大值点
                continue;
            }

            // 计算亚像素偏移
            let offset = (m2 - m1) / (4.0 * (m1 + m2 - 2.0 * mag));

            // 过滤无效偏移
            if offset.abs() > edge_point_threshold {
                continue;
            }

            // 计算亚像素坐标
            let sub_x = x as f32 + offset * dx;
            let sub_y = y as f32 + offset * dy;

            edge_points.push((sub_x, sub_y));
        }
    }

    edge_points
}

/// 可视化亚像素边缘（调试用）
pub fn visualize_edges(image: &GrayImage, edge_points: &[(f32, f32)]) -> RgbaImage {
    let mut canvas: RgbaImage = image.convert();
    let red = Rgba([255, 0, 0, 100]);

    for &(x, y) in edge_points {
        let sx = x as i32;
        let sy = y as i32;

        if sx >= 0 && sy >= 0 && sx < canvas.width() as i32 && sy < canvas.height() as i32 {
            canvas.put_pixel(sx as u32, sy as u32, red);
        }
    }

    canvas
}
