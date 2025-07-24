use std::time::Instant;

use env_logger::Builder;
use image::{imageops::blur, open};
use log::info;
use subpixel_edge::{canny_based_subpixel_edges_optimized, visualize_edges};

fn main() {
    Builder::from_default_env().format_timestamp_nanos().init();
    // 加载图像
    let img = open("test_image/edge-small.png").unwrap().to_luma8();

    let blur = blur(&img, 1.0);

    info!("start cal");

    let instance = Instant::now();

    // 亚像素边缘检测
    let edges = canny_based_subpixel_edges_optimized(&blur, 20.0, 80.0, 0.6);

    let elapsed = instance.elapsed();
    info!("检测耗时: {elapsed:?}");

    //
    let result = visualize_edges(&img, &edges);
    result.save("test_image/subpixel_edges.png").unwrap();

    info!("检测到 {} 个亚像素边缘点", edges.len());
}
