use super::edge_detection::{subpixel_edge_detection, visualize_edges};
use image::open;

fn main() {
    // 加载图像
    let img = open("input.jpg").unwrap().to_luma8();

    // 亚像素边缘检测
    let edges = subpixel_edge_detection(&img, 20.0, 0.6);

    // 可视化结果（放大5倍显示）
    let result = visualize_edges(&img, &edges, 5);
    result.save("edges.png").unwrap();

    println!("检测到 {} 个亚像素边缘点", edges.len());
}
