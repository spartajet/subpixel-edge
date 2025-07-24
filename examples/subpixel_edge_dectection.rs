use image::open;
use subpixel_edge::{subpixel_edge_detection, visualize_edges};

fn main() {
    // 加载图像
    let img = open("test_image/edge.png").unwrap().to_luma8();

    // 亚像素边缘检测
    let edges = subpixel_edge_detection(&img, 100.0, 0.8);

    // 可视化结果（放大5倍显示）
    let result = visualize_edges(&img, &edges);
    result.save("test_image/subpixel_edges.png").unwrap();

    println!("检测到 {} 个亚像素边缘点", edges.len());
}
