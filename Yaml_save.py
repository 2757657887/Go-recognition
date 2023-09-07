import cv2
import numpy as np
import os
import json


def center_crop(image, crop_size):
    """
    对输入图像进行中心裁剪。

    参数：
        image: 输入图像
        crop_size: 裁剪尺寸 (width, height)

    返回：
        中心裁剪后的图像
    """
    if crop_size[0] >= image.shape[1] or crop_size[1] >= \
            image.shape[0]:
        raise ValueError("裁剪尺寸大于图像尺寸")

    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2

    half_width = crop_size[0] // 2
    half_height = crop_size[1] // 2

    cropped = image[
              center_y - half_height:center_y + half_height,
              center_x - half_width:center_x + half_width]

    return cropped


def split_image(image, rows, cols):
    # 获取图像的高度和宽度
    height, width = image.shape[:2]
    # 计算每个小格子的高度和宽度
    cell_height = height // rows
    cell_width = width // cols

    cells = []
    for row in range(rows):
        for col in range(cols):
            # 计算当前小格子的坐标范围
            y1 = row * cell_height
            y2 = (row + 1) * cell_height
            x1 = col * cell_width
            x2 = (col + 1) * cell_width

            # 截取当前小格子的图像
            cell = image[y1:y2, x1:x2]
            cells.append(cell)

    return cells


def analyze_image(image):
    # 获取图像的高度、宽度和通道数
    height, width, _ = image.shape
    total_pixels = height * width
    # 计算黑色、白色和背景色的像素数量
    black_pixels = np.sum(
        np.logical_and.reduce(image < 10, axis=-1))
    white_pixels = np.sum(
        np.logical_and.reduce(image > 190, axis=-1))
    specific_color_pixels = np.sum(
        np.logical_and(
            np.logical_and(image[:, :, 2] >= 200,
                           image[:, :, 2] <= 220),
            np.logical_and(image[:, :, 1] >= 170,
                           image[:, :, 1] <= 190),
            np.logical_and(image[:, :, 0] >= 130,
                           image[:, :, 0] <= 150),
        )
    )
    # 计算像素占比
    black_ratio = black_pixels / total_pixels
    white_ratio = white_pixels / total_pixels
    specific_color_ratio = specific_color_pixels / total_pixels

    # 根据占比判断图片类别
    if black_ratio > 0.3:
        return 1  # 黑色
    elif white_ratio > 0.2:
        return 2  # 白色
    elif specific_color_ratio > 0.5:
        return 0  # 背景色
    else:
        return 0  # 背景色


def to_yaml(image_path, save_name):
    # 用于保存矩阵数据的字典，其中key为图片路径，value为对应的类别矩阵
    matrix_dict = {}

    # 构建目标文件夹路径
    target_folder = image_path
    file_names = os.listdir(target_folder)

    # 自定义的排序键函数
    def sort_key(filename):
        return int(filename.split('_')[0].split('-')[0])

    # 根据排序键函数对文件名进行排序
    sorted_file_names = sorted(file_names, key=sort_key)

    # 根据排序后的文件名构建路径列表
    image_paths = [os.path.join(target_folder, filename) for
                   filename in sorted_file_names]

    for path in image_paths:
        im_bgr = cv2.imread(path)
        target_size = (660, 660)
        im_bgr = cv2.resize(im_bgr, target_size)
        crop_size = (608, 608)
        cropped_image = center_crop(im_bgr, crop_size)

        # 将裁剪后的图像分割为19x19个小格子
        split_rows = 19
        split_cols = 19
        small_cells = split_image(cropped_image, split_rows,
                                  split_cols)

        # 创建一个空矩阵来保存类别信息
        class_matrix = np.zeros((split_rows, split_cols),
                                dtype=int)

        # 对每个小格子应用判断函数，得到类别信息
        row = 0
        cols = 0
        for i in small_cells:
            class_matrix[cols][row] = analyze_image(i)
            row += 1
            if row == 19:
                row = 0
                cols += 1

        # 将路径与矩阵添加到字典中
        matrix_dict[
            path] = class_matrix.tolist()  # 转换为Python原生list结构

    # 保存字典为JSON文件
    json_file_path = save_name + '.yaml'
    with open(json_file_path, 'w') as f:
        json.dump(matrix_dict, f)
    print("保存成功！")

