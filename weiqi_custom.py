import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

def identify_chessman(board_gray, board_bgr):
    try:
        if board_gray is None:
            return None

        mesh = np.linspace(22, 598, 19, dtype=np.int_)
        rows, cols = np.meshgrid(mesh, mesh)
        #print("Rows shape:", rows.shape)
        #print("Cols shape:", cols.shape)

        circles = cv2.HoughCircles(board_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=40, param2=10, minRadius=12,
                                   maxRadius=18)
        circles = np.uint32(np.around(circles[0]))

        phase = np.zeros_like(rows, dtype=np.uint8)
        im_hsv = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2HSV_FULL)

        for circle in circles:
            row = int(round((circle[1] - 22) / 32))
            col = int(round((circle[0] - 22) / 32))
            #print("Row:", row)
            #print("Col:", col)
            hsv_ = im_hsv[cols[row, col] - 5:cols[row, col] + 5, rows[row, col] - 5:rows[row, col] + 5]
            s = np.mean(hsv_[:, :, 1])
            v = np.mean(hsv_[:, :, 2])

            if 0 < v < 180:
                phase[row, col] = 1
            elif 0 < s < 90 and 114 < v < 256:    # s的最大值调高了，调出来的，以捕捉不同饱和度的棋子颜色
                phase[row, col] = 2

        return phase
    except Exception as e:
        print("An error occurred:", str(e))



if __name__ == '__main__':
    matrix_list = []  #用来放生成的矩阵，这个就是总矩阵列表
    # 构建目标文件夹路径
    target_folder = "./pic_all/"

    # 获取目标文件夹中的所有文件名，并按照文件名的开头数字顺序排序
    file_names = os.listdir(target_folder)

    # 自定义的排序键函数，提取文件名中的开头数字作为排序依据
    def sort_key(filename):
        # 提取文件名中的开头数字，例如 "1-1" 中的 "1"
        return int(filename.split('_')[0].split('-')[0])

    # 根据排序键函数对文件名进行排序
    sorted_file_names = sorted(file_names, key=sort_key)

    # 根据排序后的文件名构建路径列表
    image_paths = [os.path.join(target_folder, filename) for
                   filename in sorted_file_names]
    for path in image_paths:
        # 读取图片
        print(path)
        image = cv2.imread(path)

        # 将图片转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 使用Canny边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # 在边缘图上进行霍夫变换检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=600, maxLineGap=10)

        # 获取直线的端点，即边角坐标
        corners = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            corners.append((x1, y1))
            corners.append((x2, y2))

        # 使用K均值聚类进行点的合并
        kmeans = KMeans(n_clusters=4, n_init='auto')
        kmeans.fit(corners)
        cluster_centers = kmeans.cluster_centers_.astype(int)

        # 根据聚类结果获取四个点的坐标
        top_left = min(cluster_centers, key=lambda x: x[0] + x[1])
        top_right = max(cluster_centers, key=lambda x: x[0] - x[1])
        bottom_left = min(cluster_centers, key=lambda x: x[0] - x[1])
        bottom_right = max(cluster_centers, key=lambda x: x[0] + x[1])

        # 定义裁剪区域的四个坐标
        src_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

        # 定义目标区域的尺寸
        dst_width = 200  # 目标区域的宽度
        dst_height = 200  # 目标区域的高度
        dst_points = np.array([[0, 0], [dst_width - 1, 0], [dst_width - 1, dst_height - 1], [0, dst_height - 1]], dtype="float32")

        # 计算透视变换矩阵
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # 进行透视变换，裁剪成正方形
        warped_image = cv2.warpPerspective(image, transform_matrix, (dst_width, dst_height))

        # 将裁剪后的图像转换为灰度图
        gray_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

        gray_warped = cv2.resize(gray_warped,(620,620))
        warped_image = cv2.resize(warped_image,(620,620))
        result = identify_chessman(gray_warped,warped_image)
        print(result)