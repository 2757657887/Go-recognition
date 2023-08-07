import cv2
import numpy as np
import os



def find_chessboard(im_edge, im_gray, im_bgr):
    contours, hierarchy = cv2.findContours(im_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    rect = None

    for item in contours:
        hull = cv2.convexHull(item)
        epsilon = 0.1 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            ps = np.reshape(approx, (4, 2))
            ps = ps[np.lexsort((ps[:, 0],))]
            lt, lb = ps[:2][np.lexsort((ps[:2, 1],))]
            rt, rb = ps[2:][np.lexsort((ps[2:, 1],))]

            a = cv2.contourArea(approx)
            if a > area:
                area = a
                rect = (lt-9, lb-9, rt-9, rb-9) #调整目标角点的位置

    if rect is not None:
        pts1 = np.float32([(10, 10), (10, 650), (650, 10), (650, 650)])
        pts2 = np.float32(rect)
        m = cv2.getPerspectiveTransform(pts2, pts1)
        board_gray = cv2.warpPerspective(im_gray, m, (680, 680)) #这里我将画面改大，适应你的图像
        board_bgr = cv2.warpPerspective(im_bgr, m, (680, 680))
        return board_gray, board_bgr
    else:
        return None

def location_grid(board_gray, board_bgr):
    try:
        if board_gray is None:
            return None, None

        circles = cv2.HoughCircles(board_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=90, param2=16, minRadius=10,
                                   maxRadius=20)

        if circles is not None:
            xs, ys = circles[0, :, 0], circles[0, :, 1]
            xs.sort()
            ys.sort()

            k = 1
            while xs[k] - xs[:k].mean() < 15:
                k += 1
            x_min = int(round(xs[:k].mean()))

            k = 1
            while ys[k] - ys[:k].mean() < 15:
                k += 1
            y_min = int(round(ys[:k].mean()))

            k = -1
            while xs[k:].mean() - xs[k - 1] < 15:
                k -= 1
            x_max = int(round(xs[k:].mean()))

            k = -1
            while ys[k:].mean() - ys[k - 1] < 15:
                k -= 1
            y_max = int(round(ys[k:].mean()))

            if abs(600 - (x_max - x_min)) < abs(600 - (y_max - y_min)):
                v_min, v_max = x_min, x_max
            else:
                v_min, v_max = y_min, y_max

            pts1 = np.float32([[22, 22], [22, 598], [598, 22], [598, 598]])  # 棋盘四个角点的最终位置
            pts2 = np.float32([(v_min, v_min), (v_min, v_max), (v_max, v_min), (v_max, v_max)])
            m = cv2.getPerspectiveTransform(pts2, pts1)
            board_gray = cv2.warpPerspective(board_gray, m, (620, 620))
            board_bgr = cv2.warpPerspective(board_bgr, m, (620, 620))

            return board_gray, board_bgr
    except Exception as e:
        print("An error occurred in location_grid:", str(e))
        return None, None

    return None, None


def identify_chessman(board_gray, board_bgr):
    try:
        if board_gray is None:
            return None

        # 定义棋盘上每个格子的行列坐标
        mesh = np.linspace(22, 598, 19, dtype=np.int_)
        rows, cols = np.meshgrid(mesh, mesh)

        # 使用霍夫圆变换检测图像中的圆
        circles = cv2.HoughCircles(board_gray,
                                   cv2.HOUGH_GRADIENT, 1,
                                   20, param1=40, param2=10,
                                   minRadius=12,
                                   maxRadius=18)
        circles = np.uint32(np.around(circles[0]))

        # 创建相位图，用于标识棋盘上的棋子
        phase = np.zeros_like(rows, dtype=np.uint8)
        im_hsv = cv2.cvtColor(board_bgr,
                              cv2.COLOR_BGR2HSV_FULL)

        # 遍历检测到的圆
        for circle in circles:
            row = int(round((circle[1] - 22) / 32))  # 计算行坐标
            col = int(round((circle[0] - 22) / 32))  # 计算列坐标

            # 提取圆周围区域的HSV颜色信息
            hsv_ = im_hsv[
                   cols[row, col] - 5:cols[row, col] + 5,
                   rows[row, col] - 5:rows[row, col] + 5]
            s = np.mean(hsv_[:, :, 1])  # 平均饱和度
            v = np.mean(hsv_[:, :, 2])  # 平均亮度

            # 根据颜色信息判断棋子的相位
            if 0 < v < 130:  #######这里我调高了，用于更好的检测白子
                phase[row, col] = 1  # 亮棋子相位
            elif 0 < s < 50 and 114 < v < 256:
                phase[row, col] = 2  # 暗棋子相位

        return phase
    except Exception as e:
        print("发生错误:", str(e))
        return None

def process_chessboard_image(pic_file, offset=3.75):
    im_bgr = cv2.imread(pic_file)
    im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (3, 3), 0)
    im_edge = cv2.Canny(im_gray, 30, 50)# 使用Canny边缘检测算法获取边缘图像

    board_gray, board_bgr = find_chessboard(im_edge, im_gray, im_bgr)
    #board_gray, board_bgr = im_gray, im_bgr
    if board_gray is None:
        return None, None, None

    board_gray, board_bgr = location_grid(board_gray, board_bgr)
    phase = identify_chessman(board_gray, board_bgr)
    return board_gray, board_bgr, phase

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
    # 打印路径列表
    #for path in image_paths:
        #print(path)

    for path in image_paths:
        print(path)
        _,_,phase=process_chessboard_image(path, offset=3.75)
        if phase is None:
            continue
        print(phase)
        matrix_list.append(phase)
# 将二维矩阵列表转换为一个大的二维数组
combined_matrix = np.concatenate(matrix_list, axis=1)
# 指定CSV文件路径
csv_file_path = 'combined_matrices.csv'

# 逐个保存每个矩阵到CSV文件，并添加空白行
with open(csv_file_path, 'w') as f:
    for index,matrix in enumerate(matrix_list):
        np.savetxt(f, matrix, delimiter=',', fmt='%d')
        f.write(f"第{index}个矩阵")  # 添加空白行和矩阵名
        f.write('\n')
        f.write('\n')
print("保存成功！")
