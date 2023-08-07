# Go-recognition
# 图像分析与矩阵生成

这个Python脚本旨在对输入图像进行分析，并基于小格子的特征生成矩阵。它接受一组图像，对其进行处理，并生成一个合并的矩阵，同时将各个矩阵保存到CSV文件中。

## 功能特点

1. **中心裁剪**：脚本提供了一个名为`center_crop`的函数，接受输入图像和裁剪尺寸作为参数。它会在图像上进行中心裁剪，提取感兴趣区域。

2. **图像分割**：`split_image`函数将图像分割为一个小格子的网格。每个小格子将被独立分析。

3. **图像分析**：`analyze_image`函数计算输入图像中黑色、白色和特定颜色的像素比例。然后，它基于这些比例确定图像的类别（黑色、白色或背景色）。

4. **图像处理**：脚本调整图像大小，将其转换为灰度图像，应用高斯模糊，以及使用Canny算法进行边缘检测。

5. **矩阵生成**：对于每个裁剪和处理后的图像，脚本将根据每个小格子的分析结果生成一个类别矩阵。

6. **CSV输出**：各个矩阵合并为一个单独的矩阵，并将每个单独的矩阵保存为独立的CSV文件。CSV文件包含矩阵数据以及每个矩阵的标识符。

## 使用方法

1. 将输入图像放置在`pic_all`文件夹中。

2. 使用Python解释器运行脚本。

3. 脚本将处理每个图像，进行分析，生成各个矩阵，并将它们保存在`AUTO_combined_matrices.csv` CSV文件中。

## 需求

- Python 3.x
- OpenCV (cv2)
- NumPy

## 如何运行

1. 使用以下命令安装所需的库：
   ```
   pip install opencv-python numpy
   ```

2. 将此存储库克隆或下载到您的本地计算机。

3. 将输入图像放置在`pic_all`文件夹中。

4. 打开终端或命令提示符，导航到下载的存储库目录。

5. 使用以下命令运行脚本：
   ```
   python script.py
   ```

6. 脚本运行完成后，合并的矩阵和各个矩阵将保存在`AUTO_combined_matrices.csv`文件中。

## 注意事项

- 您可以在脚本中根据需要调整裁剪尺寸、分割行数和分割列数。

- 脚本假定您的输入图像位于`pic_all`文件夹中。如果您的文件夹结构不同，请调整文件路径。

- CSV文件将包含合并的矩阵和各个矩阵，每个矩阵都有相应的标识符。

- 脚本基于像素比例进行图像分析，可能需要根据特定用途进行调整。

- 脚本将矩阵保存为CSV文件，以便进行进一步的分析或可视化。

## 作者

@何杰




# Image Analysis and Matrix Generation

This Python script is designed to perform image analysis and generate matrices based on the characteristics of small cells within input images. It takes a collection of images, processes them, and produces a combined matrix as well as individual matrices saved to a CSV file.

## Features

1. **Center Crop**: The script provides a function `center_crop` that takes an input image and a crop size. It performs a center crop on the image to extract a region of interest.

2. **Image Splitting**: The `split_image` function divides an image into a grid of smaller cells. Each cell is analyzed independently.

3. **Image Analysis**: The `analyze_image` function calculates pixel ratios of black, white, and specific colors in the input image. It then determines the category of the image based on these ratios (black, white, or background).

4. **Image Processing**: The script resizes the images, converts them to grayscale, applies Gaussian blur, and performs edge detection using the Canny algorithm.

5. **Matrix Generation**: For each cropped and processed image, the script generates a matrix of categories based on the analysis results for each small cell.

6. **CSV Output**: The individual matrices are combined into a single matrix, and each individual matrix is saved as a separate CSV file. The CSV file contains the matrix data along with an identifier for each matrix.

## Usage

1. Place your input images in the `pic_all` folder.

2. Run the script using a Python interpreter.

3. The script will process each image, perform analysis, generate individual matrices, and save them in the `AUTO_combined_matrices.csv` CSV file.

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy

## How to Run

1. Install the required libraries using the following command:
   ```
   pip install opencv-python numpy
   ```

2. Clone or download this repository to your local machine.

3. Place your input images in the `pic_all` folder.

4. Open a terminal or command prompt and navigate to the downloaded repository's directory.

5. Run the script using the following command:
   ```
   python script.py
   ```

6. After the script finishes running, the combined matrix and individual matrices will be saved in the `AUTO_combined_matrices.csv` file.

## Notes

- You can modify the crop size, split rows, and split columns according to your needs in the script.

- The script assumes that your input images are located in the `pic_all` folder. Make sure to adjust the file paths if your folder structure is different.

- The CSV file will contain both the combined matrix and individual matrices, each with a corresponding identifier.

- The script performs image analysis based on pixel ratios, which may need adjustments for specific use cases.

- The script saves the matrices as CSV files for further analysis or visualization.

## Author

@HeJie





