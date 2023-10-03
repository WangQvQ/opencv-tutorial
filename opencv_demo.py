import cv2  
import gradio as gr  
import numpy as np  


# 原始图像处理函数
def original_image(input_image):
    return input_image


# 灰度化处理函数
def grayscale(input_image):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    return gray_image


# 平移图像处理函数
def translate_image(input_image, translation_x, translation_y):
    rows, cols, _ = input_image.shape
    translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    translated_image = cv2.warpAffine(input_image, translation_matrix, (cols, rows))
    return translated_image


# Canny 边缘检测处理函数
def edge_detection(input_image):
    edges = cv2.Canny(input_image, 100, 200)
    return edges


# Sobel 边缘检测处理函数
def sobel_edge_detection(input_image):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))
    return sobel_magnitude


# 反转颜色处理函数
def invert_colors(input_image):
    inverted_image = cv2.bitwise_not(input_image)
    return inverted_image


# 腐蚀处理函数
def erosion(input_image, iterations):
    kernel = np.ones((5, 5), np.uint8)
    eroded_image = cv2.erode(input_image, kernel, iterations=iterations)
    return eroded_image


# 膨胀处理函数
def dilation(input_image, dilation_iterations):
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(input_image, kernel, iterations=dilation_iterations)
    return dilated_image


# 均值滤波处理函数
def mean_blur(input_image):
    mean_blurred_image = cv2.blur(input_image, (5, 5))
    return mean_blurred_image


# 中值滤波处理函数
def median_blur(input_image):
    median_blurred_image = cv2.medianBlur(input_image, 5)
    return median_blurred_image


# 高斯滤波处理函数
def gaussian_blur(input_image):
    gaussian_blurred_image = cv2.GaussianBlur(input_image, (5, 5), 0)
    return gaussian_blurred_image


# 双边滤波处理函数
def bilateral_filter(input_image):
    bilateral_filtered_image = cv2.bilateralFilter(input_image, 9, 75, 75)
    return bilateral_filtered_image


# 方块滤波处理函数
def box_filter(input_image):
    box_filtered_image = cv2.boxFilter(input_image, -1, (5, 5))
    return box_filtered_image


# 直方图均衡化处理函数
def histogram_equalization(input_image):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    return cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)


# 仿射变换处理函数
def affine_transform(input_image):
    # 创建仿射变换矩阵
    rows, cols, _ = input_image.shape
    matrix = cv2.getRotationMatrix2D((cols / 4, rows / 2), 70, 0.5)  # 90度旋转和1.5倍缩放
    result_image = cv2.warpAffine(input_image, matrix, (cols, rows))
    return result_image


# 透射变换处理函数
def perspective_transform(input_image):
    # 定义四个输入图像的角点坐标
    rows, cols, _ = input_image.shape
    # 修改pts1和pts2的值以减小透射变换的弯曲程度
    pts1 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
    pts2 = np.float32([[30, 30], [cols - 50, 50], [50, rows - 50], [cols - 50, rows - 50]])
    # 计算投射矩阵
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # 进行投射变换
    result_image = cv2.warpPerspective(input_image, matrix, (cols, rows))
    return result_image


# 自定义卷积核
def custom_filter(input_image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(input_image, -1, kernel)


# 图像旋转处理函数
def rotate_image(input_image, rotation_angle):
    rows, cols, _ = input_image.shape
    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
    result_image = cv2.warpAffine(input_image, matrix, (cols, rows))
    return result_image


# 创建 Gradio 接口
input_image = gr.inputs.Image()
method = gr.inputs.Radio(
    choices=["原图", "灰度化", "反转颜色", "平移", "直方图均衡化", "腐蚀", "膨胀", "均值滤波", "中值滤波", "高斯滤波",
             "双边滤波", "方块滤波", "仿射变换", "透射变换", "图像旋转", "Sobel边缘检测", "Canny边缘检测", "自定义卷积核"], default="原图")

rotation_angle = gr.inputs.Slider(minimum=-180, maximum=180, default=45, label="图像旋转: 旋转角度")
iterations = gr.inputs.Slider(minimum=0, maximum=10, step=1, default=1, label="腐蚀: 腐蚀参数")
dilation_iterations = gr.inputs.Slider(minimum=0, maximum=10, step=1, default=1, label="膨胀: 膨胀参数")
translation_x = gr.inputs.Slider(minimum=-200, maximum=200, default=200, label="平移: X轴平移")
translation_y = gr.inputs.Slider(minimum=-200, maximum=200, default=200, label="平移: Y轴平移")

output_image = gr.outputs.Image(type="pil")


# 创建函数根据下拉菜单的选择来执行不同的方法
def apply_opencv_methods(input_image, method, rotation_angle, iterations, dilation_iterations,
                         translation_x, translation_y):
    if method == "原图":
        return original_image(input_image)
    elif method == "图像旋转":
        return rotate_image(input_image, rotation_angle)
    elif method == "腐蚀":
        return erosion(input_image, iterations)
    elif method == "膨胀":
        return dilation(input_image, dilation_iterations)
    elif method == "Sobel边缘检测":
        return sobel_edge_detection(input_image)
    elif method == "平移":
        return translate_image(input_image, translation_x, translation_y)
    elif method == "自定义卷积核":
        return custom_filter(input_image)
    else:
        methods = {
            "灰度化": grayscale,
            "Canny边缘检测": edge_detection,
            "反转颜色": invert_colors,
            "均值滤波": mean_blur,
            "中值滤波": median_blur,
            "高斯滤波": gaussian_blur,
            "双边滤波": bilateral_filter,
            "方块滤波": box_filter,
            "仿射变换": affine_transform,
            "透射变换": perspective_transform,
            "直方图均衡化": histogram_equalization,
        }
        return methods[method](input_image)


# 创建 Gradio 接口
gr.Interface(
    fn=apply_opencv_methods,
    inputs=[input_image, method, rotation_angle, iterations, dilation_iterations, translation_x,
            translation_y],
    outputs=output_image,
    live=True,
    title="图像处理初学者导引",
    description="选择一张图像, 并选择对应方法"
).launch(share=False)
