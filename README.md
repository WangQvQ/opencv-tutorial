

# OpenCV 方法演示项目



## 项目简介

这个开源项目是一个用于演示 OpenCV 方法的工具，旨在帮助初学者快速理解和掌握 OpenCV 图像处理技术。通过这个项目，你可以轻松地对图像进行各种处理，从灰度化到边缘检测，以及更多其他方法。项目使用 Gradio 创建用户友好的界面，让用户能够轻松选择不同的图像处理方法和参数。

## 为什么选择这个项目

- **教育性**：这个项目的主要目的是教育。它提供了对 OpenCV 方法的实际演示，以帮助初学者更好地理解和掌握这些技术。

- **互动性**：通过 Gradio 创建的用户界面，用户可以立即看到不同处理方法的效果，并可以自己调整参数，以更深入地理解每种方法的工作原理。

- **适用广泛**：这个项目可以帮助广大初学者，无论是学习计算机视觉、图像处理，还是对 OpenCV 有兴趣的人都会受益。

## 特性

- 提供了多种 OpenCV 图像处理方法的演示，包括灰度化、反转颜色、平移、直方图均衡化、腐蚀、膨胀、均值滤波、中值滤波、高斯滤波等。

- 支持自定义卷积核，允许用户尝试不同的卷积核来处理图像。

- 提供图像旋转、仿射变换和透射变换的演示，以及选择角度和参数的选项。

- 使用 Gradio 创建用户友好的界面，让用户能够轻松选择不同的图像处理方法和参数。

## 使用方法

1. **获取项目**：首先，你需要将这个项目克隆到你的本地计算机上。你可以使用以下命令来获取项目：

   ```bash
   git clone https://github.com/WangQvQ/opencv-tutorial.git
   ```

2. **安装依赖项**：确保你已经安装了以下依赖项：

   - OpenCV
   - Gradio
   - NumPy 

   如果你没有安装它们，你可以使用以下命令安装：

   ```bash
   pip install opencv-python-headless=4.7.0.72 gradio=3.1.5 numpy=1.22.4
   ```

3. **运行项目**：使用以下命令来运行项目：

   ```bash
   python opencv_demo.py
   ```

   运行后，你将看到一个网址，通常是 `http://localhost:7860`，你可以在浏览器中访问它。

4. **使用界面**：在浏览器中，你可以上传图像并选择不同的处理方法和参数，然后查看处理后的图像效果。

## 示例代码




以下是部分方法的代码示例:

```python
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
```

## 贡献

如果你对项目有任何改进或建议，欢迎贡献代码或提出问题。我们欢迎开发者共同改进这个项目，以使其更加有用和友好。如果你想贡献，请查看我们的[贡献指南](CONTRIBUTING.md)。

