# 导入必要的包
import numpy as np
import cv2

# 初始化类标签并设置伪随机数生成器的种子，这样我们就可以重新生成结果
labels = ["dog", "cat"]
np.random.seed(1)

# 随机初始化我们的权重矩阵和偏差向量——在一个*真实的*训练和分类任务中，这些参数将由我们的模型*学习*，但是为了这个例子的目的，让我们使用随机值
W = np.random.randn(2, 3072)
b = np.random.randn(2)

# 加载我们的示例图像，调整它的大小，然后将其平展到我们的“特征向量”表示中
orig = cv2.imread("dog.jpg")
image = cv2.resize(orig, (32, 32)).flatten()

# 通过计算权重矩阵和图像像素之间的点积来计算输出分数，然后添加偏差
scores = W.dot(image) + b

# 循环评分+标签并显示它们
for (label, score) in zip(labels, scores):
	print("[INFO] {}: {:.2f}".format(label, score))

# 画出图像上得分最高的标签作为我们的预测
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# 显示输入图像
cv2.imshow("Image", orig)
cv2.waitKey(0)





