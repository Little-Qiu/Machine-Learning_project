import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为tensor
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # CIFAR-10的均值和标准差
])

# 下载CIFAR-10数据集
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

# Step 2: 将数据加载到内存
data = []
labels = []
for inputs, targets in dataloader:
    data.append(inputs.numpy())  # 将数据转换为 numpy 数组
    labels.append(targets.numpy())  # 标签也转换为 numpy 数组

data = np.concatenate(data, axis=0)  # 所有数据
labels = np.concatenate(labels, axis=0)  # 所有标签

# Step 3: 数据预处理 - 将图像数据展平成二维数组
n_samples, n_channels, height, width = data.shape
data = data.reshape(n_samples, -1)  # 将每张图片展平

# Step 4: KMeans 聚类
kmeans = KMeans(n_clusters=10, random_state=42, init='k-means++', max_iter=300, tol=1e-4)
kmeans.fit(data)  # 直接对所有数据进行聚类


# Step 5: 聚类结果与真实标签匹配
def map_labels(kmeans, y_true):
    y_pred = kmeans.labels_
    mapped_labels = np.zeros_like(y_pred)
    for i in range(10):  # 10个簇
        # 获取簇 i 中的所有样本的真实标签
        cluster_labels = y_true[y_pred == i]
        # 获取该簇中最常见的标签
        mapped_labels[y_pred == i] = np.bincount(cluster_labels).argmax()
    return mapped_labels

# 映射聚类标签到真实标签
y_pred_mapped = map_labels(kmeans, labels)

# Step 6: 计算准确率
accuracy = accuracy_score(labels, y_pred_mapped)
print(f'KMeans Accuracy: {accuracy * 100:.2f}%')

# Step 7: 计算精确率、召回率和F1分数
precision, recall, fscore, _ = precision_recall_fscore_support(labels, y_pred_mapped, average='weighted', zero_division=0)
print(f'Precision (Weighted): {precision:.4f}')
print(f'Recall (Weighted): {recall:.4f}')
print(f'F1 Score (Weighted): {fscore:.4f}')

# Step 8: 显示分类报告
print("\nClassification Report:")
print(classification_report(labels, y_pred_mapped, zero_division=0))

# Step 9: 可视化混淆矩阵
cm = confusion_matrix(labels, y_pred_mapped)

# 使用Seaborn绘制热力图
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.title('KMeans Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Step 10: 聚类结果的可视化（前两维降维）
from sklearn.decomposition import PCA

# 降维到2D以便可视化
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

plt.figure(figsize=(10, 8))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_pred_mapped, cmap='tab10', s=20)
plt.title('KMeans Clustering Results (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.show()

'''import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights

# Step 1: 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为tensor
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # CIFAR-10的均值和标准差
])

# 下载CIFAR-10数据集
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

# Step 2: 加载 VGG16 预训练模型
vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)

# 提取特征和平均池化层
features_layer = vgg16.features
avgpool_layer = vgg16.avgpool

vgg16.eval()  # 设置为评估模式

# Step 3: 提取图像特征（包括 features 和 avgpool 层的输出）
data = []
labels = []
for inputs, targets in dataloader:
    with torch.no_grad():  # 禁用梯度计算，减少内存消耗
        # 获取 features 层的输出
        features_output = features_layer(inputs)

        # 获取 avgpool 层的输出
        avgpool_output = avgpool_layer(features_output)

        # 展平并拼接 features 输出和 avgpool 输出
        features_output_flattened = features_output.view(features_output.size(0), -1)  # Flatten features
        avgpool_output_flattened = avgpool_output.view(avgpool_output.size(0), -1)  # Flatten avgpool output

        # 拼接两个特征向量
        combined_features = torch.cat((features_output_flattened, avgpool_output_flattened), dim=1)

        data.append(combined_features.numpy())  # 将数据转换为 numpy 数组
        labels.append(targets.numpy())  # 标签也转换为 numpy 数组

# 合并所有数据
data = np.concatenate(data, axis=0)
labels = np.concatenate(labels, axis=0)

# Step 4: KMeans 聚类（增加 n_init 参数）
kmeans = KMeans(n_clusters=10, random_state=42, init='k-means++', max_iter=300, tol=1e-4, n_init=20)
kmeans.fit(data)  # 使用提取的特征进行聚类


# Step 5: 聚类结果与真实标签匹配
def map_labels(kmeans, y_true):
    y_pred = kmeans.labels_
    mapped_labels = np.zeros_like(y_pred)
    for i in range(10):  # 10个簇
        # 获取簇 i 中的所有样本的真实标签
        cluster_labels = y_true[y_pred == i]
        # 获取该簇中最常见的标签
        mapped_labels[y_pred == i] = np.bincount(cluster_labels).argmax()
    return mapped_labels


# 映射聚类标签到真实标签
y_pred_mapped = map_labels(kmeans, labels)

# Step 6: 计算准确率
accuracy = accuracy_score(labels, y_pred_mapped)
print(f'KMeans Accuracy: {accuracy * 100:.2f}%')

# Step 7: 计算精确率、召回率和F1分数
precision, recall, fscore, _ = precision_recall_fscore_support(labels, y_pred_mapped, average='weighted',
                                                               zero_division=0)
print(f'Precision (Weighted): {precision:.4f}')
print(f'Recall (Weighted): {recall:.4f}')
print(f'F1 Score (Weighted): {fscore:.4f}')

# Step 8: 显示分类报告
print("\nClassification Report:")
print(classification_report(labels, y_pred_mapped, zero_division=0))

# Step 9: 可视化混淆矩阵
cm = confusion_matrix(labels, y_pred_mapped)

# 使用Seaborn绘制热力图
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.title('KMeans Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Step 10: 聚类结果的可视化（前两维降维）
from sklearn.decomposition import PCA

# 使用PCA降维到2D
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

plt.figure(figsize=(10, 8))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_pred_mapped, cmap='tab10', s=20)
plt.title('KMeans Clustering Results (PCA Reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar()
plt.show()'''
