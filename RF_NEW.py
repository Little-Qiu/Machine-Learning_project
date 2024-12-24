import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from skimage.feature import hog
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.models as models

# 确定设备，优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为tensor
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  # CIFAR-10的均值和标准差
])

# 下载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=0)


# Step 2: 加载ResNet18模型并移除分类头
vgg16 = models.resnet18(weights='IMAGENET1K_V1')  # 使用预训练ResNet-18
vgg16_features = nn.Sequential(*list(vgg16.children())[:-1])  # 移除分类头，只保留特征提取部分

# 将模型迁移到GPU
vgg16_features = vgg16_features.to(device)

# 设置模型为评估模式
vgg16_features.eval()


# Step 3: 特征提取
def extract_resnet_features(data_loader):
    features = []
    labels = []
    with torch.no_grad():  # 禁用梯度计算，提高速度
        for inputs, targets in data_loader:
            # 将输入数据迁移到GPU
            inputs = inputs.to(device)

            # 提取特征
            feature = vgg16_features(inputs)  # 得到形状为 [batch_size, 512, 1, 1] 的特征
            feature = feature.view(feature.size(0), -1)  # 将特征展平为 [batch_size, 512]

            # 收集特征和标签
            features.append(feature.cpu().numpy())  # 将特征转移回CPU
            labels.append(targets.numpy())

    return np.concatenate(features), np.concatenate(labels)


# 提取训练集和测试集的特征
X_train, y_train = extract_resnet_features(trainloader)
X_test, y_test = extract_resnet_features(testloader)

# Step 4: 训练随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 5: 训练并评估模型
epochs = 10
sample_sizes = [int(0.1 * len(X_train)), int(0.3 * len(X_train)), int(0.5 * len(X_train)),
                int(0.7 * len(X_train)), len(X_train)]  # 增加训练样本的数量

train_accuracies = []
val_accuracies = []

for sample_size in sample_sizes:
    # 选择部分训练数据进行训练
    X_train_subset = X_train[:sample_size]
    y_train_subset = y_train[:sample_size]

    # Step 5.1: 随机森林训练
    rf_classifier.fit(X_train_subset, y_train_subset)

    # Step 5.2: 在训练集上评估性能
    y_train_pred = rf_classifier.predict(X_train_subset)
    train_accuracy = accuracy_score(y_train_subset, y_train_pred)

    # Step 5.3: 在测试集上评估性能
    y_test_pred = rf_classifier.predict(X_test)
    val_accuracy = accuracy_score(y_test, y_test_pred)

    # 记录准确率
    train_accuracies.append(train_accuracy * 100)  # 转换为百分比
    val_accuracies.append(val_accuracy * 100)  # 转换为百分比

    # 输出每个样本大小下的训练和验证结果
    print(f"Sample size: {sample_size} - "
          f"Train Accuracy: {train_accuracy * 100:.2f}% - "
          f"Val Accuracy: {val_accuracy * 100:.2f}%")

# Step 6: 绘制学习曲线
plt.figure(figsize=(12, 6))

# 绘制训练和验证准确率
plt.plot(sample_sizes, train_accuracies, label='Train Accuracy')
plt.plot(sample_sizes, val_accuracies, label='Val Accuracy')
plt.title('Learning Curve: Accuracy vs. Training Sample Size')
plt.xlabel('Training Sample Size')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

# Step 7: 评估性能并显示混淆矩阵
print(f'Random Forest Classifier Accuracy: {val_accuracies[-1]:.2f}%')

# 计算精确率、召回率和F1分数
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')
print(f'Precision (Weighted): {precision:.4f}')
print(f'Recall (Weighted): {recall:.4f}')
print(f'F1 Score (Weighted): {fscore:.4f}')

# 显示分类报告
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_test_pred))

# 可视化混淆矩阵
cm = confusion_matrix(y_test, y_test_pred)

# 使用Seaborn绘制热力图
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=trainset.classes, yticklabels=trainset.classes)
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
