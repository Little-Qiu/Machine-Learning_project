import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

#提取原始图像特征并用于训练KNN
def extract_features(data_loader):
    features = []
    labels = []
    for inputs, targets in data_loader:
        inputs = inputs.view(inputs.size(0), -1).numpy()  # 展平图像
        features.append(inputs)
        labels.append(targets.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

# 提取训练集和测试集的特征
X_train, y_train = extract_features(trainloader)
X_test, y_test = extract_features(testloader)
X_train_small, y_train_small = X_train[:50000], y_train[:50000]  # 使用较小的子集进行实验

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 对训练集进行标准化
X_test_scaled = scaler.transform(X_test)  # 对测试集进行标准化
X_train_scaled_small = scaler.fit_transform(X_train_small)  # 小样本集也要标准化

#创建 KNN 分类器
knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', n_jobs=-1)

#训练和评估模型
train_accuracies = []
val_accuracies = []

sample_sizes = [int(0.1 * len(X_train_small)), int(0.3 * len(X_train_small)), int(0.5 * len(X_train_small)),
                int(0.7 * len(X_train_small)), len(X_train_small)]  # 增加训练样本的数量

for sample_size in sample_sizes:
    #选择部分训练数据进行训练
    X_train_subset = X_train_scaled_small[:sample_size]
    y_train_subset = y_train_small[:sample_size]

    # 训练 KNN 分类器
    knn_classifier.fit(X_train_subset, y_train_subset)

    #在训练集上评估性能
    y_train_pred = knn_classifier.predict(X_train_subset)
    train_accuracy = accuracy_score(y_train_subset, y_train_pred)

    #在测试集上评估性能
    y_pred = knn_classifier.predict(X_test_scaled)
    val_accuracy = accuracy_score(y_test, y_pred)

    # 记录准确率
    train_accuracies.append(train_accuracy * 100)
    val_accuracies.append(val_accuracy * 100)

    # 输出每个样本大小下的训练和验证结果
    print(f"Sample size: {sample_size} - "
          f"Train Accuracy: {train_accuracy * 100:.2f}% - "
          f"Val Accuracy: {val_accuracy * 100:.2f}%")

#绘制学习曲线
plt.figure(figsize=(12, 6))
plt.plot(sample_sizes, train_accuracies, label='Train Accuracy')
plt.plot(sample_sizes, val_accuracies, label='Val Accuracy')
plt.title('Learning Curve: Accuracy vs. Training Sample Size')
plt.xlabel('Training Sample Size')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.show()

#输出最终性能
print(f'KNN Classifier Accuracy: {val_accuracies[-1]:.2f}%')

# 计算精确率、召回率和F1分数
precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
print(f'Precision (Weighted): {precision:.4f}')
print(f'Recall (Weighted): {recall:.4f}')
print(f'F1 Score (Weighted): {fscore:.4f}')

# 显示分类报告
print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred))

#混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=trainset.classes, yticklabels=trainset.classes)
plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
