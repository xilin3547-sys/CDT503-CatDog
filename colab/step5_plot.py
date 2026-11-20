# ==================== 第 5 步：画训练曲线 + 混淆矩阵 ====================
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics

# ---------- 图1：准确率曲线 ----------
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_acc_history, 'o-', label='Train Acc')
plt.plot(range(1, num_epochs + 1), val_acc_history, 's-', label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.grid(True)

# ---------- 图2：损失曲线 ----------
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_loss_history, 'o-', label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_loss_history, 's-', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
plt.show()
print('✅ 曲线图已保存：training_curves.png')

# ---------- 图3：混淆矩阵 ----------
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for inputs, labels in dataloaders['val']:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = metrics.confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(5, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# 在格子里写数字
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]),
                 ha='center', va='center',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print('✅ 混淆矩阵已保存：confusion_matrix.png')

# 下载图片（给报告用）
from google.colab import files
files.download('training_curves.png')
files.download('confusion_matrix.png')
