# ==================== 第 4 步：训练模型 ====================
import copy
import time

num_epochs = 5

train_acc_history, val_acc_history = [], []
train_loss_history, val_loss_history = [], []

best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

since = time.time()

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print('-' * 30)

    # ---------- 训练阶段 ----------
    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataloaders['train']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(image_datasets['train'])
    epoch_acc = running_corrects.double() / len(image_datasets['train'])
    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc.item())

    # ---------- 验证阶段 ----------
    model.eval()
    val_loss = 0.0
    val_corrects = 0

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)

    epoch_val_loss = val_loss / len(image_datasets['val'])
    epoch_val_acc = val_corrects.double() / len(image_datasets['val'])
    val_loss_history.append(epoch_val_loss)
    val_acc_history.append(epoch_val_acc.item())

    print('Train Loss: %.4f Acc: %.4f | Val Loss: %.4f Acc: %.4f'
          % (epoch_loss, epoch_acc, epoch_val_loss, epoch_val_acc))

    # 保存最佳模型
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        best_model_wts = copy.deepcopy(model.state_dict())

time_elapsed = time.time() - since
print('')
print('✅ 训练完成！用时 %d分%d秒' % (time_elapsed // 60, time_elapsed % 60))
print('最佳验证准确率: %.4f' % best_acc)

# 加载最佳权重并保存
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'cat_dog_resnet50.pth')
print('✅ 模型已保存：cat_dog_resnet50.pth')

from google.colab import files
files.download('cat_dog_resnet50.pth')
