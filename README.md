# 🐾 Cat-Dog Recognition Prototype / 猫狗识别原型

**CDT503 Final Group Project — Lingnan University**
Course Instructor: Dr. Pan Fei (潘飞博士)

---

## 📖 Introduction / 项目简介

A computer vision prototype that classifies uploaded photos as **cat** or **dog**
using a ResNet50 transfer-learning model, and displays a **bilingual (中/EN)
attribute table** for the predicted category.

本项目是一个计算机视觉原型：基于 **ResNet50 迁移学习**对上传照片进行
**猫 / 狗二分类**，并以**中英文双语表格**展示预测类别的属性参考。

---

## 🖥️ Live Demo / 在线演示

| | Link |
|---|---|
| 🌐 Hugging Face Spaces | **[待填写 — 部署后更新]** |
| 🎬 Demo Video / 演示录屏 | **[待填写 — 录屏后更新]** |

> ⚠️ Colab 生成的 `*.gradio.live` 链接约 24 小时失效，仅作现场演示用；
> 永久链接请以 Hugging Face Spaces 为准。

---

## 📊 Model Performance / 模型性能

| Metric / 指标 | Value / 数值 |
|---|---|
| Architecture / 架构 | ResNet50 (transfer learning / 迁移学习) |
| Dataset / 数据集 | cat&dog_dataset_resized_256 |
| Classes / 类别 | cat, dog (binary / 二分类) |
| Best Val Accuracy / 最佳验证准确率 | **待填写** |
| Training Device / 训练设备 | Google Colab GPU |

**Training Curves / 训练曲线**（第 5 步生成，放入 `assets/`）：

![Accuracy and Loss Curves](assets/training_curves.png)

**Confusion Matrix / 混淆矩阵**：

![Confusion Matrix](assets/confusion_matrix.png)

---

## 🗂️ Repository Structure / 仓库结构

```
CDT503-CatDog/
├── README.md              # 本文件
├── app.py                 # Gradio demo (Hugging Face Spaces)
├── requirements.txt       # Python dependencies
├── colab/
│   ├── step2_data.py      # 数据预处理
│   ├── step3_model.py     # 模型搭建 (ResNet50)
│   ├── step4_train.py     # 训练 + 保存权重
│   ├── step5_plot.py      # 曲线图 + 混淆矩阵
│   └── step6_gradio.py    # Gradio 演示网页
├── assets/                # 曲线图、混淆矩阵、演示截图
└── cat_dog_resnet50.pth   # 训练好的模型权重
```

---

## 🚀 How to Run / 运行方法

### Option 1: Google Colab (training / 训练)

1. Upload the dataset zip and unzip it
2. Run `colab/step2_data.py` → `step3_model.py` → `step4_train.py` → `step5_plot.py`
3. Run `colab/step6_gradio.py` to get a temporary demo link

### Option 2: Local / Hugging Face Spaces (inference / 推理)

```bash
pip install -r requirements.txt
python app.py
```

---

## 👥 Team / 小组成员

| Member | Role / 分工 |
|---|---|
| Member A | Background & dataset / 背景与数据 |
| Member B | Model building & training / 模型搭建与训练 |
| Member C | Report & presentation / 报告与演示 |

---

## 📄 Report / 报告

LaTeX IEEEtran report available in TeXPage (see group shared link).

---

## ⚠️ Disclaimer / 免责声明

The attribute tables shown in the demo are **test example data** for
prototyping purposes, not real statistical data.
演示中的属性表为**测试用示例数据**，仅用于原型展示，非真实统计数据。
