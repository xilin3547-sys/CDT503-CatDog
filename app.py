# -*- coding: utf-8 -*-
"""CDT503 Cat-Dog Recognition Prototype - Gradio App"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr

# ---------- 1. Load trained model ----------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "cat_dog_resnet50.pth")

class_names_cn = ["猫", "狗"]
class_names_en = ["Cat", "Dog"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ---------- 2. Image preprocessing ----------
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------- 3. Bilingual attribute dictionary (test data) ----------
ATTR_CN = {
    "猫": ["中等～高", "小型～大型", "伴侣动物", "12–18 年", "0–30,000 元",
           "猫咪独立安静，适合室内饲养，不需要遛。"],
    "狗": ["中等～极高", "超小型～超大型", "伴侣/工作/狩猎等", "6–20 年", "0–60,000 元",
           "狗狗忠诚友好，需要遛狗和陪伴，不同品种差异很大。"],
}

ATTR_EN = {
    "Cat": ["Medium to High", "Small to Large", "Companion Animal",
            "12-18 years", "0-30,000 CNY",
            "Cats are independent and quiet, suitable for indoor keeping."],
    "Dog": ["Medium to Very High", "Extra Small to Extra Large",
            "Companion / Working / Hunting", "6-20 years", "0-60,000 CNY",
            "Dogs are loyal and friendly, need walking and companionship."],
}


# ---------- 4. Prediction function ----------
def predict(image):
    if image is None:
        return {}, ""
    try:
        img = Image.fromarray(image).convert("RGB")
        x = infer_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(x)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, idx = torch.max(probs, 0)
        i = int(idx.item())
        cn = class_names_cn[i]
        en = class_names_en[i]

        result = {en: float(conf.item())}

        a = ATTR_CN[cn]
        b = ATTR_EN[en]
        md = "### 🐾 " + cn + " / " + en + "\n"
        md += "**模型预测 / Model Prediction**: " + cn + " / " + en
        md += "（置信度 / Confidence: " + format(conf.item(), ".2%") + "）\n\n"
        md += "| 属性 / Attribute | 中文 / Chinese | 英文 / English |\n"
        md += "|---|---|---|\n"
        md += "| 智力 / Intelligence | " + a[0] + " | " + b[0] + " |\n"
        md += "| 体型 / Size | " + a[1] + " | " + b[1] + " |\n"
        md += "| 类型 / Type | " + a[2] + " | " + b[2] + " |\n"
        md += "| 寿命 / Lifespan | " + a[3] + " | " + b[3] + " |\n"
        md += "| 价格 / Price | " + a[4] + " | " + b[4] + " |\n\n"
        md += "**描述 / Description**: " + a[5] + " / " + b[5] + "\n\n"
        md += "> ⚠️ 注意 / Note: 属性为测试用示例数据，非真实统计。"
        md += "Attributes are test example data, not real statistics.\n"
        return result, md
    except Exception as e:
        return {}, "❌ 错误 / Error: " + str(e)


# ---------- 5. Build interface ----------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="上传猫狗照片 / Upload Cat or Dog Photo"),
    outputs=[
        gr.Label(num_top_classes=2, label="预测结果 / Prediction Result"),
        gr.Markdown(label="属性说明 / Attribute Description (中英文)"),
    ],
    title="猫狗识别原型 / Cat-Dog Recognition Prototype",
    description=(
        "上传一张猫或狗的照片，模型识别品类，并显示中英文双语属性参考。"
        "Upload a cat or dog photo, the model identifies the category "
        "and displays bilingual attribute reference."
    ),
)

if __name__ == "__main__":
    demo.launch()
