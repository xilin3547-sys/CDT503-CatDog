# ==================== 第 6 步：Gradio 演示网页（中英文双语） ====================
import subprocess
subprocess.run(['pip', 'install', 'gradio', '-q'])

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 1. 加载已训练好的模型 ----------
num_classes = 2
class_names_cn = ['猫', '狗']
class_names_en = ['Cat', 'Dog']

model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model.load_state_dict(torch.load('cat_dog_resnet50.pth', map_location=device))
model = model.to(device)
model.eval()

# ---------- 2. 图像预处理 ----------
infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------- 3. 中英文双语属性字典（示例数据） ----------
ATTR_CN = {
    '猫': {
        '智力': '中等～高',
        '体型': '小型～大型',
        '类型': '伴侣动物',
        '寿命': '12–18 年',
        '价格': '0–30,000 元',
        '描述': '猫咪独立安静，适合室内饲养，不需要遛。',
    },
    '狗': {
        '智力': '中等～极高',
        '体型': '超小型～超大型',
        '类型': '伴侣/工作/狩猎等',
        '寿命': '6–20 年',
        '价格': '0–60,000 元',
        '描述': '狗狗忠诚友好，需要遛狗和陪伴，不同品种差异很大。',
    },
}

ATTR_EN = {
    'Cat': {
        'Intelligence': 'Medium to High',
        'Size': 'Small to Large',
        'Type': 'Companion Animal',
        'Lifespan': '12-18 years',
        'Price': '0-30,000 CNY',
        'Description': 'Cats are independent and quiet, suitable for indoor keeping.',
    },
    'Dog': {
        'Intelligence': 'Medium to Very High',
        'Size': 'Extra Small to Extra Large',
        'Type': 'Companion / Working / Hunting',
        'Lifespan': '6-20 years',
        'Price': '0-60,000 CNY',
        'Description': 'Dogs are loyal and friendly, need walking and companionship.',
    },
}

# ---------- 4. 预测函数 ----------
def predict(image):
    if image is None:
        return {}, ''

    try:
        img = Image.fromarray(image).convert('RGB')
        input_tensor = infer_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            conf, idx = torch.max(probs, 0)

        idx = int(idx.item())
        pred_cn = class_names_cn[idx]
        pred_en = class_names_en[idx]

        # 左侧：概率标签
        result = {pred_en: float(conf.item())}

        # 右侧：双语 Markdown 属性表
        cn = ATTR_CN[pred_cn]
        en = ATTR_EN[pred_en]

        md = (
            '### 🐾 ' + pred_cn + ' / ' + pred_en + '\n'
            '**模型预测 / Model Prediction**: ' + pred_cn + ' / ' + pred_en
            + '（置信度 / Confidence: ' + format(conf.item(), '.2%') + '）\n\n'
            '| 属性 / Attribute | 中文 / Chinese | 英文 / English |\n'
            '|---|---|---|\n'
            '| 智力 / Intelligence | ' + cn['智力'] + ' | ' + en['Intelligence'] + ' |\n'
            '| 体型 / Size | ' + cn['体型'] + ' | ' + en['Size'] + ' |\n'
            '| 类型 / Type | ' + cn['类型'] + ' | ' + en['Type'] + ' |\n'
            '| 寿命 / Lifespan | ' + cn['寿命'] + ' | ' + en['Lifespan'] + ' |\n'
            '| 价格 / Price | ' + cn['价格'] + ' | ' + en['Price'] + ' |\n\n'
            '**描述 / Description**: ' + cn['描述'] + ' / ' + en['Description'] + '\n\n'
            '> ⚠️ 注意 / Note: 属性为测试用示例数据，非真实统计。'
            'Attributes are test example data, not real statistics.\n'
        )

        return result, md

    except Exception as e:
        return {}, '❌ 错误 / Error: ' + str(e)

# ---------- 5. 构建界面 ----------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='numpy', label='上传猫狗照片 / Upload Cat or Dog Photo'),
    outputs=[
        gr.Label(num_top_classes=2, label='预测结果 / Prediction Result'),
        gr.Markdown(label='属性说明 / Attribute Description (中英文)'),
    ],
    title='猫狗识别原型 / Cat-Dog Recognition Prototype',
    description=(
        '上传一张猫或狗的照片，模型识别品类，并显示中英文双语属性参考。'
        'Upload a cat or dog photo, the model identifies the category '
        'and displays bilingual attribute reference.'
    ),
)

# ---------- 6. 启动 ----------
demo.launch(share=True)
