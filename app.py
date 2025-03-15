import base64
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
port = int(os.environ.get("PORT", 10000))
app = Flask(__name__)
CORS(app)

def get_resnet34(num_classes=32):
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Создаём модель с правильной архитектурой
model = get_resnet34(num_classes=32)

# Загружаем веса
model.load_state_dict(torch.load("curent6.pth", map_location=torch.device("cpu")))
model.eval()  # Переводим в режим предсказания

# Преобразование изображения
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),  # Подгоняем под размер модели
    transforms.ToTensor(),
])

@app.route("/")
def home():
    return "Hello, Render!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    # Декодируем изображение
    image_data = data["image"].split(",")[1]  # Убираем "data:image/png;base64,"
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    transformed_image = transform(image)
    transformed_image = transformed_image.unsqueeze(0)


    with torch.no_grad():
        output = model(transformed_image)
        predicted_class = torch.argmax(output, dim=1).item()

    return jsonify({"letter": chr(1040 + predicted_class)})  # Пример: 1040 = 'А' в Unicode


app.run(host="0.0.0.0", port=port)
