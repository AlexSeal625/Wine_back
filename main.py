import base64
import io
import os
import sys
import asyncio
from csv import excel
from dbm import error
from http.client import responses
from contextlib import asynccontextmanager

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from  pydantic import BaseModel
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import faiss
import psycopg2
import numpy as np
import cv2
import onnxruntime as ort
import json
from fastapi.responses import Response

app= FastAPI() #под запросы
print("Start")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Прогноз] использование устройства:{device}", flush=True)

try:
    session = ort.InferenceSession("label_detector.onnx", providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    model = models.efficientnet_b0(pretrained=True) #стандартная предобученная модель EfficientNet
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval() #режим оценивания
    model.to(device)
    print("Модель загружена", flush=True)
except Exception as e:
    print(f"ошибкаб модель не заружена:{e}", flush=True)
    sys.exit(1)

transform = transforms.Compose([
    transforms.Resize(256),   #Сначала сжимаем до 256 пикселей
    transforms.CenterCrop(224), #Вырезаем квадрат 224*224
    transforms.ToTensor(), #превращаем картинку в математический тензор
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) #нормализуем цвета
])
DEBUG_SAVE_CROPS = True  # выключить перед боевым использованием
DEBUG_CROPS_DIR = "debug_crops"
DEBUG_LOG_CHUNK_SIZE = 4000  # некоторые лог-вьюеры (в т.ч. Render) режут очень длинные строки

def save_crop_for_debugging(image: Image.Image, label: str = "crop") -> None:
    """
    Отладочное сохранение кропа этикетки. Диск на Render free tier
    эфемерный (пропадает при рестарте/редеплое) и шелл-доступа обычно нет —
    поэтому основной, гарантированно работающий канал это base64 в stdout
    (Render Dashboard -> Logs). Сохранение на диск — как бонус для
    локального запуска.
    """
    if not DEBUG_SAVE_CROPS:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{label}_{timestamp}.jpg"

    # 1) Пытаемся сохранить на диск — сработает локально, на Render в лучшем
    #    случае временно (до рестарта контейнера).
    try:
        os.makedirs(DEBUG_CROPS_DIR, exist_ok=True)
        filepath = os.path.join(DEBUG_CROPS_DIR, filename)
        image.save(filepath, format="JPEG", quality=90)
        print(f"[DEBUG] Кроп сохранён на диск: {filepath}", flush=True)
    except Exception as e:
        print(f"[DEBUG] Не удалось сохранить кроп на диск: {e}", flush=True)

    # 2) Base64 в консоль — единственный канал, который точно доступен
    #    на Render free tier без шелл-доступа.
    try:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        print(f"[DEBUG_CROP_BASE64_START] {filename} size={len(b64)}", flush=True)
        for i in range(0, len(b64), DEBUG_LOG_CHUNK_SIZE):
            print(b64[i:i + DEBUG_LOG_CHUNK_SIZE], flush=True)
        print(f"[DEBUG_CROP_BASE64_END] {filename}", flush=True)
    except Exception as e:
        print(f"[DEBUG] Не удалось закодировать кроп в base64: {e}", flush=True)
        
INDEX_FILE_PATH = "wines_base.index"
index = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index
    if not os.path.exists(INDEX_FILE_PATH):
        raise FileNotFoundError(f"Файл индекса {INDEX_FILE_PATH} не найден!")
    index = faiss.read_index(INDEX_FILE_PATH)
    print(f" База FAISS успешно загружена! Всего векторов в базе: {index.ntotal}")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        with open("init.sql", "r", encoding="utf-8") as f:
            sql_script = f.read()
        cursor.execute(sql_script)
        conn.commit()
        cursor.close()
        conn.close()
        print(" Таблицы SQL успешно инициализированы!")
    except Exception as e:
        print(f" Ошибка инициализации БД: {e}")
    yield
    print(" Сервер останавливается.")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#настройка структуры запроса
class ImageRequest(BaseModel):
    image_base64: str

def get_db_connection():
    return psycopg2.connect(
    host="dpg-d9q8f6rm8hqs73e6hbp0-a",
    database="wine_db_p4pv",
    user="wine_user",
    password="rSnOHgBrVlYBsTJGz4A0qAJWQ9Bd56xi"
    )
print("Ожидание запросов\n", flush=True)
# обработка запросов

# выносим тяжелую математику в отдельную синхронную функцию
def run_ml_pipeline(image, orig_w, orig_h, input_width, input_height):
    # 3. Конвертируем PIL в NumPy array для OpenCV
    img_np = np.array(image)
    
    # 4. Ресайз и подготовка тензора для YOLO
    resized_image = cv2.resize(img_np, (input_width, input_height))
    input_tensor = resized_image.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    # 5. Запуск модели
    outputs = session.run(None, {input_name: input_tensor})
    prediction = outputs[0]  
    
    # 6. Приводим матрицу к виду (8400, N)
    pred = prediction[0] 
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    
    # 7. Извлекаем лучшие предсказания
    scores = np.max(pred[:, 4:], axis=1)
    best_idx = np.argmax(scores)
    
    # Порог уверенности
    if scores[best_idx] > 0.25:
        box = pred[best_idx, :4]
        xc, yc, w, h = box
        
        # 8. Пересчет координат
        scale_x = orig_w / input_width
        scale_y = orig_h / input_height
        xc = xc * scale_x
        yc = yc * scale_y
        w = w * scale_x
        h = h * scale_y
        
        x_min = max(0, int(xc - w / 2))
        y_min = max(0, int(yc - h / 2))
        x_max = min(orig_w, int(xc + w / 2))
        y_max = min(orig_h, int(yc + h / 2))
        
        # 9. Кропаем
        if x_max > x_min and y_max > y_min:
            image = image.crop((x_min, y_min, x_max, y_max))
            save_crop_for_debugging(image, label="yolo_crop")
    # Извлечение фичей
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor).flatten().cpu().numpy().astype('float32')
        
    # Поиск в FAISS
    dist, indecs = index.search(np.array([embedding]), k=1)
    wine_id = int(indecs[0][0])
    distance = float(dist[0][0])
    
    if wine_id == -1:
        raise Exception("FAISS не нашёл совпадений")
        
    return wine_id, distance

# выносим сеть и БД в отдельную синхронную функцию
def fetch_wine_data(wine_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT wine_slug FROM wines WHERE id =%s;", (wine_id,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()

    if not result:
        raise Exception(f"Индекс {wine_id} есть в faiss, но записи с таким id нет в таблице SQL")

    wine_slug = result[0]
    wine_url = f"https://vino-svoe.ru/wines/{wine_slug}"
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(wine_url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            description_tag = soup.find('p', class_='wine-page__description')
            description = description_tag.text.strip() if description_tag else "Нет описания"
        else:
            description = "Ошибка подключения к сайту"
    except Exception:
        description = "Не удалось загрузить информацию"
        
    return wine_url, wine_slug, description

@app.post("/api/recognize")
async def recognize_wine(data: ImageRequest):
    print("\n[START] Начало обработки", flush=True)
    
    try:
        # Декодирование (быстрая операция, оставляем тут)
        pure_base64 = data.image_base64.split(",")[-1]
        image_data = base64.b64decode(pure_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        orig_w, orig_h = image.size
        print(f"Фотку успешно декодировал. Размер: {image.size}", flush=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать base64 строку: {str(e)}")

    try:
        # ЗАПУСК ТЯЖЕЛОГО ML В ОТДЕЛЬНОМ ПОТОКЕ (Event Loop не блокируется!)
        wine_id, distance = await asyncio.to_thread(
            run_ml_pipeline, image, orig_w, orig_h, 640, 640
        )
        print(f"Faiss выдал ID: {wine_id} Метрика: {distance:.4f}", flush=True)
        
        # ЗАПУСК СЕТИ И БД В ОТДЕЛЬНОМ ПОТОКЕ (Event Loop не блокируется!)
        wine_url, wine_slug, description = await asyncio.to_thread(fetch_wine_data, wine_id)
        print(f"Данные собраны. Slug: {wine_slug}", flush=True)

    except Exception as e:
        print(f"[ERROR] Сбой в пайплайне: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

    # Формируем итоговый словарь
    payload = {
        "status": "success",
        "url": wine_url,
        "parsed_data": {
              "wine_name": wine_slug,
              "public_rating": 4.3,
              "guide_rating": 87.5,
              "short_info": description,
              "gigachat_insights": "Насыщенный букет с нотами чёрной смородины и ванили..."
        }
    }

    # ФИКС ЧАНКОВ: Явно пакуем в JSON и считаем байты
    json_str = json.dumps(payload, ensure_ascii=False)
    json_bytes = json_str.encode("utf-8")
    
    print("[END] Запрос успешно обработан, отправляем ответ.", flush=True)

    # Возвращаем жестко зафиксированный ответ
    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={"Content-Length": str(len(json_bytes)),
                "Cache-Control": "no-transform"
                }
    )
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
