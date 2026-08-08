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

app= FastAPI() #под запросы
print("Start")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Прогноз] использование устройства:{device}", flush=True)

try:
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
@app.post("/api/recognize")
async def recognize_wine(data: ImageRequest):
    print("\n начало обработки", flush = True)
    try:
        pure_base64 = data.image_base64.split(",")[-1]
        image_data = base64.b64decode(pure_base64)
        image  =Image.open(io.BytesIO(image_data)).convert("RGB")
        print(f"фотку успешно декодировал. Размер:{image.size}", flush=True)
    except Exception as e:
        error_msg = f"не удалось прочитать base64 строку:{str(e)}"
        print(error_msg, flush=True)
        raise HTTPException(status_code=400, detail=error_msg)

    try:
        input_width = 640
        input_height = 640

        print(1)
        session = ort.InferenceSession("label_detector.onnx", providers=["CPUExecutionProvider"])
        print(session)
        input_name = session.get_inputs()[0].name

        orig_h, orig_w = image.shape[:2]

        resized_image = cv2.resize(image, (input_width, input_height))
        input_tensor = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        input_tensor = input_tensor.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        outputs = session.run(None, {input_name: input_tensor})
        prediction = outputs[0] 

        print("Инференс выполнен успешно. Формат вывода:", prediction.shape)
       
        pred = prediction[0] 
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
            
            scores = np.max(pred[:, 4:], axis=1)
            best_idx = np.argmax(scores)
            box = pred[best_idx, :4]
            xc, yc, w, h = box

            xc = xc * (orig_w )
            yc = yc * (orig_h)
            w = w * (orig_w )
            h = h * (orig_h)

            x_min = int(xc - w / 2)
            y_min = int(yc - h / 2)
            x_max = int(xc + w / 2)
            y_max = int(yc + h / 2)

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(orig_w, x_max)
            y_max = min(orig_h, y_max)

            if x_max > x_min and y_max > y_min:
                image = image[y_min:y_max, x_min:x_max]
    except Exception as e:
        error_msg = f"YOLO не смогла:{str(e)}"
        print(error_msg, flush=True)
        raise HTTPException(status_code=500, detail=error_msg)
    ####################################################################################################################
    try:
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(tensor).flatten().cpu().numpy().astype('float32')
        print(f"нейросеть сгенирировала вектор. Длина: {len(embedding)}", flush=True)
    except Exception as e:
        error_msg = f"Сбой при прогоне через EfficientNet: {str(e)}"
        print(error_msg, flush=True)
        raise HTTPException(status_code=500, detail=error_msg)

    try:
        dist, indecs = index.search(np.array([embedding]), k=1)
        wine_id = int(indecs[0][0])
        distance=float(dist[0][0])
        print(f"Faiss выдал индекс ближайшего соседа ID:{wine_id} Метрика:{distance:.4f}", flush = True)
        if wine_id == -1:
            raise Exception("FAISS не нашёл совпадений")
    except Exception as e:
        error_msg=f"Сбой поиска в векторной базе:{str(e)}"
        print(error_msg, flush=True)
        raise HTTPException(status_code=404, detail=error_msg)

    try:
        conn = get_db_connection()
        cursor=conn.cursor()
        cursor.execute("SELECT wine_slug FROM wines WHERE id =%s;", (wine_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if not result:
            raise Exception(f"Индекс {wine_id} есть в faiss, но записи с таким id нет в таблице sql")

        wine_slug = result[0]
        wine_url=f"https://vino-svoe.ru/wines/{wine_slug}"
        print(f" вино успешно извлечено slug'{wine_slug}'", flush =True)
    except Exception as e:
        error_msg = f"Ошибка при работе с SQL базой {str(e)}"
        print(error_msg, flush=True)
        raise HTTPException(status_code=500, detail=error_msg)

    print(f"запрашиваются данные с сайта", flush=True)

    headers = {'User-Agent':'Mozilla/5.0(Windows NT 10.0;Win64;x64) AppleWebKit/537.36'}
    try:
        response=requests.get(wine_url, headers=headers, timeout=5)
        if response.status_code==200:
            soup = BeautifulSoup(response.text, 'html.parser')
            description_tag=soup.find('p', class_='wine-page__description')
            description = description_tag.text.strip() if description_tag else "Нет описания"
            print(f"Данные успешно получены. Описание:{description}", flush=True)
        else:
            print(f"Сайт вернул код {response.status_code}", flush=True)
            description="Ошибка подключения к сайту"
    except Exception as e:
        print(f"Не удалось распарсить страницу:{e}", flush=True)
        description = "Не удалось загрузить информацию"
    print("Запрос полностью обработан", flush = True)

    return{
        "status": "success",
        "url": wine_url,
        "parsed_data":{
              "wine_name": wine_slug,
              "public_rating": 4.3,
              "guide_rating": 87.5,
              "short_info": description,
              "gigachat_insights": "Насыщенный букет с нотами чёрной смородины и ванили, танины хорошо структурированы..."
        }
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
