import base64
import io
import os
import sys
import asyncio
from csv import excel
from dbm import error
from http.client import responses

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from  pydantic import BaseModel
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import faiss
import psycopg2
import numpy as np

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


try:
    D = 1280
    faiss_index = faiss.IndexFlatL2(D)
    pure_test="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"
    image_data_test = base64.b64decode(pure_test)
    image_test = Image.open(io.BytesIO(image_data_test)).convert("RGB")
    fake_vector = transform(image_test).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding_test = model(fake_vector).cpu().numpy().astype('float32').reshape(1,-1)
    faiss_index.add(embedding_test)
    print(f"нейросеть сгенирировала вектор. Длина: {len(embedding_test)}", flush=True)
except Exception as e:
    print(f"[КРИТИЧЕСКАЯ ОШИБКА] Ошибка инициализации FAISS: {e}", flush=True)
    sys.exit(1)

#настройка структуры запроса
class ImageRequest(BaseModel):
    image_base64: str

def get_db_connection():
    return psycopg2.connect(
    host="db",
    database="wine_db",
    user="wine_user",
    password="wine_password"
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
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(tensor).flatten().cpu().numpy().astype('float32')
        print(f"нейросеть сгенирировала вектор. Длина: {len(embedding)}", flush=True)
    except Exception as e:
        error_msg = f"Сбой при прогоне через EfficientNet: {str(e)}"
        print(error_msg, flush=True)
        raise HTTPException(status_code=500, detail=error_msg)

    try:
        dist, indecs = faiss_index.search(np.array([embedding]), k=1)
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
        cursor.execute("SELECT wine_slug, name FROM wines WHERE id =%s;", (wine_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if not result:
            raise Exception(f"Индекс {wine_id} есть в faiss, но записи с таким id нет в таблице sql")

        wine_slug, wine_name = result
        wine_url=f"https://vino-svoe.ru/wines/{wine_slug}"
        print(f" вино успешно извлечено:'{wine_name} slug'{wine_slug}'", flush =True)
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
        "wine_name": wine_name,
        "url": wine_url,
        "parsed_data":{
            "description":description
        }
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)