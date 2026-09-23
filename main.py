import base64
import io
import os
import sys
import asyncio
import traceback

from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO

import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import torch.nn.functional as F

from PIL import Image

import faiss
import psycopg2
import numpy as np
import cv2
import onnxruntime as ort
import json

from fastapi.responses import Response

from transformers import (
    AutoImageProcessor,
    AutoModel
)


# ============================================================
# НАСТРОЙКИ
# ============================================================

# DINOv3
DINOV3_MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"

# Новый FAISS-индекс.
# ВАЖНО:
# Этот файл должен быть создан именно из DINOv3-векторов.
INDEX_FILE_PATH = "wines_base_dinov3.index"

# YOLO
YOLO_MODEL_PATH = "label_detector.onnx"

# Размер входа YOLO
YOLO_INPUT_WIDTH = 640
YOLO_INPUT_HEIGHT = 640

# Порог уверенности YOLO
YOLO_CONFIDENCE_THRESHOLD = 0.25

# ============================================================
# DEBUG
# ============================================================

DEBUG_SAVE_CROPS = True
DEBUG_CROPS_DIR = "debug_crops"
DEBUG_LOG_CHUNK_SIZE = 4000


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(
    f"[Прогноз] использование устройства: {device}",
    flush=True
)

if device.type == "cuda":
    print(
        f"[CUDA] GPU: {torch.cuda.get_device_name(0)}",
        flush=True
    )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI()


# ============================================================
# YOLO
# ============================================================

try:

    session = ort.InferenceSession(
        YOLO_MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    print(
        f"[YOLO] Модель загружена: {YOLO_MODEL_PATH}",
        flush=True
    )

    print(
        f"[YOLO] Input name: {input_name}",
        flush=True
    )

except Exception as e:

    print(
        f"[YOLO] Ошибка загрузки модели: {e}",
        flush=True
    )

    sys.exit(1)


# ============================================================
# DINOv3
# ============================================================

try:

    print(
        f"[DINOv3] Загрузка модели: {DINOV3_MODEL_NAME}",
        flush=True
    )

    processor = AutoImageProcessor.from_pretrained(
        DINOV3_MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        DINOV3_MODEL_NAME
    )

    model.eval()
    model.to(device)

    # Для DINOv3 ViT-S/16 ожидается hidden_size = 384.
    embedding_dimension = model.config.hidden_size

    print(
        "[DINOv3] Модель успешно загружена",
        flush=True
    )

    print(
        f"[DINOv3] Hidden size: {embedding_dimension}",
        flush=True
    )

    print(
        f"[DINOv3] Device: {device}",
        flush=True
    )

except Exception as e:

    print(
        f"[DINOv3] Ошибка загрузки модели: {e}",
        flush=True
    )

    traceback.print_exc()

    sys.exit(1)


# ============================================================
# FAISS
# ============================================================

index = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global index

    # --------------------------------------------------------
    # Загружаем FAISS
    # --------------------------------------------------------

    if not os.path.exists(INDEX_FILE_PATH):

        raise FileNotFoundError(
            f"Файл индекса {INDEX_FILE_PATH} не найден! "
            f"Создай новый FAISS-индекс на основе DINOv3."
        )

    index = faiss.read_index(
        INDEX_FILE_PATH
    )

    print(
        f"[FAISS] База успешно загружена.",
        flush=True
    )

    print(
        f"[FAISS] Всего векторов: {index.ntotal}",
        flush=True
    )

    print(
        f"[FAISS] Размерность индекса: {index.d}",
        flush=True
    )

    print(
        f"[DINOv3] Размерность embedding: {embedding_dimension}",
        flush=True
    )

    # --------------------------------------------------------
    # Проверяем совместимость FAISS и DINOv3
    # --------------------------------------------------------

    if index.d != embedding_dimension:

        raise RuntimeError(
            f"Несовместимая размерность FAISS!\n"
            f"FAISS: {index.d}\n"
            f"DINOv3: {embedding_dimension}\n\n"
            f"Нужно пересоздать FAISS-индекс "
            f"из DINOv3-векторов."
        )

    # --------------------------------------------------------
    # Инициализация PostgreSQL
    # --------------------------------------------------------

    try:

        conn = get_db_connection()

        cursor = conn.cursor()

        with open(
            "init.sql",
            "r",
            encoding="utf-8"
        ) as f:

            sql_script = f.read()

        cursor.execute(sql_script)

        conn.commit()

        cursor.close()
        conn.close()

        print(
            "[PostgreSQL] Таблицы SQL успешно инициализированы!",
            flush=True
        )

    except Exception as e:

        print(
            f"[PostgreSQL] Ошибка инициализации БД: {e}",
            flush=True
        )

    yield

    print(
        "[SERVER] Сервер останавливается.",
        flush=True
    )


app = FastAPI(
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class ImageRequest(BaseModel):

    image_base64: str


class WineSlug(BaseModel):

    memory_slug: str


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():

    return psycopg2.connect(
        host=os.getenv(
            "DB_HOST",
            "dpg-dagfdfh42hec73c328bg-a"
        ),

        database=os.getenv(
            "DB_NAME",
            "wine_db_p4pv_0tft"
        ),

        user=os.getenv(
            "DB_USER",
            "wine_user"
        ),

        password=os.getenv(
            "DB_PASSWORD"
        ),

        port=os.getenv(
            "DB_PORT",
            "5432"
        )
    )


print(
    "Ожидание запросов\n",
    flush=True
)


# ============================================================
# DEBUG CROP
# ============================================================

def save_crop_for_debugging(
    image: Image.Image,
    label: str = "crop"
) -> None:

    """
    Отладочное сохранение изображения.

    Локально:
        сохраняет файл в debug_crops/

    На Render:
        дополнительно выводит base64 в лог.
    """

    if not DEBUG_SAVE_CROPS:

        return

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = f"{label}_{timestamp}.jpg"

    # --------------------------------------------------------
    # Сохранение на диск
    # --------------------------------------------------------

    try:

        os.makedirs(
            DEBUG_CROPS_DIR,
            exist_ok=True
        )

        filepath = os.path.join(
            DEBUG_CROPS_DIR,
            filename
        )

        image.save(
            filepath,
            format="JPEG",
            quality=90
        )

        print(
            f"[DEBUG] Кроп сохранён на диск: {filepath}",
            flush=True
        )

    except Exception as e:

        print(
            f"[DEBUG] Не удалось сохранить кроп на диск: {e}",
            flush=True
        )

    # --------------------------------------------------------
    # Base64 в консоль
    # --------------------------------------------------------

    try:

        buffer = BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=90
        )

        b64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        print(
            f"[DEBUG_CROP_BASE64_START] "
            f"{filename} size={len(b64)}",
            flush=True
        )

        for i in range(
            0,
            len(b64),
            DEBUG_LOG_CHUNK_SIZE
        ):

            print(
                b64[
                    i:i + DEBUG_LOG_CHUNK_SIZE
                ],
                flush=True
            )

        print(
            f"[DEBUG_CROP_BASE64_END] {filename}",
            flush=True
        )

    except Exception as e:

        print(
            f"[DEBUG] Не удалось закодировать "
            f"кроп в base64: {e}",
            flush=True
        )


# ============================================================
# YOLO LETTERBOX
# ============================================================

def letterbox_preprocess(
    img_bgr,
    input_size
):

    """
    Точная подготовка изображения для YOLO.

    Сохраняем пропорции изображения,
    затем добавляем padding.
    """

    h, w = img_bgr.shape[:2]

    scale = min(
        input_size[0] / h,
        input_size[1] / w
    )

    nh = int(h * scale)
    nw = int(w * scale)

    resized = cv2.resize(
        img_bgr,
        (nw, nh),
        interpolation=cv2.INTER_LINEAR
    )

    dh = (
        input_size[0] - nh
    ) / 2

    dw = (
        input_size[1] - nw
    ) / 2

    top = int(
        round(dh - 0.1)
    )

    bottom = int(
        round(dh + 0.1)
    )

    left = int(
        round(dw - 0.1)
    )

    right = int(
        round(dw + 0.1)
    )

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    blob = (
        padded.astype(
            np.float32
        ) / 255.0
    )

    blob = blob.transpose(
        2,
        0,
        1
    )[None, :]

    return (
        blob,
        scale,
        (dw, dh)
    )


# ============================================================
# DINOv3 EMBEDDING
# ============================================================

def get_dinov3_embedding(
    image: Image.Image
) -> np.ndarray:

    """
    Получение embedding изображения через DINOv3.

    На выходе:
        numpy.ndarray shape = (384,)
        dtype = float32

    Вектор L2-нормализуется.
    """

    # --------------------------------------------------------
    # DINOv3 processor самостоятельно выполняет:
    #
    # resize
    # normalization
    # conversion to tensor
    # --------------------------------------------------------

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    # Переносим tensor на CPU/GPU
    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # Инференс
    # --------------------------------------------------------

    with torch.inference_mode():

        outputs = model(
            **inputs
        )

    # --------------------------------------------------------
    # Получаем embedding всего изображения
    # --------------------------------------------------------

    embedding = outputs.pooler_output[0]

    # --------------------------------------------------------
    # L2 normalization
    #
    # Это необходимо для cosine similarity
    # через FAISS IndexFlatIP.
    # --------------------------------------------------------

    embedding = F.normalize(
        embedding,
        p=2,
        dim=0
    )

    # --------------------------------------------------------
    # Tensor -> NumPy
    # --------------------------------------------------------

    embedding = (
        embedding
        .detach()
        .cpu()
        .numpy()
        .astype("float32")
    )

    return embedding


# ============================================================
# ML PIPELINE
# ============================================================

def run_ml_pipeline(
    image,
    orig_w,
    orig_h,
    input_width,
    input_height
):

    # ========================================================
    # 1. PIL -> NumPy
    # ========================================================

    img_np = np.array(
        image
    )

    # ========================================================
    # 2. RGB -> BGR
    #
    # YOLO ожидает BGR.
    # ========================================================

    img_bgr = cv2.cvtColor(
        img_np,
        cv2.COLOR_RGB2BGR
    )

    # ========================================================
    # 3. Letterbox для YOLO
    # ========================================================

    input_tensor, scale, pad = (
        letterbox_preprocess(
            img_bgr,
            (
                input_width,
                input_height
            )
        )
    )

    # ========================================================
    # 4. YOLO inference
    # ========================================================

    outputs = session.run(
        None,
        {
            input_name: input_tensor
        }
    )

    prediction = outputs[0]

    # ========================================================
    # 5. Приводим prediction к форме (8400, N)
    # ========================================================

    pred = prediction[0]

    if pred.shape[0] < pred.shape[1]:

        pred = pred.T

    # ========================================================
    # 6. Получаем confidence
    # ========================================================

    scores = np.max(
        pred[:, 4:],
        axis=1
    )

    best_idx = np.argmax(
        scores
    )

    best_score = float(
        scores[best_idx]
    )

    print(
        f"[YOLO] Best confidence: "
        f"{best_score:.4f}",
        flush=True
    )

    # ========================================================
    # 7. Значения bounding box по умолчанию
    #
    # Если YOLO не нашёл этикетку,
    # используем всю фотографию.
    # ========================================================

    x_min = 0
    y_min = 0
    x_max = orig_w
    y_max = orig_h

    box_detected = False

    # ========================================================
    # 8. Проверяем confidence
    # ========================================================

    if best_score > YOLO_CONFIDENCE_THRESHOLD:

        box = pred[
            best_idx,
            :4
        ]

        xc_norm = box[0]
        yc_norm = box[1]
        w_norm = box[2]
        h_norm = box[3]

        # ----------------------------------------------------
        # Координаты в пространстве YOLO
        # ----------------------------------------------------

        xc_model = (
            xc_norm *
            input_width
        )

        yc_model = (
            yc_norm *
            input_height
        )

        w_model = (
            w_norm *
            input_width
        )

        h_model = (
            h_norm *
            input_height
        )

        # ----------------------------------------------------
        # XYWH -> XYXY
        # ----------------------------------------------------

        x1_model = (
            xc_model -
            w_model / 2
        )

        y1_model = (
            yc_model -
            h_model / 2
        )

        x2_model = (
            xc_model +
            w_model / 2
        )

        y2_model = (
            yc_model +
            h_model / 2
        )

        # ----------------------------------------------------
        # Убираем padding
        # и возвращаемся к оригинальному изображению
        # ----------------------------------------------------

        dw, dh = pad

        x1_orig = (
            x1_model - dw
        ) / scale

        y1_orig = (
            y1_model - dh
        ) / scale

        x2_orig = (
            x2_model - dw
        ) / scale

        y2_orig = (
            y2_model - dh
        ) / scale

        # ----------------------------------------------------
        # Ограничиваем координаты
        # ----------------------------------------------------

        x_min = max(
            0,
            int(
                np.clip(
                    x1_orig,
                    0,
                    orig_w
                )
            )
        )

        y_min = max(
            0,
            int(
                np.clip(
                    y1_orig,
                    0,
                    orig_h
                )
            )
        )

        x_max = min(
            orig_w,
            int(
                np.clip(
                    x2_orig,
                    0,
                    orig_w
                )
            )
        )

        y_max = min(
            orig_h,
            int(
                np.clip(
                    y2_orig,
                    0,
                    orig_h
                )
            )
        )

        box_detected = True

        print(
            "[YOLO] Этикетка найдена: "
            f"({x_min},{y_min})-"
            f"({x_max},{y_max})",
            flush=True
        )

    else:

        print(
            "[YOLO] Детекция не сработала "
            "(низкая уверенность) — "
            "используется полное фото",
            flush=True
        )

    # ========================================================
    # 9. Делаем crop
    # ========================================================

    if (
        box_detected
        and x_max > x_min
        and y_max > y_min
    ):

        image = image.crop(
            (
                x_min,
                y_min,
                x_max,
                y_max
            )
        )

        print(
            "[YOLO] Кроп применён",
            flush=True
        )

    # ========================================================
    # 10. Debug
    # ========================================================

    save_crop_for_debugging(
        image,
        label="dinov3_input"
    )

    # ========================================================
    # 11. DINOv3 embedding
    # ========================================================

    embedding = get_dinov3_embedding(
        image
    )

    print(
        f"[DINOv3] Embedding shape: "
        f"{embedding.shape}",
        flush=True
    )

    print(
        f"[DINOv3] Embedding norm: "
        f"{np.linalg.norm(embedding):.4f}",
        flush=True
    )

    # ========================================================
    # 12. Проверяем размерность
    # ========================================================

    if embedding.shape[0] != index.d:

        raise RuntimeError(
            f"Размерность embedding "
            f"({embedding.shape[0]}) "
            f"не совпадает с FAISS "
            f"({index.d}). "
            f"Индекс необходимо пересоздать."
        )

    # ========================================================
    # 13. FAISS
    #
    # Используем IndexFlatIP.
    #
    # Так как embedding L2-нормализован,
    # inner product = cosine similarity.
    # ========================================================

    query = embedding.reshape(
        1,
        -1
    )

    scores, indices = index.search(
        query,
        k=1
    )

    # ========================================================
    # 14. Результат
    # ========================================================

    wine_id = int(
        indices[0][0]
    )

    similarity = float(
        scores[0][0]
    )

    print(
        f"[FAISS] wine_id: {wine_id}",
        flush=True
    )

    print(
        f"[FAISS] cosine similarity: "
        f"{similarity:.4f}",
        flush=True
    )

    if wine_id == -1:

        raise Exception(
            "FAISS не нашёл совпадений"
        )

    return (
        wine_id,
        similarity
    )


# ============================================================
# PARSE WINE SITE
# ============================================================

def find_by_slug(
    wine_slug
):

    wine_url = (
        f"https://vino-svoe.ru/"
        f"wines/{wine_slug}"
    )

    headers = {
        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36"
    }

    try:

        response = requests.get(
            wine_url,
            headers=headers,
            timeout=5
        )

        if response.status_code == 200:

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            # =================================================
            # DESCRIPTION
            # =================================================

            description_tag = soup.find(
                "p",
                class_="wine-page__description"
            )

            description = (
                description_tag.text.strip()
                if description_tag
                else "Нет описания"
            )

            # =================================================
            # NAME
            # =================================================

            wine_name_tag = soup.find(
                "h1",
                class_="wine-main-title-block__title"
            )

            wine_name = (
                wine_name_tag.text.strip()
                if wine_name_tag
                else "Нет названия"
            )

            # =================================================
            # FACTORY
            # =================================================

            factory_tag = soup.find(
                "a",
                class_="wine-main-title-block__manufacturer"
            )

            factory = (
                factory_tag.text.strip()
                if factory_tag
                else "Нет информации о заводе"
            )

            # =================================================
            # RATING
            # =================================================

            rate_tag = soup.find(
                "span",
                class_="wine-main-title-block__rating-text"
            )

            rate = (
                rate_tag.text.strip()
                if rate_tag
                else "Нет рейтинга"
            )

            # =================================================
            # ATCC
            # =================================================

            atcc_tag = soup.find_all(
                "p",
                class_="wine-detail-info__detail-value"
            )

            if atcc_tag:

                atcc_list_clean = [
                    item.text.strip()
                    for item in atcc_tag
                ]

                atcc_list = list(
                    dict.fromkeys(
                        atcc_list_clean
                    )
                )

            else:

                atcc_list = [
                    "Нет информации"
                ]

            # =================================================
            # NUM
            # =================================================

            num_tag = soup.find_all(
                "p",
                class_="wine-hero-block__card-value"
            )

            if num_tag:

                num_list_clean = [
                    item.text.strip()
                    for item in num_tag
                ]

                num_list = list(
                    dict.fromkeys(
                        num_list_clean
                    )
                )

            else:

                num_list = [
                    "Нет информации"
                ]

            # =================================================
            # DISHES
            # =================================================

            dishes_tag = soup.find_all(
                "p",
                class_="wine-dish-item__name"
            )

            if dishes_tag:

                dishes_list_clean = [
                    item.text.strip()
                    for item in dishes_tag
                ]

                dishes_list = list(
                    dict.fromkeys(
                        dishes_list_clean
                    )
                )

            else:

                dishes_list = [
                    "Нет блюд"
                ]

            # =================================================
            # IMAGE
            # =================================================

            wine_image = "Нет картинки"

            image_tag = soup.find(
                "img",
                class_="wine-hero-block__bottle"
            )

            if (
                image_tag
                and image_tag.get("src")
            ):

                image_src = image_tag["src"]

                if image_src.startswith("/"):

                    image_url = (
                        f"https://api.vino-svoe"
                        f"{image_src}"
                    )

                else:

                    image_url = image_src

                try:

                    image_result = requests.get(
                        image_url,
                        headers=headers,
                        timeout=5
                    )

                    if image_result.status_code == 200:

                        b64_encoded = (
                            base64.b64encode(
                                image_result.content
                            )
                            .decode("utf-8")
                        )

                        wine_image = (
                            "data:image/jpeg;base64,"
                            f"{b64_encoded}"
                        )

                except Exception as e:

                    print(
                        "[SITE] Не удалось скачать "
                        f"картинку товара: {e}",
                        flush=True
                    )

            # =================================================
            # LOG
            # =================================================

            print(
                f"[SITE] Данные успешно получены. "
                f"{wine_name}, "
                f"Описание: {description}, "
                f"{factory}, "
                f"{rate}, "
                f"{atcc_list}, "
                f"{num_list}, "
                f"{dishes_list}",
                flush=True
            )

        else:

            print(
                f"[SITE] Сайт вернул код "
                f"{response.status_code}",
                flush=True
            )

            (
                description,
                wine_name,
                factory,
                rate,
                atcc_list,
                num_list,
                dishes_list,
                wine_image
            ) = (
                ["Ошибка подключения к сайту"] * 8
            )

    except Exception as e:

        print(
            f"[SITE] Не удалось распарсить страницу: "
            f"{e}",
            flush=True
        )

        traceback.print_exc()

        (
            description,
            wine_name,
            factory,
            rate,
            atcc_list,
            num_list,
            dishes_list,
            wine_image
        ) = (
            ["Не удалось загрузить информацию"] * 8
        )

    return (
        wine_url,
        wine_slug,
        description,
        wine_name,
        factory,
        rate,
        atcc_list,
        num_list,
        dishes_list,
        wine_image
    )


# ============================================================
# FETCH WINE FROM DATABASE
# ============================================================

def fetch_wine_data(
    wine_id
):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT wine_slug
        FROM wines
        WHERE id = %s;
        """,
        (wine_id,)
    )

    result = cursor.fetchone()

    cursor.close()

    conn.close()

    if not result:

        raise Exception(
            f"Индекс {wine_id} есть в FAISS, "
            f"но записи с таким id нет "
            f"в таблице SQL"
        )

    wine_slug = result[0]

    (
        wine_url,
        wine_slug,
        description,
        wine_name,
        factory,
        rate,
        atcc_list,
        num_list,
        dishes_list,
        wine_image
    ) = find_by_slug(
        wine_slug
    )

    return (
        wine_url,
        wine_slug,
        description,
        wine_name,
        factory,
        rate,
        atcc_list,
        num_list,
        dishes_list,
        wine_image
    )


# ============================================================
# RESPONSE
# ============================================================

def parsed_info(
    wine_url,
    wine_slug,
    description,
    wine_name,
    factory,
    rate,
    atcc_list,
    num_list,
    dishes_list,
    wine_image
):

    # --------------------------------------------------------
    # Защита от слишком коротких списков
    # --------------------------------------------------------

    area = (
        atcc_list[0]
        if len(atcc_list) > 0
        else "Нет информации"
    )

    wine_sort = (
        atcc_list[1]
        if len(atcc_list) > 1
        else "Нет информации"
    )

    wine_type = (
        atcc_list[2]
        if len(atcc_list) > 2
        else "Нет информации"
    )

    color = (
        atcc_list[3]
        if len(atcc_list) > 3
        else "Нет информации"
    )

    temperature = (
        num_list[0]
        if len(num_list) > 0
        else "Нет информации"
    )

    alcohol = (
        num_list[1]
        if len(num_list) > 1
        else "Нет информации"
    )

    # --------------------------------------------------------
    # Payload
    # --------------------------------------------------------

    payload = {

        "status": "success",

        "url": wine_url,

        "parsed_data": {

            "name": wine_name,

            "description": description,

            "factory": factory,

            "rate": rate,

            "area": area,

            "sort": wine_sort,

            "type": wine_type,

            "color": color,

            "temperature": temperature,

            "alcohol": alcohol,

            "dishes": dishes_list,

            "wine_image": wine_image

        }
    }

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_str = json.dumps(
        payload,
        ensure_ascii=False
    )

    json_bytes = json_str.encode(
        "utf-8"
    )

    print(
        "[END] Запрос успешно обработан, "
        "отправляем ответ.",
        flush=True
    )

    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={
            "Content-Length":
                str(len(json_bytes)),

            "Cache-Control":
                "no-transform"
        }
    )


# ============================================================
# /api/recognize
# ============================================================

@app.post("/api/recognize")
async def recognize_wine(
    data: ImageRequest
):

    print(
        "\n[START] Начало обработки",
        flush=True
    )

    # ========================================================
    # 1. BASE64 -> PIL
    # ========================================================

    try:

        pure_base64 = (
            data.image_base64
            .split(",")[-1]
        )

        image_data = base64.b64decode(
            pure_base64
        )

        image = Image.open(
            io.BytesIO(image_data)
        ).convert("RGB")

        orig_w, orig_h = image.size

        print(
            f"[IMAGE] Фотку успешно "
            f"декодировал. Размер: "
            f"{image.size}",
            flush=True
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                "Не удалось прочитать "
                f"base64 строку: {str(e)}"
            )
        )

    # ========================================================
    # 2. ML
    # ========================================================

    try:

        print(
            "[ML] Запуск YOLO + DINOv3 + FAISS",
            flush=True
        )

        wine_id, similarity = await asyncio.to_thread(
            run_ml_pipeline,
            image,
            orig_w,
            orig_h,
            YOLO_INPUT_WIDTH,
            YOLO_INPUT_HEIGHT
        )

        print(
            f"[FAISS] Выдал ID: {wine_id}",
            flush=True
        )

        print(
            f"[FAISS] Cosine similarity: "
            f"{similarity:.4f}",
            flush=True
        )

        # ====================================================
        # 3. DATABASE + WEBSITE
        # ====================================================

        (
            wine_url,
            wine_slug,
            description,
            wine_name,
            factory,
            rate,
            atcc_list,
            num_list,
            dishes_list,
            wine_image
        ) = await asyncio.to_thread(
            fetch_wine_data,
            wine_id
        )

        print(
            f"[DB] Данные собраны. "
            f"Slug: {wine_slug}",
            flush=True
        )

    except Exception as e:

        print(
            f"[ERROR] Сбой в пайплайне: "
            f"{str(e)}",
            flush=True
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    # ========================================================
    # 4. RESPONSE
    # ========================================================

    return parsed_info(
        wine_url,
        wine_slug,
        description,
        wine_name,
        factory,
        rate,
        atcc_list,
        num_list,
        dishes_list,
        wine_image
    )


# ============================================================
# /api/memory
# ============================================================

@app.post("/api/memory")
async def memory_wine(
    data: WineSlug
):

    try:

        (
            wine_url,
            wine_slug,
            description,
            wine_name,
            factory,
            rate,
            atcc_list,
            num_list,
            dishes_list,
            wine_image
        ) = await asyncio.to_thread(
            find_by_slug,
            data.memory_slug
        )

        print(
            f"[MEMORY] Данные собраны. "
            f"Slug: {wine_slug}",
            flush=True
        )

    except Exception as e:

        print(
            f"[ERROR] Сбой в пайплайне: "
            f"{str(e)}",
            flush=True
        )

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    return parsed_info(
        wine_url,
        wine_slug,
        description,
        wine_name,
        factory,
        rate,
        atcc_list,
        num_list,
        dishes_list,
        wine_image
    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
