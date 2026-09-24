import base64
import io
import os
import sys
import json
import asyncio
import traceback

from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO

import requests
from bs4 import BeautifulSoup

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

import torch
import torch.nn.functional as F

from PIL import Image

import faiss
import numpy as np
import cv2
import onnxruntime as ort

from transformers import AutoImageProcessor, AutoModel


# ============================================================
# НАСТРОЙКИ
# ============================================================

# DINOv2 with registers, small (hidden_size = 384)
DINOV2_MODEL_NAME = "facebook/dinov2-with-registers-small"

# Способ получения вектора изображения:
#   "cls"  - pooler_output (CLS-токен после LayerNorm)  <- по умолчанию
#   "mean" - среднее по patch-токенам (без CLS и registers)
# ВАЖНО: должен совпадать с тем, как строился индекс wines_base.index!
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "cls")

# Твои базы:
#   wines_base.index    - FAISS IndexFlatIP, d=384, 1958 векторов, L2-нормализованы
#   wines_mapping.json  - {"0": "slug", "1": "slug", ...}  (id в FAISS -> slug)
INDEX_FILE_PATH = "wines_base.index"
MAPPING_FILE_PATH = "wines_mapping.json"

# YOLO
YOLO_MODEL_PATH = "label_detector.onnx"
YOLO_INPUT_WIDTH = 640
YOLO_INPUT_HEIGHT = 640
YOLO_CONFIDENCE_THRESHOLD = 0.25

# --------------------------------------------------------------
# Пороги отсева "это не бутылка вина" (по cosine similarity к
# ближайшему вектору в FAISS-базе).
#
# Логика: DINOv2-эмбеддинг случайной картинки (кот, стол, человек,
# другой напиток и т.д.) в среднем даёт низкую max-similarity к
# базе вин. Если YOLO не смог найти этикетку и в DINOv2 идёт всё
# фото целиком — эмбеддинг менее показательный, поэтому порог для
# этого случая строже.
#
# СТАРТОВЫЕ значения, их обязательно нужно откалибровать на своих
# реальных фото (позитивы: фото бутылок вина; негативы: что угодно
# другое) — см. комментарий ниже в is_wine_photo().
# --------------------------------------------------------------
SIMILARITY_THRESHOLD_WITH_BOX = float(
    os.getenv("SIMILARITY_THRESHOLD_WITH_BOX", "0.45")
)
SIMILARITY_THRESHOLD_NO_BOX = float(
    os.getenv("SIMILARITY_THRESHOLD_NO_BOX", "0.55")
)

# Сайт
SITE_BASE_URL = "https://vino-svoe.ru"
IMAGE_BASE_URL = "https://api.vino-svoe.ru"

# DEBUG
DEBUG_SAVE_CROPS = True
DEBUG_CROPS_DIR = "debug_crops"
DEBUG_LOG_CHUNK_SIZE = 4000


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[Прогноз] использование устройства: {device}", flush=True)

if device.type == "cuda":
    print(f"[CUDA] GPU: {torch.cuda.get_device_name(0)}", flush=True)


# ============================================================
# YOLO
# ============================================================

try:
    session = ort.InferenceSession(
        YOLO_MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    print(f"[YOLO] Модель загружена: {YOLO_MODEL_PATH}", flush=True)
    print(f"[YOLO] Input name: {input_name}", flush=True)

except Exception as e:
    print(f"[YOLO] Ошибка загрузки модели: {e}", flush=True)
    sys.exit(1)


# ============================================================
# DINOv2 with registers
# ============================================================

try:
    print(f"[DINOv2] Загрузка модели: {DINOV2_MODEL_NAME}", flush=True)

    processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_NAME)

    model = AutoModel.from_pretrained(DINOV2_MODEL_NAME)
    model.eval()
    model.to(device)

    embedding_dimension = model.config.hidden_size
    num_registers = getattr(model.config, "num_register_tokens", 4)

    print("[DINOv2] Модель успешно загружена", flush=True)
    print(f"[DINOv2] Hidden size: {embedding_dimension}", flush=True)
    print(f"[DINOv2] Register tokens: {num_registers}", flush=True)
    print(f"[DINOv2] Embedding mode: {EMBEDDING_MODE}", flush=True)

except Exception as e:
    print(f"[DINOv2] Ошибка загрузки модели: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)


# ============================================================
# FAISS + MAPPING
# ============================================================

index = None
id_to_slug = {}


@asynccontextmanager
async def lifespan(app: FastAPI):

    global index, id_to_slug

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    if not os.path.exists(INDEX_FILE_PATH):
        raise FileNotFoundError(f"Файл индекса {INDEX_FILE_PATH} не найден!")

    index = faiss.read_index(INDEX_FILE_PATH)

    print("[FAISS] База успешно загружена.", flush=True)
    print(f"[FAISS] Всего векторов: {index.ntotal}", flush=True)
    print(f"[FAISS] Размерность индекса: {index.d}", flush=True)

    if index.d != embedding_dimension:
        raise RuntimeError(
            f"Несовместимая размерность FAISS!\n"
            f"FAISS: {index.d}\n"
            f"DINOv2: {embedding_dimension}"
        )

    # --------------------------------------------------------
    # Mapping: id в FAISS -> slug вина
    # --------------------------------------------------------

    if not os.path.exists(MAPPING_FILE_PATH):
        raise FileNotFoundError(f"Файл маппинга {MAPPING_FILE_PATH} не найден!")

    with open(MAPPING_FILE_PATH, "r", encoding="utf-8") as f:
        raw_mapping = json.load(f)

    # ключи в JSON всегда строки -> приводим к int
    id_to_slug = {int(k): v for k, v in raw_mapping.items()}

    print(f"[MAPPING] Загружено записей: {len(id_to_slug)}", flush=True)

    if len(id_to_slug) != index.ntotal:
        print(
            f"[MAPPING] ВНИМАНИЕ: записей в маппинге ({len(id_to_slug)}) "
            f"не равно числу векторов в FAISS ({index.ntotal})",
            flush=True
        )

    yield

    print("[SERVER] Сервер останавливается.", flush=True)


app = FastAPI(lifespan=lifespan)

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


print("Ожидание запросов\n", flush=True)


# ============================================================
# DEBUG CROP
# ============================================================

def save_crop_for_debugging(image: Image.Image, label: str = "crop") -> None:
    """
    Локально: сохраняет файл в debug_crops/
    На Render: дополнительно выводит base64 в лог.
    """

    if not DEBUG_SAVE_CROPS:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{label}_{timestamp}.jpg"

    try:
        os.makedirs(DEBUG_CROPS_DIR, exist_ok=True)
        filepath = os.path.join(DEBUG_CROPS_DIR, filename)
        image.save(filepath, format="JPEG", quality=90)
        print(f"[DEBUG] Кроп сохранён на диск: {filepath}", flush=True)
    except Exception as e:
        print(f"[DEBUG] Не удалось сохранить кроп на диск: {e}", flush=True)

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


# ============================================================
# YOLO LETTERBOX
# ============================================================

def letterbox_preprocess(img_bgr, input_size):
    """
    Сохраняем пропорции изображения, затем добавляем padding.
    """

    h, w = img_bgr.shape[:2]

    scale = min(input_size[0] / h, input_size[1] / w)

    nh = int(h * scale)
    nw = int(w * scale)

    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    dh = (input_size[0] - nh) / 2
    dw = (input_size[1] - nw) / 2

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[None, :]

    return blob, scale, (dw, dh)


# ============================================================
# DINOv2 (with registers) EMBEDDING
# ============================================================

def get_dinov2_embedding(image: Image.Image) -> np.ndarray:
    """
    Embedding изображения через DINOv2 with registers (small).

    На выходе: numpy.ndarray shape = (384,), dtype = float32, L2-нормализован.

    Структура last_hidden_state:
        [CLS] + [register tokens x N] + [patch tokens]
    """

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model(**inputs)

    if EMBEDDING_MODE == "mean":
        # среднее только по patch-токенам (пропускаем CLS и registers)
        patch_tokens = outputs.last_hidden_state[0, 1 + num_registers:]
        embedding = patch_tokens.mean(dim=0)
    else:
        # CLS-токен после LayerNorm
        embedding = outputs.pooler_output[0]

    # L2-нормализация: inner product == cosine similarity (IndexFlatIP)
    embedding = F.normalize(embedding, p=2, dim=0)

    return embedding.detach().cpu().numpy().astype("float32")


# ============================================================
# ФИЛЬТР "ЭТО ВООБЩЕ ПОХОЖЕ НА ВИНО?"
# ============================================================

def is_wine_photo(similarity: float, box_detected: bool) -> bool:
    """
    Отсекает фото, которые непохожи ни на одно вино в базе.

    Идея: cosine similarity к ближайшему соседу в FAISS — это по сути
    "уверенность" системы. Для настоящих фото бутылок вина (даже
    незнакомых конкретных вин) similarity к чему-то в базе обычно
    заметно выше, чем для случайных посторонних фото (люди, еда,
    интерьер, другие товары и т.п.), потому что DINOv2-эмбеддинги
    визуально похожих объектов (форма бутылки, этикетка, стекло)
    группируются ближе друг к другу в пространстве эмбеддингов.

    Порог разный для двух случаев:
      - box_detected=True  -> YOLO нашёл область этикетки, эмбеддинг
        считается по кропу -> сигнал чище -> порог мягче.
      - box_detected=False -> использовалось всё фото целиком (могло
        быть что угодно в кадре) -> сигнал шумнее -> порог строже.

    КАЛИБРОВКА (обязательно сделать перед продом):
      1. Собрать ~150-300 позитивных фото (реальные бутылки вина,
         разные ракурсы/освещение/размытие) и ~150-300 негативных
         (люди, еда, интерьер, другие напитки, случайные объекты).
      2. Прогнать через run_ml_pipeline, залогировать similarity
         и box_detected для каждого.
      3. Построить ROC/PR-кривую отдельно для box_detected=True и
         box_detected=False, выбрать пороги, дающие нужный баланс
         precision/recall (для 90%+ точности отсева обычно нужно
         сознательно жертвовать частью recall на "плохих" фото
         реальных бутылок — т.е. иногда просить переснять).
      4. Подставить подобранные значения в переменные
         SIMILARITY_THRESHOLD_WITH_BOX / SIMILARITY_THRESHOLD_NO_BOX
         (или через env-переменные на Render).
    """

    threshold = (
        SIMILARITY_THRESHOLD_WITH_BOX
        if box_detected
        else SIMILARITY_THRESHOLD_NO_BOX
    )

    return similarity >= threshold


# ============================================================
# ML PIPELINE
# ============================================================

def run_ml_pipeline(image, orig_w, orig_h, input_width, input_height):

    # 1. PIL -> NumPy -> BGR (YOLO ожидает BGR)
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # 2. Letterbox
    input_tensor, scale, pad = letterbox_preprocess(
        img_bgr, (input_width, input_height)
    )

    # 3. YOLO inference
    outputs = session.run(None, {input_name: input_tensor})
    prediction = outputs[0]

    # 4. Приводим prediction к форме (8400, N)
    pred = prediction[0]

    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    # 5. Лучшая детекция
    scores = np.max(pred[:, 4:], axis=1)
    best_idx = np.argmax(scores)
    best_score = float(scores[best_idx])

    print(f"[YOLO] Best confidence: {best_score:.4f}", flush=True)

    # 6. По умолчанию — всё фото
    x_min, y_min, x_max, y_max = 0, 0, orig_w, orig_h
    box_detected = False

    if best_score > YOLO_CONFIDENCE_THRESHOLD:

        box = pred[best_idx, :4]

        xc_model = box[0] * input_width
        yc_model = box[1] * input_height
        w_model = box[2] * input_width
        h_model = box[3] * input_height

        # XYWH -> XYXY
        x1_model = xc_model - w_model / 2
        y1_model = yc_model - h_model / 2
        x2_model = xc_model + w_model / 2
        y2_model = yc_model + h_model / 2

        # убираем padding, возвращаемся к оригиналу
        dw, dh = pad

        x1_orig = (x1_model - dw) / scale
        y1_orig = (y1_model - dh) / scale
        x2_orig = (x2_model - dw) / scale
        y2_orig = (y2_model - dh) / scale

        x_min = max(0, int(np.clip(x1_orig, 0, orig_w)))
        y_min = max(0, int(np.clip(y1_orig, 0, orig_h)))
        x_max = min(orig_w, int(np.clip(x2_orig, 0, orig_w)))
        y_max = min(orig_h, int(np.clip(y2_orig, 0, orig_h)))

        box_detected = True

        print(
            f"[YOLO] Этикетка найдена: ({x_min},{y_min})-({x_max},{y_max})",
            flush=True
        )

    else:
        print(
            "[YOLO] Детекция не сработала (низкая уверенность) — "
            "используется полное фото",
            flush=True
        )

    # 7. Crop
    if box_detected and x_max > x_min and y_max > y_min:
        image = image.crop((x_min, y_min, x_max, y_max))
        print("[YOLO] Кроп применён", flush=True)

    # 8. Debug
    save_crop_for_debugging(image, label="dinov2_input")

    # 9. DINOv2 embedding
    embedding = get_dinov2_embedding(image)

    print(f"[DINOv2] Embedding shape: {embedding.shape}", flush=True)
    print(f"[DINOv2] Embedding norm: {np.linalg.norm(embedding):.4f}", flush=True)

    if embedding.shape[0] != index.d:
        raise RuntimeError(
            f"Размерность embedding ({embedding.shape[0]}) "
            f"не совпадает с FAISS ({index.d})."
        )

    # 10. FAISS (IndexFlatIP + нормализованные векторы = cosine similarity)
    query = embedding.reshape(1, -1)
    scores, indices = index.search(query, k=1)

    wine_id = int(indices[0][0])
    similarity = float(scores[0][0])

    print(f"[FAISS] wine_id: {wine_id}", flush=True)
    print(f"[FAISS] cosine similarity: {similarity:.4f}", flush=True)

    if wine_id == -1:
        raise Exception("FAISS не нашёл совпадений")

    return wine_id, similarity, box_detected


# ============================================================
# PARSE WINE SITE
# ============================================================

def _unique_texts(tags, fallback):
    if not tags:
        return [fallback]
    return list(dict.fromkeys(item.text.strip() for item in tags))


def find_by_slug(wine_slug):

    wine_url = f"{SITE_BASE_URL}/wines/{wine_slug}"

    headers = {
        "User-Agent":
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(wine_url, headers=headers, timeout=5)

        if response.status_code == 200:

            soup = BeautifulSoup(response.text, "html.parser")

            # DESCRIPTION
            tag = soup.find("p", class_="wine-page__description")
            description = tag.text.strip() if tag else "Нет описания"

            # NAME
            tag = soup.find("h1", class_="wine-main-title-block__title")
            wine_name = tag.text.strip() if tag else "Нет названия"

            # FACTORY
            tag = soup.find("a", class_="wine-main-title-block__manufacturer")
            factory = tag.text.strip() if tag else "Нет информации о заводе"

            # RATING
            tag = soup.find("span", class_="wine-main-title-block__rating-text")
            rate = tag.text.strip() if tag else "Нет рейтинга"

            # ATCC
            atcc_list = _unique_texts(
                soup.find_all("p", class_="wine-detail-info__detail-value"),
                "Нет информации"
            )

            # NUM
            num_list = _unique_texts(
                soup.find_all("p", class_="wine-hero-block__card-value"),
                "Нет информации"
            )

            # DISHES
            dishes_list = _unique_texts(
                soup.find_all("p", class_="wine-dish-item__name"),
                "Нет блюд"
            )

            # IMAGE
            wine_image = "Нет картинки"

            image_tag = soup.find("img", class_="wine-hero-block__bottle")

            if image_tag and image_tag.get("src"):

                image_src = image_tag["src"]

                if image_src.startswith("/"):
                    image_url = f"{IMAGE_BASE_URL}{image_src}"
                else:
                    image_url = image_src

                try:
                    image_result = requests.get(
                        image_url, headers=headers, timeout=5
                    )

                    if image_result.status_code == 200:
                        b64_encoded = base64.b64encode(
                            image_result.content
                        ).decode("utf-8")

                        wine_image = f"data:image/jpeg;base64,{b64_encoded}"

                except Exception as e:
                    print(
                        f"[SITE] Не удалось скачать картинку товара: {e}",
                        flush=True
                    )

            print(
                f"[SITE] Данные успешно получены. {wine_name}, "
                f"Описание: {description}, {factory}, {rate}, "
                f"{atcc_list}, {num_list}, {dishes_list}",
                flush=True
            )

        else:
            print(f"[SITE] Сайт вернул код {response.status_code}", flush=True)

            msg = "Ошибка подключения к сайту"
            description = wine_name = factory = rate = wine_image = msg
            atcc_list = num_list = dishes_list = [msg]

    except Exception as e:
        print(f"[SITE] Не удалось распарсить страницу: {e}", flush=True)
        traceback.print_exc()

        msg = "Не удалось загрузить информацию"
        description = wine_name = factory = rate = wine_image = msg
        atcc_list = num_list = dishes_list = [msg]

    return (
        wine_url, wine_slug, description, wine_name, factory,
        rate, atcc_list, num_list, dishes_list, wine_image
    )


# ============================================================
# FETCH WINE (FAISS id -> slug -> сайт)
# ============================================================

def fetch_wine_data(wine_id):

    wine_slug = id_to_slug.get(wine_id)

    if wine_slug is None:
        raise Exception(
            f"Индекс {wine_id} есть в FAISS, "
            f"но отсутствует в {MAPPING_FILE_PATH}"
        )

    return find_by_slug(wine_slug)


# ============================================================
# RESPONSE
# ============================================================

def parsed_info(
    wine_url, wine_slug, description, wine_name, factory,
    rate, atcc_list, num_list, dishes_list, wine_image
):

    def pick(lst, i):
        return lst[i] if len(lst) > i else "Нет информации"

    payload = {
        "status": "success",
        "url": wine_url,
        "parsed_data": {
            "name": wine_name,
            "description": description,
            "factory": factory,
            "rate": rate,
            "area": pick(atcc_list, 0),
            "sort": pick(atcc_list, 1),
            "type": pick(atcc_list, 2),
            "color": pick(atcc_list, 3),
            "temperature": pick(num_list, 0),
            "alcohol": pick(num_list, 1),
            "dishes": dishes_list,
            "wine_image": wine_image
        }
    }

    json_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    print("[END] Запрос успешно обработан, отправляем ответ.", flush=True)

    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={
            "Content-Length": str(len(json_bytes)),
            "Cache-Control": "no-transform"
        }
    )


# ============================================================
# /api/recognize
# ============================================================

@app.post("/api/recognize")
async def recognize_wine(data: ImageRequest):

    print("\n[START] Начало обработки", flush=True)

    # 1. BASE64 -> PIL
    try:
        pure_base64 = data.image_base64.split(",")[-1]
        image_data = base64.b64decode(pure_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        orig_w, orig_h = image.size

        print(f"[IMAGE] Фотку успешно декодировал. Размер: {image.size}", flush=True)

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось прочитать base64 строку: {str(e)}"
        )

    # 2. ML + маппинг + сайт
    try:
        print("[ML] Запуск YOLO + DINOv2-reg + FAISS", flush=True)

        wine_id, similarity, box_detected = await asyncio.to_thread(
            run_ml_pipeline,
            image, orig_w, orig_h, YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT
        )

        print(f"[FAISS] Выдал ID: {wine_id}", flush=True)
        print(f"[FAISS] Cosine similarity: {similarity:.4f}", flush=True)

        # ---- ФИЛЬТР: похоже ли вообще на бутылку вина? ----
        if not is_wine_photo(similarity, box_detected):
            print(
                f"[FILTER] Фото отклонено как не-вино: "
                f"similarity={similarity:.4f}, box_detected={box_detected}",
                flush=True
            )
            # ВАЖНО: подгоните под то, что уже ждёт фронтенд как "не нашлось".
            # Если у фронта уже есть обработка 404 -> достаточно так.
            # Если фронт ждёт именно JSON-структуру как в parsed_info() со
            # status != "success" -- поменяйте на такой Response вместо
            # HTTPException.
            raise HTTPException(
                status_code=404,
                detail="Вино на фото не распознано"
            )

        result = await asyncio.to_thread(fetch_wine_data, wine_id)

        print(f"[MAPPING] Данные собраны. Slug: {result[1]}", flush=True)

    except HTTPException:
        # пробрасываем как есть (в т.ч. 404 из фильтра выше),
        # чтобы не превратилось в 500 в блоке ниже
        raise
    except Exception as e:
        print(f"[ERROR] Сбой в пайплайне: {str(e)}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return parsed_info(*result)


# ============================================================
# /api/memory
# ============================================================

@app.post("/api/memory")
async def memory_wine(data: WineSlug):

    try:
        result = await asyncio.to_thread(find_by_slug, data.memory_slug)

        print(f"[MEMORY] Данные собраны. Slug: {result[1]}", flush=True)

    except Exception as e:
        print(f"[ERROR] Сбой в пайплайне: {str(e)}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    return parsed_info(*result)


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
