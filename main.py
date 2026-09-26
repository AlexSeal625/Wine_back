import base64
import io
import os
import sys
import json
import asyncio
import traceback
import re
import xml.etree.ElementTree as ET

from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlparse

from contextlib import asynccontextmanager
from datetime import datetime, timezone
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

DINOV2_MODEL_NAME = "facebook/dinov2-with-registers-small"

EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "cls")

INDEX_FILE_PATH = "wines_base.index"
MAPPING_FILE_PATH = "wines_mapping.json"

# YOLO
YOLO_MODEL_PATH = "label_detector.onnx"
YOLO_INPUT_WIDTH = 640
YOLO_INPUT_HEIGHT = 640
YOLO_CONFIDENCE_THRESHOLD = 0.25

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
# DINOv2 with registers
# ============================================================

try:
    print(
        f"[DINOv2] Загрузка модели: {DINOV2_MODEL_NAME}",
        flush=True
    )

    processor = AutoImageProcessor.from_pretrained(
        DINOV2_MODEL_NAME
    )

    model = AutoModel.from_pretrained(
        DINOV2_MODEL_NAME
    )

    model.eval()
    model.to(device)

    embedding_dimension = model.config.hidden_size

    num_registers = getattr(
        model.config,
        "num_register_tokens",
        4
    )

    print(
        "[DINOv2] Модель успешно загружена",
        flush=True
    )

    print(
        f"[DINOv2] Hidden size: {embedding_dimension}",
        flush=True
    )

    print(
        f"[DINOv2] Register tokens: {num_registers}",
        flush=True
    )

    print(
        f"[DINOv2] Embedding mode: {EMBEDDING_MODE}",
        flush=True
    )

except Exception as e:
    print(
        f"[DINOv2] Ошибка загрузки модели: {e}",
        flush=True
    )

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
        raise FileNotFoundError(
            f"Файл индекса {INDEX_FILE_PATH} не найден!"
        )

    index = faiss.read_index(
        INDEX_FILE_PATH
    )

    print(
        "[FAISS] База успешно загружена.",
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
        raise FileNotFoundError(
            f"Файл маппинга {MAPPING_FILE_PATH} не найден!"
        )

    with open(
        MAPPING_FILE_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        raw_mapping = json.load(f)

    id_to_slug = {
        int(k): v
        for k, v in raw_mapping.items()
    }

    print(
        f"[MAPPING] Загружено записей: {len(id_to_slug)}",
        flush=True
    )

    if len(id_to_slug) != index.ntotal:
        print(
            f"[MAPPING] ВНИМАНИЕ: записей в маппинге "
            f"({len(id_to_slug)}) не равно числу векторов "
            f"в FAISS ({index.ntotal})",
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


class ImageTest(BaseModel):
    test_slug: str


# ============================================================
# FEED MODELS
# ============================================================

class FeedNewsItem(BaseModel):
    title: str
    description: str
    url: str
    image_url: str
    source: str
    published_at: str


class FeedWineItem(BaseModel):
    slug: str
    name: str
    image_url: str


class FeedPayload(BaseModel):
    news: list[FeedNewsItem]
    wines: list[FeedWineItem]


class FeedResponse(BaseModel):
    status: str
    feed: FeedPayload


# ============================================================
# FEED CACHE
# ============================================================

# 6 часов
FEED_CACHE_TTL = 6 * 60 * 60

# Количество элементов в выдаче
FEED_NEWS_LIMIT = 20
FEED_WINE_LIMIT = 30

# Таймауты внешних запросов
FEED_NEWS_TIMEOUT = 8
FEED_WINE_TIMEOUT = 5

# Источники новостей.
#
# Они используются только во время обновления кэша.
# Сам /feed при валидном кэше внешние сайты НЕ запрашивает.
FEED_NEWS_SOURCES = (
    "https://www.decanter.com/feed/",
    "https://www.wineenthusiast.com/feed/",
    "https://www.thedrinksbusiness.com/feed/",
)

# Уже сериализованный JSON.
# Благодаря этому при обычном запросе /feed не происходит
# повторной сборки Pydantic-моделей и json.
_feed_cache_bytes = None

# Момент последнего успешного обновления cache.
_feed_cache_timestamp = 0.0

# Защита от ситуации, когда одновременно приходит много
# запросов после истечения TTL.
_feed_cache_lock = asyncio.Lock()


# ============================================================
# FEED: CLEAN SLUG
# ============================================================

def clean_feed_wine_slug(slug: str) -> str:
    """
    Убирает технические хвосты из slug:

        vino_failed
        vino_faile
        vino_fail

    а также повторения:

        vino_failed_failed
        vino_faile_failed

    При этом обычная часть slug не изменяется.
    """

    value = str(slug).strip()

    while True:

        cleaned = re.sub(
            r"(?i)(?:_(?:failed|faile|fail))+$",
            "",
            value
        )

        if cleaned == value:
            return value

        value = cleaned


# ============================================================
# FEED: XML HELPERS
# ============================================================

def _xml_local_name(tag: str) -> str:
    """
    Убирает namespace из XML-тега.

    Например:

        {http://purl.org/rss/1.0/}item

    превращается в:

        item
    """

    return tag.rsplit("}", 1)[-1].lower()


def _xml_child_text(element, names) -> str:

    names = set(names)

    for child in element.iter():

        if child is element:
            continue

        if _xml_local_name(child.tag) in names:

            text = "".join(
                child.itertext()
            ).strip()

            if text:
                return text

    return ""


def _xml_link(element, feed_url: str) -> str:

    for child in element.iter():

        if child is element:
            continue

        if _xml_local_name(child.tag) != "link":
            continue

        href = (
            child.attrib.get("href")
            or ""
        ).strip()

        if href:

            rel = child.attrib.get("rel")

            if rel in (
                None,
                "",
                "alternate"
            ):
                return urljoin(
                    feed_url,
                    href
                )

        text = "".join(
            child.itertext()
        ).strip()

        if text:
            return urljoin(
                feed_url,
                text
            )

    return ""


def _xml_image_url(
    element,
    description: str,
    feed_url: str
) -> str:

    for child in element.iter():

        if child is element:
            continue

        local_name = _xml_local_name(
            child.tag
        )

        # Media RSS
        if local_name in {
            "content",
            "thumbnail"
        }:

            url = (
                child.attrib.get("url")
                or ""
            ).strip()

            if url:
                return urljoin(
                    feed_url,
                    url
                )

        # RSS enclosure
        if local_name == "enclosure":

            url = (
                child.attrib.get("url")
                or ""
            ).strip()

            mime = (
                child.attrib.get("type")
                or ""
            ).lower()

            if url and (
                mime.startswith("image/")
                or not mime
            ):
                return urljoin(
                    feed_url,
                    url
                )

    # Последний fallback:
    # ищем img внутри description/content.
    if description:

        soup = BeautifulSoup(
            description,
            "html.parser"
        )

        image_tag = soup.find("img")

        if image_tag and image_tag.get("src"):

            return urljoin(
                feed_url,
                image_tag["src"]
            )

    return ""


# ============================================================
# FEED: DATE
# ============================================================

def _parse_feed_datetime(
    value: str
) -> str:

    if not value:
        return ""

    # RFC 822 / RSS
    try:

        dt = parsedate_to_datetime(
            value
        )

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return (
            dt.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    except Exception:
        pass

    # ISO 8601 / Atom
    try:

        normalized = (
            value
            .strip()
            .replace(
                "Z",
                "+00:00"
            )
        )

        dt = datetime.fromisoformat(
            normalized
        )

        if dt.tzinfo is None:
            dt = dt.replace(
                tzinfo=timezone.utc
            )

        return (
            dt.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    except Exception:
        return ""


# ============================================================
# FEED: DESCRIPTION
# ============================================================

def _short_feed_description(
    value: str,
    limit: int = 280
) -> str:

    if not value:
        return ""

    text = BeautifulSoup(
        value,
        "html.parser"
    ).get_text(
        " ",
        strip=True
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if len(text) <= limit:
        return text

    return (
        text[:limit]
        .rsplit(" ", 1)[0]
        .rstrip(".,;:!?")
        + "..."
    )


# ============================================================
# FEED: NEWS SOURCE
# ============================================================

def _fetch_news_feed(
    feed_url: str
):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(compatible; WineAppFeed/1.0; "
            "+https://vino-svoe.ru)"
        )
    }

    response = requests.get(
        feed_url,
        headers=headers,
        timeout=FEED_NEWS_TIMEOUT
    )

    response.raise_for_status()

    root = ET.fromstring(
        response.content
    )

    feed_title = _xml_child_text(
        root,
        {"title"}
    )

    if not feed_title:
        feed_title = urlparse(
            feed_url
        ).netloc

    elements = [
        element
        for element in root.iter()
        if _xml_local_name(
            element.tag
        ) in {
            "item",
            "entry"
        }
    ]

    result = []

    for element in elements:

        title = _xml_child_text(
            element,
            {"title"}
        )

        url = _xml_link(
            element,
            feed_url
        )

        published_raw = _xml_child_text(
            element,
            {
                "pubdate",
                "published",
                "updated",
                "date"
            }
        )

        published_at = _parse_feed_datetime(
            published_raw
        )

        description_raw = _xml_child_text(
            element,
            {
                "description",
                "summary",
                "content",
                "encoded"
            }
        )

        if (
            not title
            or not url
            or not published_at
        ):
            continue

        result.append({
            "title": BeautifulSoup(
                title,
                "html.parser"
            ).get_text(
                " ",
                strip=True
            ),

            "description":
                _short_feed_description(
                    description_raw
                ),

            "url": url,

            "image_url":
                _xml_image_url(
                    element,
                    description_raw,
                    feed_url
                ),

            "source": BeautifulSoup(
                feed_title,
                "html.parser"
            ).get_text(
                " ",
                strip=True
            ),

            "published_at":
                published_at,
        })

    return result


# ============================================================
# FEED: LOAD NEWS
# ============================================================

def _load_feed_news():

    all_news = []

    for feed_url in FEED_NEWS_SOURCES:

        try:

            all_news.extend(
                _fetch_news_feed(
                    feed_url
                )
            )

        except Exception as e:

            print(
                f"[FEED][NEWS] Не удалось "
                f"загрузить {feed_url}: {e}",
                flush=True
            )

    # Дедупликация по URL.
    unique = {}

    for item in all_news:
        unique.setdefault(
            item["url"],
            item
        )

    news = list(
        unique.values()
    )

    # Самые новые сначала.
    news.sort(
        key=lambda item:
            item["published_at"],
        reverse=True
    )

    return news[
        :FEED_NEWS_LIMIT
    ]


# ============================================================
# FEED: WINE
# ============================================================

def _fetch_feed_wine(
    slug: str
):
    """
    Лёгкая версия find_by_slug().

    В отличие от find_by_slug():
    - не скачивает картинку;
    - не кодирует её в base64;
    - не собирает описание;
    - не собирает рейтинг;
    - не собирает блюда.

    Нужны только:
        slug
        name
        image_url
    """

    wine_url = (
        f"{SITE_BASE_URL}/wines/{slug}"
    )

    headers = {
        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36"
    }

    response = requests.get(
        wine_url,
        headers=headers,
        timeout=FEED_WINE_TIMEOUT
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    name_tag = soup.find(
        "h1",
        class_="wine-main-title-block__title"
    )

    image_tag = soup.find(
        "img",
        class_="wine-hero-block__bottle"
    )

    name = (
        name_tag.get_text(strip=True)
        if name_tag
        else ""
    )

    image_src = (
        image_tag.get("src")
        if image_tag
        else ""
    )

    if not name or not image_src:
        return None

    if image_src.startswith("/"):
        image_url = (
            f"{IMAGE_BASE_URL}"
            f"{image_src}"
        )
    else:
        image_url = image_src

    return {
        "slug": slug,
        "name": name,
        "image_url": image_url,
    }


# ============================================================
# FEED: LOAD WINES
# ============================================================

def _load_feed_wines():

    candidates = []
    seen = set()

    # Берём существующий FAISS -> slug mapping.
    # Никакой отдельной БД для ленты не создаём.
    for wine_id in sorted(id_to_slug):

        slug = clean_feed_wine_slug(
            id_to_slug[wine_id]
        )

        if not slug:
            continue

        if slug in seen:
            continue

        seen.add(slug)

        candidates.append(
            slug
        )

        if len(candidates) >= FEED_WINE_LIMIT:
            break

    if not candidates:
        raise RuntimeError(
            f"Не удалось сформировать "
            f"список вин из "
            f"{MAPPING_FILE_PATH}"
        )

    # Эти запросы выполняются только во время
    # обновления 6-часового cache.
    #
    # На обычный GET /feed они НЕ выполняются.
    with ThreadPoolExecutor(
        max_workers=8
    ) as executor:

        results = list(
            executor.map(
                _fetch_feed_wine,
                candidates
            )
        )

    wines = [
        item
        for item in results
        if item is not None
    ]

    if len(wines) < len(candidates):

        print(
            f"[FEED][WINES] Получено "
            f"{len(wines)} из "
            f"{len(candidates)} вин",
            flush=True
        )

    return wines[
        :FEED_WINE_LIMIT
    ]


# ============================================================
# FEED: BUILD PAYLOAD
# ============================================================

def _build_feed_response() -> bytes:

    news = _load_feed_news()

    wines = _load_feed_wines()

    if not news:
        raise RuntimeError(
            "Не удалось получить "
            "новости для винной ленты"
        )

    if not wines:
        raise RuntimeError(
            "Не удалось получить "
            "вина для винной ленты"
        )

    payload = FeedResponse(
        status="success",

        feed=FeedPayload(

            news=[
                FeedNewsItem(
                    **item
                )
                for item in news
            ],

            wines=[
                FeedWineItem(
                    **item
                )
                for item in wines
            ],
        )
    )

    # Сразу сериализуем в bytes.
    #
    # Поэтому при следующих запросах /feed
    # FastAPI не собирает JSON заново.
    return payload.json(
        ensure_ascii=False,
        separators=(",", ":")
    ).encode("utf-8")


# ============================================================
# FEED: CACHE
# ============================================================

def _feed_cache_is_valid() -> bool:

    return (
        _feed_cache_bytes is not None
        and (
            asyncio
            .get_running_loop()
            .time()
            - _feed_cache_timestamp
        ) < FEED_CACHE_TTL
    )


def _feed_response(
    stale: bool = False
):

    headers = {
        "Content-Length":
            str(
                len(
                    _feed_cache_bytes
                )
            ),

        "Cache-Control":
            f"public, "
            f"max-age={FEED_CACHE_TTL}, "
            f"no-transform",
    }

    if stale:
        headers[
            "X-Feed-Cache"
        ] = "stale"

    return Response(
        content=_feed_cache_bytes,
        media_type="application/json",
        headers=headers,
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

    if not DEBUG_SAVE_CROPS:
        return

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = (
        f"{label}_{timestamp}.jpg"
    )

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
            f"[DEBUG] Кроп сохранён на диск: "
            f"{filepath}",
            flush=True
        )

    except Exception as e:

        print(
            f"[DEBUG] Не удалось сохранить "
            f"кроп на диск: {e}",
            flush=True
        )

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
            f"[DEBUG_CROP_BASE64_END] "
            f"{filename}",
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
# DINOv2 EMBEDDING
# ============================================================

def get_dinov2_embedding(
    image: Image.Image
) -> np.ndarray:

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.inference_mode():

        outputs = model(
            **inputs
        )

    if EMBEDDING_MODE == "mean":

        patch_tokens = (
            outputs.last_hidden_state[
                0,
                1 + num_registers:
            ]
        )

        embedding = (
            patch_tokens.mean(
                dim=0
            )
        )

    else:

        embedding = (
            outputs.pooler_output[0]
        )

    embedding = F.normalize(
        embedding,
        p=2,
        dim=0
    )

    return (
        embedding
        .detach()
        .cpu()
        .numpy()
        .astype("float32")
    )


# ============================================================
# ФИЛЬТР "ЭТО ВООБЩЕ ПОХОЖЕ НА ВИНО?"
# ============================================================

def is_wine_photo(
    similarity: float,
    box_detected: bool
) -> bool:

    threshold = (
        SIMILARITY_THRESHOLD_WITH_BOX
        if box_detected
        else SIMILARITY_THRESHOLD_NO_BOX
    )

    return similarity >= threshold


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

    # 1. PIL -> NumPy -> BGR

    img_np = np.array(
        image
    )

    img_bgr = cv2.cvtColor(
        img_np,
        cv2.COLOR_RGB2BGR
    )

    # 2. Letterbox

    input_tensor, scale, pad = (
        letterbox_preprocess(
            img_bgr,
            (
                input_width,
                input_height
            )
        )
    )

    # 3. YOLO

    outputs = session.run(
        None,
        {
            input_name:
                input_tensor
        }
    )

    prediction = outputs[0]

    # 4. Приводим prediction
    # к форме (8400, N)

    pred = prediction[0]

    if (
        pred.shape[0]
        <
        pred.shape[1]
    ):
        pred = pred.T

    # 5. Лучшая детекция

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

    # 6. По умолчанию
    # используется всё фото

    x_min = 0
    y_min = 0
    x_max = orig_w
    y_max = orig_h

    box_detected = False

    if (
        best_score
        >
        YOLO_CONFIDENCE_THRESHOLD
    ):

        box = pred[
            best_idx,
            :4
        ]

        xc_model = (
            box[0]
            *
            input_width
        )

        yc_model = (
            box[1]
            *
            input_height
        )

        w_model = (
            box[2]
            *
            input_width
        )

        h_model = (
            box[3]
            *
            input_height
        )

        x1_model = (
            xc_model
            -
            w_model / 2
        )

        y1_model = (
            yc_model
            -
            h_model / 2
        )

        x2_model = (
            xc_model
            +
            w_model / 2
        )

        y2_model = (
            yc_model
            +
            h_model / 2
        )

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
            f"[YOLO] Этикетка найдена: "
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

    # 7. Crop

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

    # 8. Debug

    save_crop_for_debugging(
        image,
        label="dinov2_input"
    )

    # 9. DINOv2

    embedding = (
        get_dinov2_embedding(
            image
        )
    )

    print(
        f"[DINOv2] Embedding shape: "
        f"{embedding.shape}",
        flush=True
    )

    print(
        f"[DINOv2] Embedding norm: "
        f"{np.linalg.norm(embedding):.4f}",
        flush=True
    )

    if (
        embedding.shape[0]
        !=
        index.d
    ):

        raise RuntimeError(
            f"Размерность embedding "
            f"({embedding.shape[0]}) "
            f"не совпадает с FAISS "
            f"({index.d})."
        )

    # 10. FAISS

    query = embedding.reshape(
        1,
        -1
    )

    scores, indices = index.search(
        query,
        k=1
    )

    wine_id = int(
        indices[0][0]
    )

    similarity = float(
        scores[0][0]
    )

    print(
        f"[FAISS] wine_id: "
        f"{wine_id}",
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
        similarity,
        box_detected
    )


# ============================================================
# PARSE WINE SITE
# ============================================================

def _unique_texts(
    tags,
    fallback
):

    if not tags:
        return [fallback]

    return list(
        dict.fromkeys(
            item.text.strip()
            for item in tags
        )
    )


def find_by_slug(
    wine_slug
):

    wine_url = (
        f"{SITE_BASE_URL}/wines/{wine_slug}"
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

            # DESCRIPTION

            tag = soup.find(
                "p",
                class_="wine-page__description"
            )

            description = (
                tag.text.strip()
                if tag
                else "Нет описания"
            )

            # NAME

            tag = soup.find(
                "h1",
                class_="wine-main-title-block__title"
            )

            wine_name = (
                tag.text.strip()
                if tag
                else "Нет названия"
            )

            # FACTORY

            tag = soup.find(
                "a",
                class_="wine-main-title-block__manufacturer"
            )

            factory = (
                tag.text.strip()
                if tag
                else "Нет информации о заводе"
            )

            # RATING

            tag = soup.find(
                "span",
                class_="wine-main-title-block__rating-text"
            )

            if tag:

                rate = tag.text.strip()

                rate = float(
                    rate.split()[-1]
                )

            else:

                rate = "Нет рейтинга"

            # ATCC

            atcc_list = _unique_texts(
                soup.find_all(
                    "p",
                    class_="wine-detail-info__detail-value"
                ),
                "Нет информации"
            )

            # NUM

            num_list = _unique_texts(
                soup.find_all(
                    "p",
                    class_="wine-hero-block__card-value"
                ),
                "Нет информации"
            )

            # DISHES

            dishes_list = _unique_texts(
                soup.find_all(
                    "p",
                    class_="wine-dish-item__name"
                ),
                "Нет блюд"
            )

            # IMAGE

            wine_image = (
                "Нет картинки"
            )

            image_tag = soup.find(
                "img",
                class_="wine-hero-block__bottle"
            )

            if (
                image_tag
                and image_tag.get("src")
            ):

                image_src = (
                    image_tag["src"]
                )

                if image_src.startswith("/"):
                    image_url = (
                        f"{IMAGE_BASE_URL}"
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

                    if (
                        image_result.status_code
                        == 200
                    ):

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
                        "[SITE] Не удалось "
                        "скачать картинку товара: "
                        f"{e}",
                        flush=True
                    )

            print(
                f"[SITE] Данные успешно "
                f"получены. {wine_name}, "
                f"Описание: {description}, "
                f"{factory}, {rate}, "
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

            msg = (
                "Ошибка подключения "
                "к сайту"
            )

            description = msg
            wine_name = msg
            factory = msg
            rate = msg
            wine_image = msg

            atcc_list = [msg]
            num_list = [msg]
            dishes_list = [msg]

    except Exception as e:

        print(
            "[SITE] Не удалось распарсить "
            f"страницу: {e}",
            flush=True
        )

        traceback.print_exc()

        msg = (
            "Не удалось загрузить "
            "информацию"
        )

        description = msg
        wine_name = msg
        factory = msg
        rate = msg
        wine_image = msg

        atcc_list = [msg]
        num_list = [msg]
        dishes_list = [msg]

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

def somelier(wine_slug):
    wine_url = f"{SITE_BASE_URL}/wines/{wine_slug}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Referer": SITE_BASE_URL
    }

    recommended_wines = []

    with requests.Session() as session:
        session.headers.update(headers)
        try:
            response = session.get(wine_url, timeout=5)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                recommendation_cards = soup.find_all('a', class_='wine-item', limit=5)

                for idx, card in enumerate(recommendation_cards, 1):
                    # 1. Извлекаем название
                    title_tag = card.find('h2', class_='wine-item__title') or card.find('p', class_='wine-item__title')
                    title_text = title_tag.text.strip() if title_tag else f"Вино #{idx}"

                    card_image_b64 = "Нет картинки"

                    # 2. Ищем ТОЧНО тег картинки вина по классу 'wine-item__img'
                    img_tag = card.find('img', class_='wine-item__img')

                    # Если по классу не нашлось, ищем тег img внутри контейнера 'wine-item__img-container'
                    if not img_tag:
                        img_container = card.find('div', class_='wine-item__img-container')
                        if img_container:
                            img_tag = img_container.find('img')

                    if img_tag:
                        # 3. Берём srcset или src
                        raw_src = img_tag.get('srcset') or img_tag.get('src')

                        if raw_src:
                            # Парсим srcset: берем вариант с максимальным разрешением (последний в списке)
                            if ',' in raw_src:
                                urls = [item.strip().split(' ')[0] for item in raw_src.split(',') if item.strip()]
                                raw_src = urls[-1] if urls else raw_src
                            elif ' ' in raw_src:
                                raw_src = raw_src.split(' ')[0]

                            # Собираем валидный адрес (в вашем HTML это абсолютный URL https://api.vino-svoe.ru/...)
                            img_url = urljoin(SITE_BASE_URL, raw_src)

                            try:
                                time.sleep(0.1)
                                img_res = session.get(img_url, timeout=4)

                                if img_res.status_code == 200 and len(img_res.content) > 0:
                                    b64_data = base64.b64encode(img_res.content).decode('utf-8')
                                    card_image_b64 = f"data:image/webp;base64,{b64_data}"
                                else:
                                    print(f"[SITE #{idx}] Ошибка {img_res.status_code} по ссылке: {img_url}",
                                          flush=True)

                            except Exception as e:
                                print(f"[SITE #{idx}] Ошибка скачивания картинки '{title_text}': {e}", flush=True)

                    recommended_wines.append([title_text, card_image_b64])

        except Exception as e:
            print(f"[SITE] Не удалось распарсить страницу: {e}", flush=True)
            traceback.print_exc()

    if not recommended_wines:
        recommended_wines = [["Нет информации", "Нет картинки"] for _ in range(5)]

    return recommended_wines
# ============================================================
# FETCH WINE
# ============================================================

def fetch_wine_data(wine_id):

    wine_slug = id_to_slug.get(wine_id)

    if wine_slug is None:
        raise Exception(
            f"Индекс {wine_id} есть в FAISS, "
            f"но отсутствует в {MAPPING_FILE_PATH}"
        )
    (
        wine_url, wine_slug, description, wine_name, factory,
        rate, atcc_list, num_list, dishes_list, wine_image
    ) = find_by_slug(wine_slug)

    # Получаем сомелье-рекомендации
    recommended_wines = somelier(wine_slug)

    # Возвращаем ВСЕ 11 элементов единым плоским кортежем
    return (
        wine_url, wine_slug, description, wine_name, factory,
        rate, atcc_list, num_list, dishes_list, wine_image, recommended_wines
    )


# ============================================================
# RESPONSE
# ============================================================

def parsed_info(
    wine_url, wine_slug, description, wine_name, factory,
    rate, atcc_list, num_list, dishes_list, wine_image, recommended_wines
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
            "wine_image": wine_image,
            "top5": recommended_wines
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
# /feed
# ============================================================

@app.get(
    "/feed",
    response_model=FeedResponse
)
async def get_feed():

    global _feed_cache_bytes
    global _feed_cache_timestamp

    # --------------------------------------------------------
    # Быстрый путь:
    # cache ещё действителен.
    #
    # Здесь НЕ происходит:
    # - запросов новостей;
    # - запросов vino-svoe.ru;
    # - работы с FAISS;
    # - формирования JSON.
    # --------------------------------------------------------

    if _feed_cache_is_valid():

        return _feed_response()

    # --------------------------------------------------------
    # Cache истёк.
    #
    # Только один запрос обновляет его.
    # --------------------------------------------------------

    async with _feed_cache_lock:

        # Пока текущий запрос ждал lock,
        # другой запрос уже мог обновить cache.

        if _feed_cache_is_valid():

            return _feed_response()

        try:

            # requests блокирующий, поэтому обновление
            # выполняется вне event loop.
            new_cache = await asyncio.to_thread(
                _build_feed_response
            )

            _feed_cache_bytes = (
                new_cache
            )

            _feed_cache_timestamp = (
                asyncio
                .get_running_loop()
                .time()
            )

            print(
                "[FEED] Кэш успешно обновлён.",
                flush=True
            )

            return _feed_response()

        except Exception as e:

            print(
                f"[FEED] Ошибка обновления "
                f"кэша: {e}",
                flush=True
            )

            traceback.print_exc()

            # Если старый cache существует,
            # лучше отдать его, чем полностью
            # сломать экран ленты.
            if _feed_cache_bytes is not None:

                print(
                    "[FEED] Используется "
                    "устаревший кэш.",
                    flush=True
                )

                return _feed_response(
                    stale=True
                )

            raise HTTPException(
                status_code=500,
                detail=(
                    "Не удалось сформировать "
                    "винную ленту: "
                    f"{str(e)}"
                )
            )


# ============================================================
# /api/recognize
# ============================================================

@app.post(
    "/api/recognize"
)
async def recognize_wine(
    data: ImageRequest
):

    print(
        "\n[START] Начало обработки",
        flush=True
    )

    # 1. BASE64 -> PIL

    try:

        pure_base64 = (
            data.image_base64
            .split(",")[-1]
        )

        image_data = base64.b64decode(
            pure_base64
        )

        image = Image.open(
            io.BytesIO(
                image_data
            )
        ).convert("RGB")

        orig_w, orig_h = (
            image.size
        )

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

    # 2. ML + mapping + site

    try:

        print(
            "[ML] Запуск YOLO + "
            "DINOv2-reg + FAISS",
            flush=True
        )

        (
            wine_id,
            similarity,
            box_detected
        ) = await asyncio.to_thread(
            run_ml_pipeline,
            image,
            orig_w,
            orig_h,
            YOLO_INPUT_WIDTH,
            YOLO_INPUT_HEIGHT
        )

        print(
            f"[FAISS] Выдал ID: "
            f"{wine_id}",
            flush=True
        )

        print(
            f"[FAISS] Cosine similarity: "
            f"{similarity:.4f}",
            flush=True
        )

        # Фильтр

        if not is_wine_photo(
            similarity,
            box_detected
        ):

            print(
                "[FILTER] Фото отклонено "
                "как не-вино: "
                f"similarity={similarity:.4f}, "
                f"box_detected={box_detected}",
                flush=True
            )

            raise HTTPException(
                status_code=404,
                detail=(
                    "Вино на фото "
                    "не распознано"
                )
            )

        result = await asyncio.to_thread(
            fetch_wine_data,
            wine_id
        )

        print(
            f"[MAPPING] Данные собраны. "
            f"Slug: {result[1]}",
            flush=True
        )

    except HTTPException:

        raise

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
        *result
    )


# ============================================================
# /api/memory
# ============================================================

@app.post(
    "/api/memory"
)
async def memory_wine(
    data: WineSlug
):

    try:

        result = await asyncio.to_thread(
            find_by_slug,
            data.memory_slug
        )

        print(
            f"[MEMORY] Данные собраны. "
            f"Slug: {result[1]}",
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
        *result
    )


# ============================================================
# /api/test_for_slug
# ============================================================

@app.post(
    "/api/test_for_slug"
)
async def test_by_slug(
    data: ImageTest
):

    try:

        pure_base64 = (
            data.test_slug
            .split(",")[-1]
        )

        image_data = base64.b64decode(
            pure_base64
        )

        image = Image.open(
            io.BytesIO(
                image_data
            )
        ).convert("RGB")

        orig_w, orig_h = (
            image.size
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                "Некорректная base64 строка: "
                f"{str(e)}"
            )
        )

    try:

        (
            wine_id,
            distance,
            _
        ) = await asyncio.to_thread(
            run_ml_pipeline,
            image,
            orig_w,
            orig_h,
            640,
            640
        )

        conn = get_db_connection()

        cursor = conn.cursor()

        cursor.execute(
            "SELECT wine_slug "
            "FROM wines "
            "WHERE id = %s;",
            (wine_id,)
        )

        result = cursor.fetchone()

        cursor.close()

        conn.close()

        if not result:

            raise HTTPException(
                status_code=404,
                detail=(
                    f"Индекс {wine_id} найден "
                    f"в FAISS, но отсутствует "
                    f"в БД"
                )
            )

        wine_slug = result[0]

    except HTTPException as http_ex:

        raise http_ex

    except Exception as e:

        print(
            f"[ERROR] Сбой в пайплайне: "
            f"{str(e)}",
            flush=True
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    payload = {
        "wine_slug":
            wine_slug
    }

    json_bytes = json.dumps(
        payload,
        ensure_ascii=False
    ).encode("utf-8")

    return Response(
        content=json_bytes,
        media_type="application/json",
        headers={
            "Content-Length":
                str(
                    len(json_bytes)
                ),
            "Cache-Control":
                "no-transform"
        }
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
