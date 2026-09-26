import base64
import io
import os
import sys
import json
import asyncio
import traceback
import re
import time

from concurrent.futures import ThreadPoolExecutor
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
# FEED CACHE / SETTINGS
# ============================================================

# 6 часов
FEED_CACHE_TTL = 6 * 60 * 60

# Количество элементов
FEED_NEWS_LIMIT = 20
FEED_WINE_LIMIT = 30

# Таймауты внешних запросов
FEED_CATEGORY_TIMEOUT = 8
FEED_ARTICLE_TIMEOUT = 8
FEED_WINE_TIMEOUT = 5

# Главная страница раздела статей
FEED_ARTICLES_CATEGORY_URL = (
    f"{SITE_BASE_URL}/category/articles"
)

# Уже сериализованный JSON
_feed_cache_bytes = None

# Время последнего успешного обновления
_feed_cache_timestamp = 0.0

# Защита от параллельного обновления
_feed_cache_lock = asyncio.Lock()


# ============================================================
# FEED: CLEAN SLUG
# ============================================================

def clean_feed_wine_slug(slug: str) -> str:
    """
    Убирает технические хвосты:

        vino_failed
        vino_faile
        vino_fail
        vino_failed_failed
        vino_faile_failed

    Обычный slug не изменяется.
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
# FEED: URL HELPERS
# ============================================================

def _absolute_site_url(
    value: str
) -> str:

    if not value:
        return ""

    value = value.strip()

    if value.startswith("//"):
        return f"https:{value}"

    return urljoin(
        SITE_BASE_URL,
        value
    )


def _is_article_url(
    url: str
) -> bool:

    try:

        parsed = urlparse(url)

        base_host = urlparse(
            SITE_BASE_URL
        ).netloc

        if (
            parsed.netloc
            and parsed.netloc != base_host
        ):
            return False

        path = parsed.path.rstrip("/")

        return (
            path.startswith("/articles/")
            and len(path) > len("/articles/")
        )

    except Exception:

        return False


# ============================================================
# FEED: TEXT HELPERS
# ============================================================

def _clean_feed_text(
    value: str
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

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def _short_feed_description(
    value: str,
    limit: int = 280
) -> str:

    text = _clean_feed_text(
        value
    )

    if not text:
        return ""

    if len(text) <= limit:
        return text

    shortened = (
        text[:limit]
        .rsplit(" ", 1)[0]
        .rstrip(".,;:!?-")
    )

    return shortened + "..."


# ============================================================
# FEED: DISCOVER CATEGORIES
# ============================================================

def _discover_feed_categories():

    """
    Заходит на:

        /category/articles

    и автоматически ищет ссылки
    на подразделы этого раздела.
    """

    response = requests.get(
        FEED_ARTICLES_CATEGORY_URL,
        headers=FEED_HEADERS,
        timeout=FEED_CATEGORY_TIMEOUT
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    categories = {}

    base_host = urlparse(
        SITE_BASE_URL
    ).netloc

    for link in soup.find_all(
        "a",
        href=True
    ):

        href = link.get(
            "href",
            ""
        ).strip()

        if not href:
            continue

        absolute_url = _absolute_site_url(
            href
        )

        try:

            parsed = urlparse(
                absolute_url
            )

        except Exception:

            continue

        if (
            parsed.netloc
            and parsed.netloc != base_host
        ):
            continue

        path = parsed.path.rstrip("/")

        if not path.startswith(
            "/category/articles/"
        ):
            continue

        if path == "/category/articles":
            continue

        name = _clean_feed_text(
            link.get_text(
                " ",
                strip=True
            )
        )

        if not name:
            continue

        categories[
            absolute_url
        ] = name

    result = [
        {
            "url": url,
            "name": name
        }
        for url, name in categories.items()
    ]

    print(
        f"[FEED][CATEGORIES] "
        f"Найдено подразделов: {len(result)}",
        flush=True
    )

    for category in result:

        print(
            f"[FEED][CATEGORIES] "
            f"{category['name']} -> "
            f"{category['url']}",
            flush=True
        )

    return result


# ============================================================
# FEED: FIND ARTICLE LINKS
# ============================================================

def _extract_article_links(
    soup: BeautifulSoup
):

    links = []
    seen = set()

    for link in soup.find_all(
        "a",
        href=True
    ):

        href = link.get(
            "href",
            ""
        ).strip()

        absolute_url = _absolute_site_url(
            href
        )

        if not _is_article_url(
            absolute_url
        ):
            continue

        parsed = urlparse(
            absolute_url
        )

        clean_url = (
            f"{parsed.scheme}://"
            f"{parsed.netloc}"
            f"{parsed.path}"
        )

        if clean_url in seen:
            continue

        seen.add(
            clean_url
        )

        links.append(
            clean_url
        )

    return links


# ============================================================
# FEED: CATEGORY ARTICLES
# ============================================================

def _fetch_category_articles(
    category_url: str,
    category_name: str
):

    response = requests.get(
        category_url,
        headers=FEED_HEADERS,
        timeout=FEED_CATEGORY_TIMEOUT
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    article_urls = _extract_article_links(
        soup
    )

    result = []

    for article_url in article_urls:

        result.append({
            "url": article_url,
            "category": category_name
        })

    print(
        f"[FEED][CATEGORY] "
        f"{category_name}: "
        f"{len(result)} статей",
        flush=True
    )

    return result


# ============================================================
# FEED: DATE
# ============================================================

RUSSIAN_MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}


def _parse_article_date(
    value: str
) -> str:

    if not value:
        return ""

    value = _clean_feed_text(
        value
    )

    # --------------------------------------------------------
    # ISO
    # --------------------------------------------------------

    try:

        normalized = (
            value
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
            dt.astimezone(
                timezone.utc
            )
            .isoformat()
            .replace(
                "+00:00",
                "Z"
            )
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # Русская дата
    # --------------------------------------------------------

    match = re.search(
        r"(\d{1,2})\s+"
        r"(января|февраля|марта|апреля|мая|июня|"
        r"июля|августа|сентября|октября|ноября|декабря)"
        r"(?:\s+(\d{4}))?",
        value.lower()
    )

    if match:

        day = int(
            match.group(1)
        )

        month = RUSSIAN_MONTHS[
            match.group(2)
        ]

        year = (
            int(match.group(3))
            if match.group(3)
            else datetime.now(
                timezone.utc
            ).year
        )

        try:

            dt = datetime(
                year,
                month,
                day,
                tzinfo=timezone.utc
            )

            return (
                dt.isoformat()
                .replace(
                    "+00:00",
                    "Z"
                )
            )

        except Exception:
            pass

    return ""


# ============================================================
# FEED: ARTICLE IMAGE
# ============================================================

def _extract_article_image(
    soup: BeautifulSoup
) -> str:

    # --------------------------------------------------------
    # OpenGraph
    # --------------------------------------------------------

    for property_name in (
        "og:image",
        "twitter:image"
    ):

        meta = soup.find(
            "meta",
            attrs={
                "property": property_name
            }
        )

        if not meta:

            meta = soup.find(
                "meta",
                attrs={
                    "name": property_name
                }
            )

        if meta:

            content = (
                meta.get(
                    "content",
                    ""
                ).strip()
            )

            if content:

                return _absolute_site_url(
                    content
                )

    # --------------------------------------------------------
    # Основные классы изображения
    # --------------------------------------------------------

    preferred_classes = (
        "article__image",
        "article-image",
        "article-page__image",
        "article-page__img",
        "news-card__image",
        "article-card__image",
    )

    for class_name in preferred_classes:

        image = soup.find(
            "img",
            class_=class_name
        )

        if image:

            src = (
                image.get("src")
                or image.get("data-src")
                or ""
            ).strip()

            if src:

                return _absolute_site_url(
                    src
                )

    # --------------------------------------------------------
    # srcset
    # --------------------------------------------------------

    for image in soup.find_all(
        "img"
    ):

        srcset = (
            image.get(
                "srcset",
                ""
            ).strip()
        )

        if srcset:

            variants = []

            for part in srcset.split(","):

                part = part.strip()

                if not part:
                    continue

                url = part.split()[0]

                variants.append(
                    url
                )

            if variants:

                return _absolute_site_url(
                    variants[-1]
                )

        src = (
            image.get("src")
            or image.get("data-src")
            or ""
        ).strip()

        if src:

            return _absolute_site_url(
                src
            )

    return ""


# ============================================================
# FEED: ARTICLE DESCRIPTION
# ============================================================

def _extract_article_description(
    soup: BeautifulSoup
) -> str:

    # meta description
    meta = soup.find(
        "meta",
        attrs={
            "name": "description"
        }
    )

    if meta:

        content = (
            meta.get(
                "content",
                ""
            ).strip()
        )

        if content:

            return _short_feed_description(
                content
            )

    # og description
    meta = soup.find(
        "meta",
        attrs={
            "property": "og:description"
        }
    )

    if meta:

        content = (
            meta.get(
                "content",
                ""
            ).strip()
        )

        if content:

            return _short_feed_description(
                content
            )

    # Первый нормальный абзац
    h1 = soup.find("h1")

    if h1:

        for element in h1.find_all_next(
            ["p"]
        ):

            text = _clean_feed_text(
                element.get_text(
                    " ",
                    strip=True
                )
            )

            if len(text) >= 50:

                return _short_feed_description(
                    text
                )

    return ""


# ============================================================
# FEED: ARTICLE DATE FROM PAGE
# ============================================================

def _extract_article_date(
    soup: BeautifulSoup
) -> str:

    # --------------------------------------------------------
    # time datetime
    # --------------------------------------------------------

    for time_tag in soup.find_all(
        "time"
    ):

        datetime_value = (
            time_tag.get(
                "datetime",
                ""
            ).strip()
        )

        parsed = _parse_article_date(
            datetime_value
        )

        if parsed:
            return parsed

        visible_text = (
            time_tag.get_text(
                " ",
                strip=True
            )
        )

        parsed = _parse_article_date(
            visible_text
        )

        if parsed:
            return parsed

    # --------------------------------------------------------
    # Ищем русскую дату в тексте
    # --------------------------------------------------------

    text = soup.get_text(
        " ",
        strip=True
    )

    match = re.search(
        r"\b\d{1,2}\s+"
        r"(?:января|февраля|марта|апреля|мая|июня|"
        r"июля|августа|сентября|октября|ноября|декабря)"
        r"(?:\s+\d{4})?\b",
        text.lower()
    )

    if match:

        return _parse_article_date(
            match.group(0)
        )

    return ""


# ============================================================
# FEED: ARTICLE TITLE
# ============================================================

def _extract_article_title(
    soup: BeautifulSoup
) -> str:

    h1 = soup.find("h1")

    if h1:

        title = _clean_feed_text(
            h1.get_text(
                " ",
                strip=True
            )
        )

        if title:
            return title

    meta = soup.find(
        "meta",
        attrs={
            "property": "og:title"
        }
    )

    if meta:

        title = (
            meta.get(
                "content",
                ""
            ).strip()
        )

        if title:
            return title

    if soup.title:

        title = _clean_feed_text(
            soup.title.get_text(
                " ",
                strip=True
            )
        )

        if title:
            return title

    return ""


# ============================================================
# FEED: LOAD SINGLE ARTICLE
# ============================================================

def _fetch_feed_article(
    article
):

    article_url = article["url"]
    category = article["category"]

    try:

        response = requests.get(
            article_url,
            headers=FEED_HEADERS,
            timeout=FEED_ARTICLE_TIMEOUT
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        title = _extract_article_title(
            soup
        )

        if not title:
            return None

        description = (
            _extract_article_description(
                soup
            )
        )

        image_url = (
            _extract_article_image(
                soup
            )
        )

        published_at = (
            _extract_article_date(
                soup
            )
        )

        if not published_at:

            print(
                f"[FEED][ARTICLE] "
                f"Не удалось определить дату: "
                f"{article_url}",
                flush=True
            )

            return None

        return {
            "title": title,
            "description": description,
            "url": article_url,
            "image_url": image_url,
            "source": (
                f"Своё Вино · {category}"
            ),
            "published_at": published_at,
        }

    except Exception as e:

        print(
            f"[FEED][ARTICLE] "
            f"Ошибка {article_url}: {e}",
            flush=True
        )

        return None


# ============================================================
# FEED: LOAD NEWS
# ============================================================

def _load_feed_news():

    # --------------------------------------------------------
    # 1. Автоматически ищем подразделы
    # --------------------------------------------------------

    try:

        categories = (
            _discover_feed_categories()
        )

    except Exception as e:

        print(
            f"[FEED][CATEGORIES] "
            f"Ошибка загрузки категорий: {e}",
            flush=True
        )

        traceback.print_exc()

        categories = []

    # Если подразделы не нашли,
    # всё равно пробуем главную страницу.
    if not categories:

        categories = [
            {
                "url":
                    FEED_ARTICLES_CATEGORY_URL,
                "name":
                    "Статьи"
            }
        ]

    # --------------------------------------------------------
    # 2. Собираем URL статей
    # --------------------------------------------------------

    category_articles = []

    for category in categories:

        try:

            items = _fetch_category_articles(
                category["url"],
                category["name"]
            )

            category_articles.extend(
                items
            )

        except Exception as e:

            print(
                f"[FEED][CATEGORY] "
                f"Не удалось загрузить "
                f"{category['url']}: {e}",
                flush=True
            )

    # --------------------------------------------------------
    # 3. Дедупликация
    # --------------------------------------------------------

    unique_articles = {}

    for item in category_articles:

        unique_articles.setdefault(
            item["url"],
            item
        )

    candidates = list(
        unique_articles.values()
    )

    if not candidates:

        raise RuntimeError(
            "На странице "
            f"{FEED_ARTICLES_CATEGORY_URL} "
            "не найдено статей"
        )

    # --------------------------------------------------------
    # 4. Группировка по подразделам
    # --------------------------------------------------------

    grouped = {}

    for item in candidates:

        grouped.setdefault(
            item["category"],
            []
        ).append(
            item
        )

    # --------------------------------------------------------
    # 5. Сначала берём по одной статье
    #    из каждого подраздела
    # --------------------------------------------------------

    selected = []
    selected_urls = set()

    for category_name in sorted(
        grouped.keys()
    ):

        category_items = grouped[
            category_name
        ]

        if not category_items:
            continue

        item = category_items[0]

        if item["url"] not in selected_urls:

            selected.append(
                item
            )

            selected_urls.add(
                item["url"]
            )

    # --------------------------------------------------------
    # 6. Добиваем список остальными статьями
    # --------------------------------------------------------

    for item in candidates:

        if len(selected) >= (
            FEED_NEWS_LIMIT * 2
        ):
            break

        if item["url"] in selected_urls:
            continue

        selected.append(
            item
        )

        selected_urls.add(
            item["url"]
        )

    # --------------------------------------------------------
    # 7. Загружаем страницы статей
    #
    # Только при обновлении cache.
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=8
    ) as executor:

        results = list(
            executor.map(
                _fetch_feed_article,
                selected
            )
        )

    news = [
        item
        for item in results
        if item is not None
    ]

    # --------------------------------------------------------
    # 8. Финальная дедупликация
    # --------------------------------------------------------

    unique_news = {}

    for item in news:

        unique_news.setdefault(
            item["url"],
            item
        )

    news = list(
        unique_news.values()
    )

    # --------------------------------------------------------
    # 9. Самые новые сверху
    # --------------------------------------------------------

    news.sort(
        key=lambda item:
            item["published_at"],
        reverse=True
    )

    # --------------------------------------------------------
    # 10. Ограничение
    # --------------------------------------------------------

    news = news[
        :FEED_NEWS_LIMIT
    ]

    print(
        f"[FEED][NEWS] "
        f"Итоговое количество статей: "
        f"{len(news)}",
        flush=True
    )

    for item in news:

        print(
            f"[FEED][NEWS] "
            f"{item['published_at']} | "
            f"{item['source']} | "
            f"{item['title']}",
            flush=True
        )

    return news


# ============================================================
# FEED: WINE
# ============================================================

def _fetch_feed_wine(
    slug: str
):
    """
    Лёгкая версия find_by_slug().

    Не скачивает изображение.
    Не конвертирует изображение в base64.
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

    try:

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
            name_tag.get_text(
                strip=True
            )
            if name_tag
            else ""
        )

        image_src = (
            image_tag.get(
                "src",
                ""
            )
            if image_tag
            else ""
        )

        if not name or not image_src:
            return None

        image_url = _absolute_site_url(
            image_src
        )

        # Если картинка относительная
        # и должна идти через API vino-svoe.
        if image_src.startswith("/"):
            image_url = (
                f"{IMAGE_BASE_URL}"
                f"{image_src}"
            )

        return {
            "slug": slug,
            "name": name,
            "image_url": image_url,
        }

    except Exception as e:

        print(
            f"[FEED][WINE] "
            f"Ошибка загрузки {slug}: {e}",
            flush=True
        )

        return None


# ============================================================
# FEED: LOAD WINES
# ============================================================

def _load_feed_wines():

    candidates = []
    seen = set()

    # Берём существующий mapping:
    #
    # FAISS ID -> slug
    #
    # Никакой новой БД не создаём.
    for wine_id in sorted(
        id_to_slug
    ):

        slug = clean_feed_wine_slug(
            id_to_slug[wine_id]
        )

        if not slug:
            continue

        if slug in seen:
            continue

        seen.add(
            slug
        )

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

    # Запросы выполняются только
    # во время обновления кэша.
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

    return json.dumps(
        payload.model_dump(),
        ensure_ascii=False,
        separators=(",", ":")
    ).encode(
        "utf-8"
    )


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
        ).decode(
            "utf-8"
        )

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

    nh = int(
        h * scale
    )

    nw = int(
        w * scale
    )

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
# ФИЛЬТР
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

    img_np = np.array(
        image
    )

    img_bgr = cv2.cvtColor(
        img_np,
        cv2.COLOR_RGB2BGR
    )

    input_tensor, scale, pad = (
        letterbox_preprocess(
            img_bgr,
            (
                input_width,
                input_height
            )
        )
    )

    outputs = session.run(
        None,
        {
            input_name:
                input_tensor
        }
    )

    prediction = outputs[0]

    pred = prediction[0]

    if (
        pred.shape[0]
        <
        pred.shape[1]
    ):

        pred = pred.T

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

    save_crop_for_debugging(
        image,
        label="dinov2_input"
    )

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


# ============================================================
# SOMELIER
# ============================================================

def somelier(
    wine_slug
):

    wine_url = (
        f"{SITE_BASE_URL}/wines/{wine_slug}"
    )

    headers = {
        "User-Agent":
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/120.0.0.0 "
            "Safari/537.36",

        "Accept":
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,image/avif,"
            "image/webp,*/*;q=0.8",

        "Referer":
            SITE_BASE_URL
    }

    recommended_wines = []

    with requests.Session() as session:

        session.headers.update(
            headers
        )

        try:

            response = session.get(
                wine_url,
                timeout=5
            )

            if response.status_code == 200:

                soup = BeautifulSoup(
                    response.text,
                    "html.parser"
                )

                recommendation_cards = (
                    soup.find_all(
                        "a",
                        class_="wine-item",
                        limit=5
                    )
                )

                for idx, card in enumerate(
                    recommendation_cards,
                    1
                ):

                    title_tag = (
                        card.find(
                            "h2",
                            class_="wine-item__title"
                        )
                        or
                        card.find(
                            "p",
                            class_="wine-item__title"
                        )
                    )

                    title_text = (
                        title_tag.text.strip()
                        if title_tag
                        else f"Вино #{idx}"
                    )

                    card_image_b64 = (
                        "Нет картинки"
                    )

                    img_tag = card.find(
                        "img",
                        class_="wine-item__img"
                    )

                    if not img_tag:

                        img_container = (
                            card.find(
                                "div",
                                class_=
                                "wine-item__img-container"
                            )
                        )

                        if img_container:

                            img_tag = (
                                img_container.find(
                                    "img"
                                )
                            )

                    if img_tag:

                        raw_src = (
                            img_tag.get(
                                "srcset"
                            )
                            or
                            img_tag.get(
                                "src"
                            )
                        )

                        if raw_src:

                            if "," in raw_src:

                                urls = [
                                    item.strip().split(" ")[0]
                                    for item in raw_src.split(",")
                                    if item.strip()
                                ]

                                raw_src = (
                                    urls[-1]
                                    if urls
                                    else raw_src
                                )

                            elif " " in raw_src:

                                raw_src = (
                                    raw_src.split(" ")[0]
                                )

                            img_url = urljoin(
                                SITE_BASE_URL,
                                raw_src
                            )

                            try:

                                time.sleep(
                                    0.1
                                )

                                img_res = session.get(
                                    img_url,
                                    timeout=4
                                )

                                if (
                                    img_res.status_code
                                    == 200
                                    and len(
                                        img_res.content
                                    ) > 0
                                ):

                                    b64_data = (
                                        base64.b64encode(
                                            img_res.content
                                        )
                                        .decode("utf-8")
                                    )

                                    card_image_b64 = (
                                        "data:image/webp;base64,"
                                        f"{b64_data}"
                                    )

                                else:

                                    print(
                                        f"[SITE #{idx}] "
                                        f"Ошибка "
                                        f"{img_res.status_code} "
                                        f"по ссылке: "
                                        f"{img_url}",
                                        flush=True
                                    )

                            except Exception as e:

                                print(
                                    f"[SITE #{idx}] "
                                    f"Ошибка скачивания "
                                    f"картинки "
                                    f"'{title_text}': "
                                    f"{e}",
                                    flush=True
                                )

                    recommended_wines.append(
                        [
                            title_text,
                            card_image_b64
                        ]
                    )

        except Exception as e:

            print(
                f"[SITE] Не удалось "
                f"распарсить страницу: {e}",
                flush=True
            )

            traceback.print_exc()

    if not recommended_wines:

        recommended_wines = [
            [
                "Нет информации",
                "Нет картинки"
            ]
            for _ in range(5)
        ]

    return recommended_wines


# ============================================================
# FETCH WINE
# ============================================================

def fetch_wine_data(
    wine_id
):

    wine_slug = id_to_slug.get(
        wine_id
    )

    if wine_slug is None:

        raise Exception(
            f"Индекс {wine_id} есть в FAISS, "
            f"но отсутствует в "
            f"{MAPPING_FILE_PATH}"
        )

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

    recommended_wines = (
        somelier(
            wine_slug
        )
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
        wine_image,
        recommended_wines
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
    wine_image,
    recommended_wines
):

    def pick(
        lst,
        i
    ):

        return (
            lst[i]
            if len(lst) > i
            else "Нет информации"
        )

    payload = {
        "status": "success",
        "url": wine_url,
        "parsed_data": {
            "name": wine_name,
            "description": description,
            "factory": factory,
            "rate": rate,
            "area": pick(
                atcc_list,
                0
            ),
            "sort": pick(
                atcc_list,
                1
            ),
            "type": pick(
                atcc_list,
                2
            ),
            "color": pick(
                atcc_list,
                3
            ),
            "temperature": pick(
                num_list,
                0
            ),
            "alcohol": pick(
                num_list,
                1
            ),
            "dishes": dishes_list,
            "wine_image": wine_image,
            "top5": recommended_wines
        }
    }

    json_bytes = json.dumps(
        payload,
        ensure_ascii=False
    ).encode(
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
                str(
                    len(json_bytes)
                ),
            "Cache-Control":
                "no-transform"
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
    # Быстрый путь
    # --------------------------------------------------------

    if _feed_cache_is_valid():

        return _feed_response()

    # --------------------------------------------------------
    # Обновление cache
    # --------------------------------------------------------

    async with _feed_cache_lock:

        # Пока ждали lock,
        # другой запрос мог обновить cache.

        if _feed_cache_is_valid():

            return _feed_response()

        try:

            # requests блокирующий,
            # поэтому уходим из event loop.
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

            # Если есть старый cache,
            # отдаём его.
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

    # --------------------------------------------------------
    # BASE64 -> PIL
    # --------------------------------------------------------

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
        ).convert(
            "RGB"
        )

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

    # --------------------------------------------------------
    # ML + mapping + site
    # --------------------------------------------------------

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

        if similarity < 0.8:

            print(
                "[FILTER] Совпадение "
                f"отброшено: "
                f"Top-1 F1 "
                f"({similarity:.4f}) < 0.80",
                flush=True
            )

            raise HTTPException(
                status_code=404,
                detail=(
                    "Точного совпадения "
                    "в базе не найдено"
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
        ).convert(
            "RGB"
        )

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
    ).encode(
        "utf-8"
    )

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
