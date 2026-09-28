# Wine Backend

Backend-сервис приложения для распознавания вина по фотографии.

Основной ML-пайплайн:

* **YOLO / ONNX Runtime** — обнаружение этикетки на фотографии;
* **DINOv2 with Registers** — получение визуального embedding;
* **FAISS** — поиск наиболее похожего вина по векторному представлению;
* **PostgreSQL** — хранение данных о винах;
* **FastAPI** — REST API;
* **BeautifulSoup / Requests** — получение дополнительной информации с сайта [vino-svoe.ru](https://vino-svoe.ru).

## Структура проекта

```text
Wine_back/
├── main.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── init.sql
├── label_detector.onnx
├── label_detector.onnx.data
├── wines_base.index
├── wines_mapping.json
└── render.yaml
```

### ML-модели и данные

* `label_detector.onnx` — модель обнаружения этикетки;
* `label_detector.onnx.data` — дополнительные данные ONNX-модели;
* `wines_base.index` — FAISS-индекс embeddings вин;
* `wines_mapping.json` — соответствие ID в FAISS и slug вина.

---

# Локальный запуск

## 1. Необходимое ПО

Для запуска необходимо установить:

* Git
* Docker
* Docker Compose

Проверить установку:

```bash
docker --version
docker compose version
```

## 2. Клонирование репозитория

```bash
git clone https://github.com/AlexSeal625/Wine_back.git
cd Wine_back
```

## 3. Запуск

Запустите backend и PostgreSQL:

```bash
docker compose up --build
```

Запуска сервера вручную с пробросом портов после сборки всех контейнеров:
```bash
docker compose run --rm --service-ports web python main.py
```

При первом запуске Docker:

1. создаст контейнер PostgreSQL;
2. создаст базу `wine_db`;
3. выполнит `init.sql`;
4. соберёт Docker-образ backend;
5. установит Python-зависимости;
6. запустит FastAPI на порту `8000`.

После успешного запуска API будет доступен по адресу:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

ReDoc:

```text
http://localhost:8000/redoc
```

## 4. Запуск в фоновом режиме

Если не нужно держать терминал открытым:

```bash
docker compose up --build -d
```

Проверить состояние контейнеров:

```bash
docker compose ps
```

Посмотреть логи backend:

```bash
docker compose logs -f web
```

Посмотреть логи PostgreSQL:

```bash
docker compose logs -f db
```

## 5. Остановка

Остановить контейнеры:

```bash
docker compose down
```

При этом данные PostgreSQL сохраняются в Docker volume `postgres_data`.

Чтобы полностью удалить контейнеры **вместе с локальной базой данных**:

```bash
docker compose down -v
```

После этого при следующем запуске PostgreSQL будет создан заново, а `init.sql` выполнится повторно.

---

# API

## Распознавание вина

### `POST /api/recognize`

Принимает фотографию вина в формате Base64.

Пример запроса:

```json
{
  "image_base64": "<BASE64_IMAGE>"
}
```

Сервер выполняет следующий pipeline:

```text
Изображение
    ↓
YOLO
    ↓
Обнаружение этикетки
    ↓
DINOv2
    ↓
Embedding
    ↓
FAISS
    ↓
Поиск ближайшего вина
    ↓
Получение данных по slug
    ↓
JSON-ответ
```

При успешном распознавании возвращается информация о вине, включая:

* название;
* производителя;
* описание;
* рейтинг;
* регион;
* сорт;
* тип;
* цвет;
* температуру подачи;
* крепость;
* блюда;
* оригинальное изображение вина;
* рекомендации похожих вин;
* метрики распознавания.

---

## Получение данных о вине

### `POST /api/memory`

Используется для получения информации о вине по его `slug`.

Пример:

```json
{
  "memory_slug": "wine-slug"
}
```

---

## Винная лента

### `GET /feed`

Возвращает данные для экрана винной ленты.

Ответ содержит:

```json
{
  "status": "success",
  "feed": {
    "news": [],
    "wines": []
  }
}
```

`news` содержит материалы винной тематики, а `wines` — рекомендации вин.

Данные ленты кэшируются на стороне backend. Время жизни кэша — **6 часов**.

---

## Тестирование распознавания

### `POST /api/test_for_slug`

Endpoint предназначен для тестирования ML-пайплайна на изображении.

Поддерживается передача изображения:

* как файла;
* через Base64.

---

# Конфигурация

Основные параметры находятся в `main.py`.

### DINOv2

Используется:

```text
facebook/dinov2-with-registers-small
```

Размерность embedding:

```text
384
```

Поэтому `wines_base.index` должен содержать векторы размерности `384`.

### YOLO

Модель:

```text
label_detector.onnx
```

Размер входного изображения:

```text
640 × 640
```

Порог обнаружения:

```text
0.25
```

### FAISS

Используется индекс:

```text
wines_base.index
```

Связь между ID FAISS и конкретным вином хранится в:

```text
wines_mapping.json
```

---

# Переменные окружения

Пороговые значения распознавания можно изменить через переменные окружения:

```bash
SIMILARITY_THRESHOLD_WITH_BOX=0.45
SIMILARITY_THRESHOLD_NO_BOX=0.55
```

Также можно изменить способ формирования embedding:

```bash
EMBEDDING_MODE=cls
```

Доступный альтернативный режим:

```bash
EMBEDDING_MODE=mean
```

Например:

```bash
docker compose down

SIMILARITY_THRESHOLD_WITH_BOX=0.50 \
SIMILARITY_THRESHOLD_NO_BOX=0.60 \
docker compose up --build
```

---

# Локальная разработка без Docker

При необходимости backend можно запускать непосредственно из Python.

Требуется **Python 3.10**.

Создание виртуального окружения:

```bash
python -m venv .venv
```

Активация в Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Установка PyTorch CPU:

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

Установка остальных зависимостей:

```bash
pip install -r requirements.txt
```

После этого необходимо запустить PostgreSQL и настроить параметры подключения к базе данных.

Запуск FastAPI:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

После запуска API будет доступен по адресу:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

> Для обычного локального запуска рекомендуется использовать Docker Compose: он автоматически поднимает PostgreSQL и backend с необходимыми ML-файлами.

---

# Docker-компоненты

Локальная инфраструктура состоит из двух сервисов:

```text
┌──────────────────────┐
│      FastAPI         │
│      Backend         │
│       :8000          │
└──────────┬───────────┘
           │
           │ PostgreSQL
           ▼
┌──────────────────────┐
│     PostgreSQL 15    │
│       :5432          │
└──────────────────────┘
```

PostgreSQL использует следующие параметры:

```text
Database: wine_db
User:     wine_user
Password: wine_password
Port:     5432
```

Инициализация базы выполняется автоматически из:

```text
init.sql
```

---

# Проверка работоспособности

После запуска откройте:

```text
http://localhost:8000/docs
```

Если Swagger UI открывается, FastAPI успешно запущен.

В логах backend должны появиться сообщения о загрузке:

```text
[YOLO] Модель загружена
[DINOv2] Модель успешно загружена
[FAISS] База успешно загружена.
```

После этого backend готов принимать запросы.

---

# Остановка и очистка

Остановить приложение:

```bash
docker compose down
```

Удалить контейнеры и базу:

```bash
docker compose down -v
```

Пересобрать backend после изменения кода:

```bash
docker compose up --build
```

Запустить уже собранную версию:

```bash
docker compose up -d
```

