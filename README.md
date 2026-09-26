# IT-Ledi — AML Alert Risk API (backend)

FastAPI-сервис, который оценивает вероятность того, что оповещение финансового
мониторинга будет передано на расследование (`eskalatsiya`), на основе истории
транзакций клиента. Модель — LightGBM, OOF ROC-AUC ≈ 0.6277.

Это backend-часть проекта. Фронтенд (Next.js-панель) живёт в отдельном репозитории.

## Быстрый старт (Python 3.10+)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Проверить, что сервис поднялся:

```bash
curl http://localhost:8000/health
# {"status":"ok","model_features":79}
```

Интерактивная документация API: http://localhost:8000/docs

## Структура

| Файл | Назначение |
|---|---|
| `main.py` | FastAPI-приложение и все эндпоинты |
| `features.py` | Инженерия признаков — идентична пайплайну обучения модели |
| `demo.py` | Генерация фоновой истории транзакций для живого демо |
| `model_full_data.pkl` | Обученная модель LightGBM (100% train-данных) |
| `feature_columns.pkl` | Список признаков в порядке, ожидаемом моделью |

## Эндпоинты

- `GET /health` — статус сервиса
- `POST /predict` — оценка одного оповещения по списку транзакций (JSON)
- `POST /predict/batch` — пакетная оценка: загрузка `signals.csv` + `transactions.csv/.parquet` → `predictions.csv`
- `POST /predict/batch/preview` — то же самое, но JSON-ответ (первые N строк) для отображения в интерфейсе
- `GET /feature-importance` — важность признаков модели

Пример запроса:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "signal_id": "SG_TEST",
    "signal_sanasi": "2026-01-15T00:00:00",
    "transactions": [
      {"tranzaksiya_vaqti":"2026-01-01T10:00:00","kirim_chiqim":"kirim","tranzaksiya_turi":"karta","miqdor_indeksi":0.5},
      {"tranzaksiya_vaqti":"2026-01-10T08:00:00","kirim_chiqim":"kirim","tranzaksiya_turi":"xalqaro","miqdor_indeksi":2.5}
    ]
  }'
```

## Формат входных файлов для пакетной загрузки

**Сигналы (`signals.csv`):**
```
signal_id,signal_sanasi
SG_000001,2025-07-11
```

**Транзакции (`transactions.csv` или `.parquet`):**
```
signal_id,tranzaksiya_vaqti,kirim_chiqim,tranzaksiya_turi,miqdor_indeksi
SG_000001,2025-01-01 10:00:00,kirim,karta,0.5
```

## Заметки

- CORS в `main.py` открыт на `*` для локальной разработки — на проде укажите
  конкретный домен фронтенда.
- `import lightgbm` в `main.py` стоит **до** `import pandas` намеренно. На Windows
  OpenMP-рантайм, который тянет LightGBM, конфликтует с тем, что загружает
  numpy/scipy, и тот, кто импортируется вторым, получает сломанный пул потоков —
  `LGBM_BoosterPredictForMat` падает с `access violation reading 0x0`.
  Не удаляйте этот импорт как «неиспользуемый».
- `model_full_data.pkl` обучен на 100% train-выборки без внутреннего holdout —
  это «продакшн»-версия модели для реального использования, а не для валидации.
  Оценка качества (ROC-AUC) считалась отдельно через кросс-валидацию.
