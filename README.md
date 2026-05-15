# SWR Pro Bot 📊

Telegram-бот для анализа крипторынка: Supply/Weakness/Resistance зоны, RSI, MACD.  
Деплоится в Railway в один клик.

---

## Что умеет

- 🟥🟩 **SWR-зоны** — прямоугольные области поддержки и сопротивления (как в TradingView)
- 📊 **График** в тёмном стиле: свечи + зоны + RSI + MACD, PNG 1650×1200px
- 📡 **Текстовый сигнал**: LONG / SHORT / Нейтрально с обоснованием
- 🔄 **4 пары**: BTC, ETH, SOL, BNB / 3 таймфрейма: 1h, 4h, 1d
- ⚡ Async-native (ccxt async + python-telegram-bot v21)

---

## Быстрый старт

### 1. Создать бота

Написать [@BotFather](https://t.me/BotFather) → `/newbot` → скопировать токен.

### 2. Деплой в Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app)

1. Форкнуть/загрузить репозиторий на GitHub
2. Зайти на [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**
3. Выбрать репозиторий
4. В разделе **Variables** добавить:

| Key | Value |
|-----|-------|
| `BOT_TOKEN` | токен от BotFather |
| `LOG_LEVEL` | `INFO` (опционально) |

5. Railway сам подхватит `Procfile` и запустит `python bot.py`

> ℹ️ Бот работает как **worker** (не веб-сервис) — порт не нужен.

### 3. Локальный запуск

```bash
git clone <repo>
cd swr_bot

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

export BOT_TOKEN="ваш_токен"
python bot.py
```

---

## Команды бота

| Команда | Описание |
|---------|----------|
| `/start` | Главное меню с кнопками |
| `/chart` | График с SWR-зонами |
| `/signal` | Текстовый торговый сигнал |
| `/help` | Справка |

---

## Архитектура

```
bot.py
├── fetch_ohlcv()        — async загрузка данных с Binance (ccxt)
├── compute_swr_zones()  — поиск свингов → кластеризация → зоны
├── add_indicators()     — RSI-14, MACD 12/26/9
├── create_chart()       — Plotly dark chart → PNG bytes
├── generate_signal_text()— RSI + MACD + зоны → торговый вывод
└── Telegram handlers    — async, inline keyboards, user_data state
```

---

## Как работают SWR-зоны

1. Находим **swing high** и **swing low** через скользящий экстремум (окно ±5 свечей)
2. Объединяем точки с расстоянием < 0.6% в один уровень (кластеризация)
3. Рисуем **прямоугольную зону** ±0.3% от уровня
4. Прозрачность зоны = количество касаний (чем ярче — тем сильнее уровень)

---

## Зависимости

| Пакет | Зачем |
|-------|-------|
| `python-telegram-bot` | Telegram Bot API, async |
| `ccxt` | Биржевые данные (Binance) |
| `pandas` / `numpy` | Обработка OHLCV |
| `plotly` + `kaleido` | Рендер графика в PNG |

---

> ⚠️ Бот предназначен для информационных целей. Не является финансовой рекомендацией.
