"""
SWR Pro Bot — Supply / Weakness / Resistance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Зоны:
  🔴 Красная (сопротивление) — потенциальный ШОРТ
  🟢 Зелёная (поддержка)    — потенциальный ЛОНГ

Маркеры:
  ▼ красный треугольник   — первое касание красной зоны → сигнал ШОРТ
  ▲ зелёный треугольник   — первое касание зелёной зоны → сигнал ЛОНГ
  ● красный кружок        — повторное касание красной зоны (подтверждение)
  ● зелёный кружок        — повторное касание зелёной зоны (подтверждение)
"""

import os
import io
import asyncio
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import ccxt.async_support as ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes,
)
from telegram.constants import ParseMode

# ══════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════

BOT_TOKEN: str = os.environ["BOT_TOKEN"]
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

SYMBOLS      = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
DEFAULT_SYM  = "BTC/USDT"
DEFAULT_TF   = "4h"
CANDLE_LIMIT = 200

SWING_ORDER    = 5      # окно поиска свинг-экстремумов (свечей с каждой стороны)
ZONE_TOLERANCE = 0.006  # 0.6% — радиус кластеризации уровней
ZONE_HEIGHT    = 0.004  # 0.4% — полувысота зоны от центра

# ══════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
logger = logging.getLogger("swr_bot")

# ══════════════════════════════════════════════════════
#  DATA LAYER
# ══════════════════════════════════════════════════════

async def fetch_ohlcv(symbol: str, timeframe: str, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    exchange = ccxt.binance({"enableRateLimit": True})
    try:
        raw = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        return df
    except Exception as exc:
        logger.error("fetch_ohlcv [%s %s]: %s", symbol, timeframe, exc)
        return pd.DataFrame()
    finally:
        await exchange.close()

# ══════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # RSI-14
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + avg_g / avg_l.replace(0, np.nan))
    # MACD 12/26/9
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    # EMA200 — trend filter
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()
    return df


def get_trend(df: pd.DataFrame) -> str:
    """
    'bull' — цена выше EMA200 → только лонги
    'bear' — цена ниже EMA200 → только шорты
    'neutral' — EMA ещё не устоялась (мало данных)
    """
    current = float(df["close"].iloc[-1])
    ema200  = float(df["ema200"].iloc[-1])
    if df["ema200"].isna().sum() > 50:
        return "neutral"
    if current > ema200 * 1.001:
        return "bull"
    if current < ema200 * 0.999:
        return "bear"
    return "neutral"

# ══════════════════════════════════════════════════════
#  SWR ZONE ENGINE
# ══════════════════════════════════════════════════════

def _swing_indices(series: pd.Series, order: int, mode: str) -> list:
    vals = series.values
    n    = len(vals)
    out  = []
    for i in range(order, n - order):
        w = vals[i - order: i + order + 1]
        if mode == "high" and vals[i] == w.max():
            out.append(i)
        elif mode == "low" and vals[i] == w.min():
            out.append(i)
    return out


def _cluster(vals: list, tol: float) -> list:
    if not vals:
        return []
    vals = sorted(set(vals))
    clusters, bucket = [], [vals[0]]
    for v in vals[1:]:
        if (v - bucket[-1]) / bucket[-1] < tol:
            bucket.append(v)
        else:
            clusters.append(float(np.mean(bucket)))
            bucket = [v]
    clusters.append(float(np.mean(bucket)))
    return clusters


def compute_swr_zones(df: pd.DataFrame):
    """
    Возвращает (sup_zones, res_zones).

    Логика сигналов основана на ЗАКРЫТИИ СВЕЧИ (rejection):

    ШОРТ-сигнал (красная зона):
      - Свеча зашла внутрь зоны (high >= bottom)
      - Свеча ЗАКРЫЛАСЬ НИЖЕ зоны (close < bottom)
      → цена отклонилась, продавец держит
      → треугольник ▼ = первый такой rejection
      → кружок ● = каждый следующий rejection

    ЛОНГ-сигнал (зелёная зона):
      - Свеча зашла внутрь зоны (low <= top)
      - Свеча ЗАКРЫЛАСЬ ВЫШЕ зоны (close > top)
      → цена отклонилась, покупатель держит
      → треугольник ▲ = первый такой rejection
      → кружок ● = каждый следующий rejection

    Зона формируется по свинг-экстремумам.
    Ширина зоны = ATR * 0.5 (адаптивная, не фиксированная).
    """
    hi_idx = _swing_indices(df["high"], SWING_ORDER, "high")
    lo_idx = _swing_indices(df["low"],  SWING_ORDER, "low")

    hi_pts = [(df.index[i], float(df["high"].iloc[i])) for i in hi_idx]
    lo_pts = [(df.index[i], float(df["low"].iloc[i]))  for i in lo_idx]

    current = float(df["close"].iloc[-1])

    # Адаптивная ширина зоны через ATR-14
    atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
    half_zone = atr * 0.5

    res_levels = _cluster([y for _, y in hi_pts if y > current], ZONE_TOLERANCE)[:6]
    sup_levels = _cluster([y for _, y in lo_pts if y < current], ZONE_TOLERANCE)[-6:]

    def find_rejections(price: float, zone_top: float, zone_bot: float, side: str) -> list:
        """
        Проходим по всем свечам и ищем rejection:
        - side='res': свеча зашла в зону (high >= zone_bot), закрылась ниже (close < zone_bot)
        - side='sup': свеча зашла в зону (low <= zone_top), закрылась выше (close > zone_top)
        Возвращает список {"x", "y"} отсортированных по времени.
        """
        events = []
        for i in range(len(df)):
            row   = df.iloc[i]
            ts    = df.index[i]
            hi    = float(row["high"])
            lo    = float(row["low"])
            cl    = float(row["close"])
            op    = float(row["open"])

            if side == "res":
                touched  = hi >= zone_bot          # зашли в зону снизу
                rejected = cl < zone_bot           # закрылись обратно ниже
                bearish  = cl < op                 # свеча медвежья
                if touched and rejected and bearish:
                    events.append({"x": ts, "y": hi})

            elif side == "sup":
                touched  = lo <= zone_top          # зашли в зону сверху
                rejected = cl > zone_top           # закрылись обратно выше
                bullish  = cl > op                 # свеча бычья
                if touched and rejected and bullish:
                    events.append({"x": ts, "y": lo})

        return events

    def make_zone(price: float, side: str) -> dict:
        top = price + half_zone
        bot = price - half_zone
        events = find_rejections(price, top, bot, side)

        # Если rejection-свечей нет — ищем хотя бы простые касания (для отображения зоны)
        if not events:
            if side == "res":
                fallback = [(df.index[i], float(df["high"].iloc[i]))
                            for i in hi_idx if bot <= df["high"].iloc[i] <= top * 1.003]
            else:
                fallback = [(df.index[i], float(df["low"].iloc[i]))
                            for i in lo_idx if bot * 0.997 <= df["low"].iloc[i] <= top]
            fallback.sort(key=lambda t: t[0])
            events = [{"x": x, "y": y} for x, y in fallback]

        signal   = events[0]   if events          else None
        confirms = events[1:]  if len(events) > 1 else []

        return {
            "price":    price,
            "top":      top,
            "bottom":   bot,
            "strength": max(1, len(events)),
            "x_start":  events[0]["x"] if events else df.index[0],
            "signal":   signal,
            "confirms": confirms,
        }

    sup_zones = [make_zone(p, "sup") for p in sup_levels]
    res_zones = [make_zone(p, "res") for p in res_levels]
    return sup_zones, res_zones

# ══════════════════════════════════════════════════════
#  LIQUIDITY LEVELS
# ══════════════════════════════════════════════════════

def compute_liquidity_levels(df: pd.DataFrame) -> dict:
    """
    Ищет скопления ликвидности — уровни где стоят стопы толпы.

    Логика:
    - Equal highs (buy liquidity):  серия свинг-хаев в пределах ATR*0.3
      → над ними стоят стопы шортистов. Цена идёт туда, снимает их.
    - Equal lows  (sell liquidity): серия свинг-лоу в пределах ATR*0.3
      → под ними стоят стопы лонгистов.

    Swept levels — уровни которые уже были пробиты и цена вернулась.
    Это самые сильные зоны разворота (sweep + rejection).
    """
    atr = float((df["high"] - df["low"]).rolling(14).mean().iloc[-1])
    tol = atr * 0.3  # допуск для "равных" уровней

    hi_idx = _swing_indices(df["high"], 3, "high")  # более чувствительный order=3
    lo_idx = _swing_indices(df["low"],  3, "low")

    hi_vals = [float(df["high"].iloc[i]) for i in hi_idx]
    lo_vals = [float(df["low"].iloc[i])  for i in lo_idx]

    def find_equal_levels(vals: list, min_count: int = 2) -> list:
        """Группирует близкие уровни — минимум min_count касаний."""
        if not vals:
            return []
        vals = sorted(vals)
        groups, bucket = [], [vals[0]]
        for v in vals[1:]:
            if v - bucket[0] <= tol:
                bucket.append(v)
            else:
                if len(bucket) >= min_count:
                    groups.append({
                        "price": float(np.mean(bucket)),
                        "count": len(bucket),
                    })
                bucket = [v]
        if len(bucket) >= min_count:
            groups.append({"price": float(np.mean(bucket)), "count": len(bucket)})
        return groups

    buy_liq  = find_equal_levels(hi_vals)   # равные хаи → стопы шортистов выше
    sell_liq = find_equal_levels(lo_vals)   # равные лои → стопы лонгистов ниже

    current = float(df["close"].iloc[-1])

    # Помечаем swept уровни — цена уже была там и вернулась
    def is_swept(price: float, side: str) -> bool:
        """Пробой уровня с возвратом — признак sweep."""
        for i in range(1, len(df) - 1):
            if side == "high":
                if float(df["high"].iloc[i]) > price and float(df["close"].iloc[i]) < price:
                    return True
            else:
                if float(df["low"].iloc[i]) < price and float(df["close"].iloc[i]) > price:
                    return True
        return False

    for lvl in buy_liq:
        lvl["swept"] = is_swept(lvl["price"], "high")
    for lvl in sell_liq:
        lvl["swept"] = is_swept(lvl["price"], "low")

    return {
        "buy_liq":  [l for l in buy_liq  if l["price"] > current],   # выше цены
        "sell_liq": [l for l in sell_liq if l["price"] < current],   # ниже цены
    }

# ══════════════════════════════════════════════════════
#  CHART ENGINE
# ══════════════════════════════════════════════════════

BG   = "#131722"
BG2  = "#1e222d"
GRID = "#2a2e39"
TEXT = "#d1d4dc"
GRN  = "#26a69a"
RED  = "#ef5350"


def _fmt(price: float) -> str:
    """80412 → '80 412'"""
    return "{:,.0f}".format(price).replace(",", " ")


def _add_zone_fill(fig, x0, x1, y_bot: float, y_top: float,
                   fill_rgba: str, line_color: str) -> None:
    """Заливочный прямоугольник через Scatter (надёжнее add_shape для временной оси)."""
    fig.add_trace(
        go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[y_bot, y_bot, y_top, y_top, y_bot],
            fill="toself",
            fillcolor=fill_rgba,
            line=dict(color=line_color, width=1),
            mode="lines",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=1,
    )


def create_chart(df: pd.DataFrame, symbol: str, tf: str,
                 sup_zones: list, res_zones: list,
                 liq_levels: dict = None) -> bytes:

    df      = add_indicators(df)
    current = float(df["close"].iloc[-1])
    trend   = get_trend(df)
    x_end   = df.index[-1]
    if liq_levels is None:
        liq_levels = {"buy_liq": [], "sell_liq": []}

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.62, 0.19, 0.19],
    )

    # ── 1. ЗОНЫ ПОДДЕРЖКИ (зелёные, ЛОНГ) ─────────────
    for z in sup_zones:
        alpha = min(0.13 + z["strength"] * 0.07, 0.40)

        # Прямоугольник от первого касания до правого края
        _add_zone_fill(
            fig,
            x0=z["x_start"], x1=x_end,
            y_bot=z["bottom"], y_top=z["top"],
            fill_rgba="rgba(38,166,154,{:.2f})".format(alpha),
            line_color=GRN,
        )
        # Пунктирная линия по центру
        fig.add_shape(
            type="line",
            x0=z["x_start"], x1=x_end,
            y0=z["price"],   y1=z["price"],
            line=dict(color=GRN, width=1, dash="dot"),
            row=1, col=1,
        )
        # Подпись справа
        fig.add_annotation(
            x=x_end, y=z["price"],
            text="S  {}".format(_fmt(z["price"])),
            font=dict(color=GRN, size=9, family="monospace"),
            showarrow=False, xanchor="left", xshift=5,
            row=1, col=1,
        )

        # ▲ Зелёный треугольник — ПЕРВОЕ касание (сигнал ЛОНГ)
        if z["signal"]:
            fig.add_trace(
                go.Scatter(
                    x=[z["signal"]["x"]],
                    y=[z["signal"]["y"] * 0.9988],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=13,
                        color=GRN,
                        line=dict(color="#ffffff", width=1),
                    ),
                    hovertemplate="ЛОНГ сигнал<br>%{x}<extra></extra>",
                    showlegend=False,
                ),
                row=1, col=1,
            )

        # ● Зелёные кружки — ПОСЛЕДУЮЩИЕ касания (подтверждения)
        if z["confirms"]:
            fig.add_trace(
                go.Scatter(
                    x=[c["x"] for c in z["confirms"]],
                    y=[c["y"] * 0.9992 for c in z["confirms"]],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=9,
                        color=GRN,
                        line=dict(color="#ffffff", width=0.8),
                    ),
                    hovertemplate="ЛОНГ подтверждение<br>%{x}<extra></extra>",
                    showlegend=False,
                ),
                row=1, col=1,
            )

    # ── 2. ЗОНЫ СОПРОТИВЛЕНИЯ (красные, ШОРТ) ──────────
    for z in res_zones:
        alpha = min(0.13 + z["strength"] * 0.07, 0.40)

        _add_zone_fill(
            fig,
            x0=z["x_start"], x1=x_end,
            y_bot=z["bottom"], y_top=z["top"],
            fill_rgba="rgba(239,83,80,{:.2f})".format(alpha),
            line_color=RED,
        )
        fig.add_shape(
            type="line",
            x0=z["x_start"], x1=x_end,
            y0=z["price"],   y1=z["price"],
            line=dict(color=RED, width=1, dash="dot"),
            row=1, col=1,
        )
        fig.add_annotation(
            x=x_end, y=z["price"],
            text="R  {}".format(_fmt(z["price"])),
            font=dict(color=RED, size=9, family="monospace"),
            showarrow=False, xanchor="left", xshift=5,
            row=1, col=1,
        )

        # ▼ Красный треугольник — ПЕРВОЕ касание (сигнал ШОРТ)
        if z["signal"]:
            fig.add_trace(
                go.Scatter(
                    x=[z["signal"]["x"]],
                    y=[z["signal"]["y"] * 1.0012],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=13,
                        color=RED,
                        line=dict(color="#ffffff", width=1),
                    ),
                    hovertemplate="ШОРТ сигнал<br>%{x}<extra></extra>",
                    showlegend=False,
                ),
                row=1, col=1,
            )

        # ● Красные кружки — ПОСЛЕДУЮЩИЕ касания (подтверждения)
        if z["confirms"]:
            fig.add_trace(
                go.Scatter(
                    x=[c["x"] for c in z["confirms"]],
                    y=[c["y"] * 1.0008 for c in z["confirms"]],
                    mode="markers",
                    marker=dict(
                        symbol="circle",
                        size=9,
                        color=RED,
                        line=dict(color="#ffffff", width=0.8),
                    ),
                    hovertemplate="ШОРТ подтверждение<br>%{x}<extra></extra>",
                    showlegend=False,
                ),
                row=1, col=1,
            )

    # ── 3. СВЕЧИ (поверх зон) ───────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            name="Price",
            increasing_fillcolor=GRN, increasing_line_color=GRN,
            decreasing_fillcolor=RED, decreasing_line_color=RED,
            line_width=1,
        ),
        row=1, col=1,
    )

    # Пунктир текущей цены
    fig.add_hline(
        y=current,
        line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
        row=1, col=1,
    )

    # ── 3b. EMA200 ─────────────────────────────────────
    TREND_COLOR = {"bull": "rgba(38,166,154,0.7)", "bear": "rgba(239,83,80,0.7)", "neutral": "rgba(150,150,150,0.5)"}
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["ema200"],
            name="EMA200",
            line=dict(color=TREND_COLOR.get(trend, "rgba(150,150,150,0.5)"), width=1.5, dash="dash"),
            showlegend=False,
            hovertemplate="EMA200: %{y:,.0f}<extra></extra>",
        ),
        row=1, col=1,
    )
    # Подпись EMA200
    fig.add_annotation(
        x=x_end, y=float(df["ema200"].iloc[-1]),
        text="EMA200",
        font=dict(color=TREND_COLOR.get(trend, "rgba(150,150,150,0.5)"), size=9, family="monospace"),
        showarrow=False, xanchor="left", xshift=5,
        row=1, col=1,
    )

    # ── 3c. LIQUIDITY LEVELS ─────────────────────────────
    # Buy liquidity (равные хаи, стопы шортистов) — пунктир над ценой
    for lvl in liq_levels.get("buy_liq", []):
        line_color = "rgba(255,213,79,0.9)" if lvl["swept"] else "rgba(255,213,79,0.5)"
        dash_style = "solid" if lvl["swept"] else "dot"
        fig.add_hline(
            y=lvl["price"],
            line=dict(color=line_color, width=1, dash=dash_style),
            row=1, col=1,
        )
        label = "LIQ ×{} {}".format(lvl["count"], "✓swept" if lvl["swept"] else "")
        fig.add_annotation(
            x=df.index[int(len(df)*0.02)], y=lvl["price"],
            text=label,
            font=dict(color="rgba(255,213,79,0.8)", size=8, family="monospace"),
            showarrow=False, xanchor="left", yshift=6,
            row=1, col=1,
        )

    # Sell liquidity (равные лои, стопы лонгистов) — пунктир под ценой
    for lvl in liq_levels.get("sell_liq", []):
        line_color = "rgba(255,213,79,0.9)" if lvl["swept"] else "rgba(255,213,79,0.5)"
        dash_style = "solid" if lvl["swept"] else "dot"
        fig.add_hline(
            y=lvl["price"],
            line=dict(color=line_color, width=1, dash=dash_style),
            row=1, col=1,
        )
        label = "LIQ ×{} {}".format(lvl["count"], "✓swept" if lvl["swept"] else "")
        fig.add_annotation(
            x=df.index[int(len(df)*0.02)], y=lvl["price"],
            text=label,
            font=dict(color="rgba(255,213,79,0.8)", size=8, family="monospace"),
            showarrow=False, xanchor="left", yshift=-10,
            row=1, col=1,
        )

    # ── 4. RSI ──────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["rsi"],
            name="RSI",
            line=dict(color="#7b61ff", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(123,97,255,0.06)",
        ),
        row=2, col=1,
    )
    for lvl, col in [(70, RED), (50, "#444"), (30, GRN)]:
        fig.add_hline(
            y=lvl,
            line=dict(color=col, width=1, dash="dash"),
            opacity=0.5, row=2, col=1,
        )
    fig.add_hrect(y0=70, y1=100, fillcolor=RED, opacity=0.04, row=2, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor=GRN, opacity=0.04, row=2, col=1)

    # ── 5. MACD ─────────────────────────────────────────
    bar_colors = [GRN if v >= 0 else RED for v in df["macd_hist"]]
    fig.add_trace(
        go.Bar(x=df.index, y=df["macd_hist"],
               marker_color=bar_colors, opacity=0.75, showlegend=False),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["macd"],
                   line=dict(color="#2962ff", width=1.5), showlegend=False),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["macd_signal"],
                   line=dict(color="#ff6d00", width=1.5), showlegend=False),
        row=3, col=1,
    )

    # ── LAYOUT ──────────────────────────────────────────
    ax = dict(
        gridcolor=GRID,
        zerolinecolor=GRID,
        tickfont=dict(color=TEXT, size=9),
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG2,
        title=dict(
            text="<b>{}</b>  ·  {}  ·  SWR Pro".format(symbol, tf),
            font=dict(color=TEXT, size=13),
            x=0.01,
        ),
        height=840, width=1120,
        margin=dict(l=10, r=115, t=45, b=10),
        showlegend=False,
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(**ax)
    fig.update_yaxes(**ax)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    return fig.to_image(format="png", engine="kaleido", scale=1.5)

# ══════════════════════════════════════════════════════
#  SIGNAL TEXT
# ══════════════════════════════════════════════════════

def generate_signal_text(df: pd.DataFrame, symbol: str, tf: str,
                          sup_zones: list, res_zones: list,
                          liq_levels: dict = None) -> str:
    current = float(df["close"].iloc[-1])
    rsi     = float(df["rsi"].iloc[-1])
    macd    = float(df["macd"].iloc[-1])
    macd_s  = float(df["macd_signal"].iloc[-1])
    trend   = get_trend(df)
    if liq_levels is None:
        liq_levels = {"buy_liq": [], "sell_liq": []}

    nearest_sup = max((z["price"] for z in sup_zones if z["price"] < current), default=None)
    nearest_res = min((z["price"] for z in res_zones if z["price"] > current), default=None)
    dist_sup = (current - nearest_sup) / current if nearest_sup else 1.0
    dist_res = (nearest_res - current) / current if nearest_res else 1.0

    # Ближайшая ликвидность выше и ниже
    near_buy_liq  = min((l["price"] for l in liq_levels["buy_liq"]),  default=None)
    near_sell_liq = max((l["price"] for l in liq_levels["sell_liq"]), default=None)
    swept_above = any(l["swept"] for l in liq_levels["buy_liq"]  if l["price"] < current * 1.02)
    swept_below = any(l["swept"] for l in liq_levels["sell_liq"] if l["price"] > current * 0.98)

    # ── Логика сигнала с EMA200 trend filter ──
    # В аптренде — только лонги, в даунтренде — только шорты
    long_allowed  = trend in ("bull", "neutral")
    short_allowed = trend in ("bear", "neutral")

    if dist_sup < 0.010 and rsi < 45 and macd > macd_s and long_allowed:
        if swept_below:
            emoji, signal = "🟢", "ЛОНГ  —  зона поддержки + sweep ликвидности снизу"
        else:
            emoji, signal = "🟢", "ЛОНГ  —  цена у зоны поддержки, тренд подтверждает"
    elif dist_res < 0.010 and rsi > 55 and macd < macd_s and short_allowed:
        if swept_above:
            emoji, signal = "🔴", "ШОРТ  —  зона сопротивления + sweep ликвидности сверху"
        else:
            emoji, signal = "🔴", "ШОРТ  —  цена у зоны сопротивления, тренд подтверждает"
    elif dist_sup < 0.018 and rsi < 40 and long_allowed:
        emoji, signal = "🟡", "Близко к поддержке  —  следи за разворотом"
    elif dist_res < 0.018 and rsi > 60 and short_allowed:
        emoji, signal = "🟡", "Близко к сопротивлению  —  следи за разворотом"
    elif not long_allowed and dist_sup < 0.010:
        emoji, signal = "⚠️", "Зона поддержки, но тренд медвежий — лонг против тренда"
    elif not short_allowed and dist_res < 0.010:
        emoji, signal = "⚠️", "Зона сопротивления, но тренд бычий — шорт против тренда"
    else:
        emoji, signal = "⚪", "Нейтрально  —  цена между зонами"

    # Тренд-строка
    trend_str = {"bull": "🟢 Бычий (выше EMA200)", "bear": "🔴 Медвежий (ниже EMA200)", "neutral": "⚪ Нейтральный"}.get(trend, "⚪")

    rsi_bar   = "▓" * int(rsi / 10) + "░" * (10 - int(rsi / 10))
    sup_str   = _fmt(nearest_sup)   if nearest_sup   else "—"
    res_str   = _fmt(nearest_res)   if nearest_res   else "—"
    sup_dist  = "  _{:.1f}% ниже_".format(dist_sup * 100) if nearest_sup else ""
    res_dist  = "  _{:.1f}% выше_".format(dist_res * 100) if nearest_res else ""
    liq_above = "  `{}`".format(_fmt(near_buy_liq))  if near_buy_liq  else ""
    liq_below = "  `{}`".format(_fmt(near_sell_liq)) if near_sell_liq else ""

    return "\n".join([
        "*{}* · `{}` · SWR Pro".format(symbol, tf),
        "",
        "💰 Цена:  `{} USDT`".format(_fmt(current)),
        "{} *{}*".format(emoji, signal),
        "",
        "📈 Тренд:  {}".format(trend_str),
        "📊 RSI-14:  `{:.1f}`  `{}`".format(rsi, rsi_bar),
        "📉 MACD:  `{:+.2f}`  /  Signal: `{:+.2f}`".format(macd, macd_s),
        "",
        "🟢 Поддержка:{}{}".format("  `{}`".format(sup_str), sup_dist),
        "🔴 Сопротивление:{}{}".format("  `{}`".format(res_str), res_dist),
        "💛 Ликвидность выше:{}".format(liq_above if liq_above else "  —"),
        "💛 Ликвидность ниже:{}".format(liq_below if liq_below else "  —"),
        "",
        "⏱ `{}`".format(datetime.utcnow().strftime("%Y-%m-%d  %H:%M UTC")),
    ])

# ══════════════════════════════════════════════════════
#  KEYBOARDS
# ══════════════════════════════════════════════════════

def kb_main(symbol: str, tf: str) -> InlineKeyboardMarkup:
    def sym_btn(s: str) -> InlineKeyboardButton:
        label = ("✓ " if s == symbol else "") + s.replace("/USDT", "")
        return InlineKeyboardButton(label, callback_data="sym|{}".format(s))

    def tf_btn(t: str) -> InlineKeyboardButton:
        label = ("✓ " if t == tf else "") + t
        return InlineKeyboardButton(label, callback_data="tf|{}".format(t))

    return InlineKeyboardMarkup([
        [tf_btn("1h"), tf_btn("4h"), tf_btn("1d")],
        [sym_btn(s) for s in SYMBOLS],
        [
            InlineKeyboardButton("📊 График",  callback_data="action|chart"),
            InlineKeyboardButton("📡 Сигнал",  callback_data="action|signal"),
        ],
    ])

# ══════════════════════════════════════════════════════
#  HANDLERS
# ══════════════════════════════════════════════════════

def _state(ctx: ContextTypes.DEFAULT_TYPE):
    return (
        ctx.user_data.get("symbol", DEFAULT_SYM),
        ctx.user_data.get("tf",     DEFAULT_TF),
    )


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    name = update.effective_user.first_name or "Трейдер"
    sym, tf = _state(ctx)
    await update.message.reply_text(
        "*Привет, {}!*\n\n".format(name) +
        "Я — *SWR Pro Bot*\n\n"
        "🟢 Зелёная зона = поддержка → *ЛОНГ*\n"
        "🔴 Красная зона = сопротивление → *ШОРТ*\n\n"
        "▲▼ Треугольник = сигнал входа\n"
        "●  Кружок = подтверждение зоны\n\n"
        "Выбери пару и таймфрейм ⬇️",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_main(sym, tf),
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*SWR Pro Bot — справка*\n\n"
        "/start  — главное меню\n"
        "/chart  — график с зонами\n"
        "/signal — текстовый анализ\n"
        "/help   — эта справка\n\n"
        "*Как читать график:*\n"
        "🟢 Зелёная зона — поддержка (ЛОНГ)\n"
        "🔴 Красная зона — сопротивление (ШОРТ)\n"
        "▲ Зелёный треугольник — сигнал ЛОНГ\n"
        "▼ Красный треугольник — сигнал ШОРТ\n"
        "● Кружок — подтверждение (зона держит)\n"
        "Чем ярче зона — тем сильнее уровень",
        parse_mode=ParseMode.MARKDOWN,
    )


async def _do_chart(update: Update, ctx: ContextTypes.DEFAULT_TYPE,
                    symbol: str, tf: str) -> None:
    msg = await update.effective_message.reply_text(
        "⏳ Загружаю `{}` · `{}`...".format(symbol, tf),
        parse_mode=ParseMode.MARKDOWN,
    )
    df = await fetch_ohlcv(symbol, tf)
    if df.empty:
        await msg.edit_text("❌ Не удалось загрузить данные.")
        return

    df = add_indicators(df)
    sup_zones, res_zones = compute_swr_zones(df)
    liq_levels = compute_liquidity_levels(df)

    try:
        img     = create_chart(df, symbol, tf, sup_zones, res_zones, liq_levels)
        current = float(df["close"].iloc[-1])
        sup_str = "  ".join(_fmt(z["price"]) for z in sup_zones[-3:]) or "—"
        res_str = "  ".join(_fmt(z["price"]) for z in res_zones[:3])  or "—"
        caption = (
            "*{}* · `{}`\n".format(symbol, tf) +
            "Цена: `{}` USDT\n".format(_fmt(current)) +
            "🟢 Поддержки: `{}`\n".format(sup_str) +
            "🔴 Сопротивления: `{}`".format(res_str)
        )
        await msg.delete()
        await update.effective_message.reply_photo(
            photo=io.BytesIO(img),
            caption=caption,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=kb_main(symbol, tf),
        )
    except Exception as exc:
        logger.exception("chart error: %s", exc)
        await msg.edit_text("❌ Ошибка генерации графика.")


async def _do_signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE,
                     symbol: str, tf: str) -> None:
    msg = await update.effective_message.reply_text(
        "🔍 Анализирую `{}` · `{}`...".format(symbol, tf),
        parse_mode=ParseMode.MARKDOWN,
    )
    df = await fetch_ohlcv(symbol, tf)
    if df.empty:
        await msg.edit_text("❌ Не удалось загрузить данные.")
        return

    df = add_indicators(df)
    sup_zones, res_zones = compute_swr_zones(df)
    liq_levels = compute_liquidity_levels(df)
    text = generate_signal_text(df, symbol, tf, sup_zones, res_zones, liq_levels)
    await msg.edit_text(text, parse_mode=ParseMode.MARKDOWN,
                        reply_markup=kb_main(symbol, tf))


async def cmd_alerts(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    current = ctx.user_data.get("alerts_on", True)
    ctx.user_data["alerts_on"] = not current
    if ctx.user_data["alerts_on"]:
        lines_text = [
            "🔔 *Уведомления включены*",
            "",
            "Буду писать когда появится новый треугольник:",
            "▲ ЛОНГ — rejection от зоны поддержки",
            "▼ ШОРТ — rejection от зоны сопротивления",
            "",
            "Мониторю: BTC, ETH, SOL, BNB",
            "Таймфреймы: 1h, 4h, 1d",
            "",
            "_Отправь /alerts ещё раз чтобы выключить._",
        ]
        text = "\n".join(lines_text)
    else:
        _sent_alerts.pop(update.effective_user.id, None)
        text = "\n".join(["🔕 *Уведомления выключены*", "", "_Отправь /alerts чтобы включить снова._"])
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def cmd_chart(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _do_chart(update, ctx, *_state(ctx))


async def cmd_signal(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await _do_signal(update, ctx, *_state(ctx))


async def on_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    await q.answer()
    kind, val = q.data.split("|", 1)

    if kind == "tf":
        ctx.user_data["tf"] = val
    elif kind == "sym":
        ctx.user_data["symbol"] = val
    elif kind == "action":
        sym, tf = _state(ctx)
        if val == "chart":
            await _do_chart(update, ctx, sym, tf)
        elif val == "signal":
            await _do_signal(update, ctx, sym, tf)
        return

    # Обновляем клавиатуру после смены пары/таймфрейма
    sym, tf = _state(ctx)
    try:
        await q.edit_message_reply_markup(reply_markup=kb_main(sym, tf))
    except Exception:
        pass

# ══════════════════════════════════════════════════════
#  ALERT ENGINE  (фоновый мониторинг)
# ══════════════════════════════════════════════════════

# Таймфреймы для мониторинга и их интервал проверки в секундах
ALERT_SCHEDULE: dict = {
    "1h": 900,    # каждые 15 мин
    "4h": 1800,   # каждые 30 мин
    "1d": 3600,   # каждый час
}

# Пары для мониторинга — те же что в боте
ALERT_SYMBOLS = SYMBOLS

# sent_alerts: {user_id: set("BTC/USDT|4h|long|80000", ...)}
# Ключ = пара|таймфрейм|сторона|округлённая цена зоны
# Храним в памяти — при рестарте сбрасывается (нормально)
_sent_alerts: dict = {}


def _alert_key(symbol: str, tf: str, side: str, zone_price: float) -> str:
    """Уникальный ключ сигнала чтобы не дублировать уведомления."""
    rounded = round(zone_price, -2)   # округляем до 100 чтобы не дёргаться
    return "{}|{}|{}|{}".format(symbol, tf, side, rounded)


def _check_fresh_signal(df: pd.DataFrame, sup_zones: list, res_zones: list) -> list:
    """
    Проверяет последнюю ЗАКРЫТУЮ свечу (iloc[-2], не текущую незакрытую).
    Возвращает список найденных сигналов: [{"side", "zone_price", "candle_price"}]
    """
    if len(df) < 3:
        return []

    # Берём предпоследнюю свечу — она точно закрыта
    row   = df.iloc[-2]
    hi    = float(row["high"])
    lo    = float(row["low"])
    cl    = float(row["close"])
    op    = float(row["open"])

    signals = []

    # Проверяем rejection от зон сопротивления (ШОРТ)
    for z in res_zones:
        touched  = hi >= z["bottom"]
        rejected = cl < z["bottom"]
        bearish  = cl < op
        if touched and rejected and bearish:
            signals.append({
                "side":        "short",
                "zone_price":  z["price"],
                "candle_price": hi,
                "zone_top":    z["top"],
                "zone_bot":    z["bottom"],
            })

    # Проверяем rejection от зон поддержки (ЛОНГ)
    for z in sup_zones:
        touched  = lo <= z["top"]
        rejected = cl > z["top"]
        bullish  = cl > op
        if touched and rejected and bullish:
            signals.append({
                "side":        "long",
                "zone_price":  z["price"],
                "candle_price": lo,
                "zone_top":    z["top"],
                "zone_bot":    z["bottom"],
            })

    return signals


def _format_alert(symbol, tf, sig, df, liq_levels):
    trend   = get_trend(df)
    current = float(df["close"].iloc[-1])
    rsi     = float(df["rsi"].iloc[-1])

    if sig["side"] == "short":
        emoji, action, marker = "🔴", "ШОРТ", "▼"
        trend_warn = "\n⚠️ _Против тренда (цена выше EMA200)_" if trend == "bull" else ""
    else:
        emoji, action, marker = "🟢", "ЛОНГ", "▲"
        trend_warn = "\n⚠️ _Против тренда (цена ниже EMA200)_" if trend == "bear" else ""

    trend_str = {"bull": "🟢 Бычий", "bear": "🔴 Медвежий", "neutral": "⚪ Нейтральный"}.get(trend, "⚪")

    liq_boost = ""
    if sig["side"] == "long":
        swept = any(l["swept"] and l["price"] > sig["zone_bot"] * 0.995
                    for l in liq_levels.get("sell_liq", []))
        if swept:
            liq_boost = "\n💛 _Рядом снята ликвидность — сигнал усилен_"
    else:
        swept = any(l["swept"] and l["price"] < sig["zone_top"] * 1.005
                    for l in liq_levels.get("buy_liq", []))
        if swept:
            liq_boost = "\n💛 _Рядом снята ликвидность — сигнал усилен_"

    parts = [
        "{} {} *{} сигнал* {}".format(emoji, marker, action, marker),
        "",
        "*{}*  ·  `{}`".format(symbol, tf),
        "Цена:  `{} USDT`".format(_fmt(current)),
        "Зона:  `{} — {}`".format(_fmt(sig["zone_bot"]), _fmt(sig["zone_top"])),
        "",
        "📈 Тренд:  {}".format(trend_str),
        "📊 RSI:  `{:.1f}`".format(rsi),
        "{}{}".format(trend_warn, liq_boost),
        "",
        "⏱ `{}`".format(datetime.utcnow().strftime("%Y-%m-%d  %H:%M UTC")),
    ]
    return "\n".join(parts)


async def _monitor_loop(app) -> None:
    """
    Фоновый цикл мониторинга.
    Проверяет все пары × таймфреймы по расписанию ALERT_SCHEDULE.
    Шлёт уведомления всем пользователям у которых включены алерты.
    """
    logger.info("Alert monitor started.")
    last_check: dict = {}   # {tf: timestamp последней проверки}

    while True:
        await asyncio.sleep(60)   # проверяем расписание каждую минуту
        now = datetime.utcnow().timestamp()

        for tf, interval in ALERT_SCHEDULE.items():
            if now - last_check.get(tf, 0) < interval:
                continue
            last_check[tf] = now

            for symbol in ALERT_SYMBOLS:
                try:
                    df = await fetch_ohlcv(symbol, tf, limit=CANDLE_LIMIT)
                    if df.empty or len(df) < 50:
                        continue

                    df = add_indicators(df)
                    sup_zones, res_zones = compute_swr_zones(df)
                    liq_levels = compute_liquidity_levels(df)

                    fresh = _check_fresh_signal(df, sup_zones, res_zones)
                    if not fresh:
                        continue

                    # Рассылаем всем подписанным пользователям
                    for user_id, user_data in app.user_data.items():
                        if not user_data.get("alerts_on", True):
                            continue

                        sent = _sent_alerts.setdefault(user_id, set())

                        for sig in fresh:
                            key = _alert_key(symbol, tf, sig["side"], sig["zone_price"])
                            if key in sent:
                                continue   # уже отправляли этот сигнал

                            text = _format_alert(symbol, tf, sig, df, liq_levels)
                            try:
                                await app.bot.send_message(
                                    chat_id=user_id,
                                    text=text,
                                    parse_mode=ParseMode.MARKDOWN,
                                )
                                sent.add(key)
                                logger.info("Alert sent → user %s | %s %s %s",
                                            user_id, symbol, tf, sig["side"])
                            except Exception as send_err:
                                logger.warning("Alert send error user %s: %s", user_id, send_err)

                except Exception as exc:
                    logger.error("Monitor error [%s %s]: %s", symbol, tf, exc)


# ══════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════

async def post_init(app: Application) -> None:
    """Регистрирует команды и запускает фоновый мониторинг."""
    await app.bot.set_my_commands([
        BotCommand("start",   "Главное меню"),
        BotCommand("chart",   "График с SWR-зонами"),
        BotCommand("signal",  "Торговый сигнал"),
        BotCommand("alerts",  "Вкл/выкл уведомления о сигналах"),
        BotCommand("help",    "Справка"),
    ])
    logger.info("Commands registered.")
    # Запускаем фоновый мониторинг в том же event loop
    asyncio.create_task(_monitor_loop(app))
    logger.info("Alert monitor task created.")


def main() -> None:
    logger.info("Starting SWR Pro Bot...")
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )
    app.add_handler(CommandHandler("start",   cmd_start))
    app.add_handler(CommandHandler("chart",   cmd_chart))
    app.add_handler(CommandHandler("signal",  cmd_signal))
    app.add_handler(CommandHandler("alerts",  cmd_alerts))
    app.add_handler(CommandHandler("help",    cmd_help))
    app.add_handler(CallbackQueryHandler(on_button))
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
