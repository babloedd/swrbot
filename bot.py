"""
SWR Pro Bot — Supply/Weakness/Resistance Telegram Bot
Production-ready | Railway-deployable | Async-native
"""

import os
import io
import logging
import asyncio
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import ccxt.async_support as ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    BotCommand,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)
from telegram.constants import ParseMode

# ─────────────────────────────────────────────
#  CONFIG  (всё через ENV — Railway-friendly)
# ─────────────────────────────────────────────

BOT_TOKEN: str = os.environ["BOT_TOKEN"]           # обязательно
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

SYMBOLS: list[str] = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
TIMEFRAMES: dict[str, str] = {"1h": "1h", "4h": "4h", "1d": "1d"}

DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TF = "4h"
CANDLE_LIMIT = 200

SWING_ORDER = 5          # окно поиска свингов
ZONE_TOLERANCE = 0.006   # 0.6% — слияние близких зон
ZONE_HEIGHT = 0.003      # высота зоны = 0.3% от цены

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
logger = logging.getLogger("swr_bot")


# ─────────────────────────────────────────────
#  DATA LAYER
# ─────────────────────────────────────────────

async def fetch_ohlcv(symbol: str, timeframe: str, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    """Асинхронная загрузка OHLCV с Binance."""
    exchange = ccxt.binance({"enableRateLimit": True})
    try:
        raw = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("ts", inplace=True)
        return df
    except Exception as exc:
        logger.error("fetch_ohlcv error [%s %s]: %s", symbol, timeframe, exc)
        return pd.DataFrame()
    finally:
        await exchange.close()


# ─────────────────────────────────────────────
#  SWR ZONE ENGINE
# ─────────────────────────────────────────────

def _find_swing_indices(series: pd.Series, order: int, mode: str) -> np.ndarray:
    """Возвращает индексы свинг-хаев или свинг-лоу."""
    vals = series.values
    n = len(vals)
    result = []
    for i in range(order, n - order):
        window = vals[i - order: i + order + 1]
        center = vals[i]
        if mode == "high" and center == window.max():
            result.append(i)
        elif mode == "low" and center == window.min():
            result.append(i)
    return np.array(result)


def _cluster(levels: list[float], tol: float) -> list[float]:
    """Объединяет уровни, расстояние между которыми < tol (в долях)."""
    if not levels:
        return []
    levels = sorted(set(levels))
    clusters, bucket = [], [levels[0]]
    for lvl in levels[1:]:
        if (lvl - bucket[-1]) / bucket[-1] < tol:
            bucket.append(lvl)
        else:
            clusters.append(float(np.mean(bucket)))
            bucket = [lvl]
    clusters.append(float(np.mean(bucket)))
    return clusters


def compute_swr_zones(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """
    Возвращает зоны поддержки и сопротивления в виде словарей:
      {"price": float, "top": float, "bottom": float, "strength": int}
    strength = сколько раз цена касалась зоны (для прозрачности прямоугольника).
    """
    hi_idx = _find_swing_indices(df["high"], SWING_ORDER, "high")
    lo_idx = _find_swing_indices(df["low"],  SWING_ORDER, "low")

    raw_res = [df["high"].iloc[i] for i in hi_idx]
    raw_sup = [df["low"].iloc[i]  for i in lo_idx]

    current = df["close"].iloc[-1]

    res_levels = _cluster([r for r in raw_res if r > current], ZONE_TOLERANCE)[:6]
    sup_levels = _cluster([s for s in raw_sup if s < current], ZONE_TOLERANCE)[-6:]

    def to_zone(price: float, levels_raw: list[float]) -> dict:
        half = price * ZONE_HEIGHT / 2
        touches = sum(1 for p in levels_raw if abs(p - price) / price < ZONE_TOLERANCE * 2)
        return {
            "price":    price,
            "top":      price + half,
            "bottom":   price - half,
            "strength": max(1, touches),
        }

    sup_zones = [to_zone(p, raw_sup) for p in sup_levels]
    res_zones = [to_zone(p, raw_res) for p in res_levels]

    return sup_zones, res_zones


# ─────────────────────────────────────────────
#  INDICATORS
# ─────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # RSI-14
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + avg_g / avg_l.replace(0, np.nan))

    # MACD (12/26/9)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    return df


# ─────────────────────────────────────────────
#  CHART ENGINE
# ─────────────────────────────────────────────

# Цветовая палитра в стиле TradingView dark
PALETTE = {
    "bg":         "#131722",
    "bg2":        "#1e222d",
    "grid":       "#2a2e39",
    "text":       "#d1d4dc",
    "green":      "#26a69a",
    "red":        "#ef5350",
    "sup_fill":   "rgba(38,166,154,0.15)",
    "sup_line":   "rgba(38,166,154,0.9)",
    "res_fill":   "rgba(239,83,80,0.15)",
    "res_line":   "rgba(239,83,80,0.9)",
    "macd_pos":   "#26a69a",
    "macd_neg":   "#ef5350",
    "rsi_line":   "#7b61ff",
    "price_line": "rgba(255,255,255,0.5)",
}


def create_chart(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    sup_zones: list[dict],
    res_zones: list[dict],
) -> bytes:
    df = add_indicators(df)
    current = df["close"].iloc[-1]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.60, 0.20, 0.20],
    )

    # ── 1. Свечи ──────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"], high=df["high"],
            low=df["low"],   close=df["close"],
            name="Price",
            increasing_fillcolor=PALETTE["green"],
            increasing_line_color=PALETTE["green"],
            decreasing_fillcolor=PALETTE["red"],
            decreasing_line_color=PALETTE["red"],
            line_width=1,
        ),
        row=1, col=1,
    )

    # ── 2. SWR-зоны (прямоугольники) ──────────
    x_start = df.index[max(0, len(df) - 80)]   # показываем зону с последних ~80 свечей
    x_end   = df.index[-1]

    for z in sup_zones:
        opacity = min(0.12 + z["strength"] * 0.05, 0.35)
        fig.add_shape(
            type="rect",
            x0=x_start, x1=x_end,
            y0=z["bottom"], y1=z["top"],
            fillcolor=PALETTE["sup_fill"].replace("0.15", str(opacity)),
            line=dict(color=PALETTE["sup_line"], width=1),
            row=1, col=1,
        )
        fig.add_annotation(
            x=x_end, y=z["price"],
            text=f"S  {z['price']:,.0f}",
            font=dict(color=PALETTE["sup_line"], size=10, family="JetBrains Mono, monospace"),
            showarrow=False, xanchor="left",
            row=1, col=1,
        )

    for z in res_zones:
        opacity = min(0.12 + z["strength"] * 0.05, 0.35)
        fig.add_shape(
            type="rect",
            x0=x_start, x1=x_end,
            y0=z["bottom"], y1=z["top"],
            fillcolor=PALETTE["res_fill"].replace("0.15", str(opacity)),
            line=dict(color=PALETTE["res_line"], width=1),
            row=1, col=1,
        )
        fig.add_annotation(
            x=x_end, y=z["price"],
            text=f"R  {z['price']:,.0f}",
            font=dict(color=PALETTE["res_line"], size=10, family="JetBrains Mono, monospace"),
            showarrow=False, xanchor="left",
            row=1, col=1,
        )

    # Линия текущей цены
    fig.add_hline(
        y=current,
        line=dict(color=PALETTE["price_line"], width=1, dash="dot"),
        row=1, col=1,
    )

    # ── 3. RSI ────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["rsi"],
            name="RSI", line=dict(color=PALETTE["rsi_line"], width=1.5),
        ),
        row=2, col=1,
    )
    for level, color in [(70, PALETTE["red"]), (30, PALETTE["green"])]:
        fig.add_hline(y=level, line=dict(color=color, width=1, dash="dash"),
                      opacity=0.5, row=2, col=1)
    # Зона перекупленности / перепроданности
    fig.add_hrect(y0=70, y1=100, fillcolor=PALETTE["red"],   opacity=0.04, row=2, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor=PALETTE["green"], opacity=0.04, row=2, col=1)

    # ── 4. MACD ───────────────────────────────
    hist_colors = [
        PALETTE["macd_pos"] if v >= 0 else PALETTE["macd_neg"]
        for v in df["macd_hist"]
    ]
    fig.add_trace(
        go.Bar(
            x=df.index, y=df["macd_hist"],
            name="Hist", marker_color=hist_colors, opacity=0.7,
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["macd"],
                   name="MACD", line=dict(color="#2962ff", width=1.5)),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["macd_signal"],
                   name="Signal", line=dict(color="#ff6d00", width=1.5)),
        row=3, col=1,
    )

    # ── Layout ────────────────────────────────
    axis_style = dict(
        gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"],
        tickfont=dict(color=PALETTE["text"], size=9),
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["bg2"],
        title=dict(
            text=f"<b>{symbol}</b>  ·  {timeframe}  ·  SWR Pro",
            font=dict(color=PALETTE["text"], size=14),
            x=0.02,
        ),
        height=800, width=1100,
        margin=dict(l=10, r=120, t=50, b=10),
        showlegend=False,
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(**axis_style)
    fig.update_yaxes(**axis_style)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    return fig.to_image(format="png", engine="kaleido", scale=1.5)


# ─────────────────────────────────────────────
#  SIGNAL LOGIC
# ─────────────────────────────────────────────

def generate_signal_text(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    sup_zones: list[dict],
    res_zones: list[dict],
) -> str:
    current = df["close"].iloc[-1]
    rsi     = df["rsi"].iloc[-1]
    macd    = df["macd"].iloc[-1]
    macd_s  = df["macd_signal"].iloc[-1]

    nearest_sup = max((z["price"] for z in sup_zones if z["price"] < current), default=None)
    nearest_res = min((z["price"] for z in res_zones if z["price"] > current), default=None)

    # Основной сигнал
    if rsi < 32 and macd > macd_s:
        signal_emoji, signal_text = "🟢", "LONG / Покупка"
    elif rsi > 68 and macd < macd_s:
        signal_emoji, signal_text = "🔴", "SHORT / Продажа"
    elif rsi < 45 and nearest_sup and (current - nearest_sup) / current < 0.008:
        signal_emoji, signal_text = "🟡", "Возможный отскок от поддержки"
    elif rsi > 55 and nearest_res and (nearest_res - current) / current < 0.008:
        signal_emoji, signal_text = "🟡", "Возможное отклонение от сопротивления"
    else:
        signal_emoji, signal_text = "⚪", "Нейтрально — ждём подтверждения"

    rsi_bar = "▓" * int(rsi / 10) + "░" * (10 - int(rsi / 10))

    lines = [
        f"*{symbol}* · `{timeframe}` · SWR Pro",
        "",
        f"💰 Цена:  `{current:,.2f} USDT`",
        f"{signal_emoji} Сигнал:  *{signal_text}*",
        "",
        f"📊 RSI-14:  `{rsi:.1f}`  `{rsi_bar}`",
        f"📈 MACD:  `{macd:+.2f}`  /  Signal: `{macd_s:+.2f}`",
        "",
        f"🟢 Поддержка:  `{nearest_sup:,.2f}`" if nearest_sup else "🟢 Поддержка:  —",
        f"🔴 Сопротивление:  `{nearest_res:,.2f}`" if nearest_res else "🔴 Сопротивление:  —",
        "",
        f"⏱ `{datetime.utcnow().strftime('%Y-%m-%d  %H:%M UTC')}`",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────
#  KEYBOARDS
# ─────────────────────────────────────────────

def kb_main(symbol: str, tf: str) -> InlineKeyboardMarkup:
    sym_short = symbol.replace("/USDT", "")
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("1h",  callback_data="tf|1h"),
            InlineKeyboardButton("4h",  callback_data="tf|4h"),
            InlineKeyboardButton("1d",  callback_data="tf|1d"),
        ],
        [
            InlineKeyboardButton(f"{'✓ ' if symbol == s else ''}{ s.replace('/USDT','')}", callback_data=f"sym|{s}")
            for s in SYMBOLS
        ],
        [
            InlineKeyboardButton("📊 График",      callback_data="action|chart"),
            InlineKeyboardButton("📡 Сигнал",      callback_data="action|signal"),
        ],
    ])


# ─────────────────────────────────────────────
#  HANDLERS
# ─────────────────────────────────────────────

def get_user_state(context: ContextTypes.DEFAULT_TYPE) -> tuple[str, str]:
    symbol = context.user_data.get("symbol", DEFAULT_SYMBOL)
    tf     = context.user_data.get("tf", DEFAULT_TF)
    return symbol, tf


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    name = update.effective_user.first_name or "Трейдер"
    symbol, tf = get_user_state(context)
    await update.message.reply_text(
        f"👋 *Привет, {name}!*\n\n"
        "Я — *SWR Pro Bot*. Рисую зоны поддержки и сопротивления,\n"
        "считаю RSI + MACD и даю торговые сигналы.\n\n"
        "Выбери пару, таймфрейм и жми кнопки ⬇️",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_main(symbol, tf),
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*SWR Pro Bot — команды:*\n\n"
        "/start  — главное меню\n"
        "/chart  — график текущей пары\n"
        "/signal — текстовый сигнал\n"
        "/help   — эта справка\n\n"
        "Зоны на графике:\n"
        "🔴 красные — сопротивление (supply)\n"
        "🟢 зелёные — поддержка (demand)\n"
        "Чем ярче зона — тем больше касаний.",
        parse_mode=ParseMode.MARKDOWN,
    )


async def _do_chart(update: Update, context: ContextTypes.DEFAULT_TYPE,
                    symbol: str, tf: str) -> None:
    msg = await update.effective_message.reply_text(
        f"⏳ Загружаю `{symbol}` · `{tf}`...",
        parse_mode=ParseMode.MARKDOWN,
    )
    df = await fetch_ohlcv(symbol, tf)
    if df.empty:
        await msg.edit_text("❌ Не удалось загрузить данные. Попробуйте позже.")
        return

    df = add_indicators(df)
    sup_zones, res_zones = compute_swr_zones(df)

    try:
        img = create_chart(df, symbol, tf, sup_zones, res_zones)
        current = df["close"].iloc[-1]
        sup_str = ", ".join(f"`{z['price']:,.0f}`" for z in sup_zones[-3:]) or "—"
        res_str = ", ".join(f"`{z['price']:,.0f}`" for z in res_zones[:3]) or "—"
        caption = (
            f"*{symbol}* · `{tf}`\n"
            f"Цена: `{current:,.2f}` USDT\n"
            f"Поддержки: {sup_str}\n"
            f"Сопротивления: {res_str}"
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
        await msg.edit_text("❌ Ошибка генерации графика. Проверьте логи.")


async def _do_signal(update: Update, context: ContextTypes.DEFAULT_TYPE,
                     symbol: str, tf: str) -> None:
    msg = await update.effective_message.reply_text(
        f"🔍 Анализирую `{symbol}` · `{tf}`...",
        parse_mode=ParseMode.MARKDOWN,
    )
    df = await fetch_ohlcv(symbol, tf)
    if df.empty:
        await msg.edit_text("❌ Не удалось загрузить данные.")
        return

    df = add_indicators(df)
    sup_zones, res_zones = compute_swr_zones(df)
    text = generate_signal_text(df, symbol, tf, sup_zones, res_zones)

    await msg.edit_text(text, parse_mode=ParseMode.MARKDOWN,
                        reply_markup=kb_main(symbol, tf))


async def cmd_chart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol, tf = get_user_state(context)
    await _do_chart(update, context, symbol, tf)


async def cmd_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    symbol, tf = get_user_state(context)
    await _do_signal(update, context, symbol, tf)


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    kind, value = query.data.split("|", 1)

    if kind == "tf":
        context.user_data["tf"] = value
        symbol, tf = get_user_state(context)
        await query.edit_message_reply_markup(reply_markup=kb_main(symbol, tf))

    elif kind == "sym":
        context.user_data["symbol"] = value
        symbol, tf = get_user_state(context)
        await query.edit_message_reply_markup(reply_markup=kb_main(symbol, tf))

    elif kind == "action":
        symbol, tf = get_user_state(context)
        if value == "chart":
            await _do_chart(update, context, symbol, tf)
        elif value == "signal":
            await _do_signal(update, context, symbol, tf)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

async def post_init(app: Application) -> None:
    await app.bot.set_my_commands([
        BotCommand("start",  "Главное меню"),
        BotCommand("chart",  "График с SWR-зонами"),
        BotCommand("signal", "Торговый сигнал"),
        BotCommand("help",   "Справка"),
    ])
    logger.info("Bot commands registered.")


def main() -> None:
    logger.info("Starting SWR Pro Bot…")
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("chart",  cmd_chart))
    app.add_handler(CommandHandler("signal", cmd_signal))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CallbackQueryHandler(on_button))

    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
