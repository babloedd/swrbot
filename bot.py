"""
SWR Pro Bot — Supply/Weakness/Resistance Telegram Bot
Зелёные зоны = ЛОНГ (поддержка), Красные = ШОРТ (сопротивление)
Кружки на графике = касания зон (точки входа)
"""

import os
import io
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

# ──────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────

BOT_TOKEN: str = os.environ["BOT_TOKEN"]
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

SYMBOLS      = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
DEFAULT_SYM  = "BTC/USDT"
DEFAULT_TF   = "4h"
CANDLE_LIMIT = 200

SWING_ORDER    = 5      # окно поиска экстремумов
ZONE_TOLERANCE = 0.006  # 0.6% — радиус кластеризации
ZONE_HEIGHT    = 0.004  # 0.4% — высота зоны

# ──────────────────────────────────────────────────────
#  LOGGING
# ──────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
logger = logging.getLogger("swr_bot")

# ──────────────────────────────────────────────────────
#  DATA
# ──────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────
#  INDICATORS
# ──────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + avg_g / avg_l.replace(0, np.nan))
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    return df

# ──────────────────────────────────────────────────────
#  SWR ZONE ENGINE
# ──────────────────────────────────────────────────────

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
    hi_idx = _swing_indices(df["high"], SWING_ORDER, "high")
    lo_idx = _swing_indices(df["low"],  SWING_ORDER, "low")

    hi_pts = [(df.index[i], float(df["high"].iloc[i])) for i in hi_idx]
    lo_pts = [(df.index[i], float(df["low"].iloc[i]))  for i in lo_idx]

    current = float(df["close"].iloc[-1])

    res_levels = _cluster([y for _, y in hi_pts if y > current], ZONE_TOLERANCE)[:6]
    sup_levels = _cluster([y for _, y in lo_pts if y < current], ZONE_TOLERANCE)[-6:]

    def make_zone(price: float, pts: list) -> dict:
        half = price * ZONE_HEIGHT / 2
        top  = price + half
        bot  = price - half
        tch  = [{"x": x, "y": y} for x, y in pts if bot * 0.9975 <= y <= top * 1.0025]
        tch.sort(key=lambda d: d["x"])
        return {
            "price":    price,
            "top":      top,
            "bottom":   bot,
            "strength": max(1, len(tch)),
            "x_start":  tch[0]["x"] if tch else df.index[0],
            "touches":  tch,
        }

    sup_zones = [make_zone(p, lo_pts) for p in sup_levels]
    res_zones = [make_zone(p, hi_pts) for p in res_levels]
    return sup_zones, res_zones

# ──────────────────────────────────────────────────────
#  CHART
# ──────────────────────────────────────────────────────

BG   = "#131722"
BG2  = "#1e222d"
GRID = "#2a2e39"
TEXT = "#d1d4dc"
GRN  = "#26a69a"
RED  = "#ef5350"


def _fmt(price: float) -> str:
    return f"{price:,.0f}".replace(",", " ")


def _zone_rect(fig, x0, x1, y0: float, y1: float, fill: str, line: str) -> None:
    fig.add_trace(
        go.Scatter(
            x=[x0, x1, x1, x0, x0],
            y=[y0, y0, y1, y1, y0],
            fill="toself",
            fillcolor=fill,
            line=dict(color=line, width=1),
            mode="lines",
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1, col=1,
    )


def create_chart(df: pd.DataFrame, symbol: str, tf: str,
                 sup_zones: list, res_zones: list) -> bytes:

    df      = add_indicators(df)
    current = float(df["close"].iloc[-1])
    x_end   = df.index[-1]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.62, 0.19, 0.19],
    )

    # ── Зоны поддержки (ЗЕЛЁНЫЕ = ЛОНГ) ──
    for z in sup_zones:
        alpha = min(0.12 + z["strength"] * 0.07, 0.38)
        _zone_rect(
            fig, z["x_start"], x_end, z["bottom"], z["top"],
            fill="rgba(38,166,154,{})".format(round(alpha, 2)),
            line=GRN,
        )
        fig.add_shape(
            type="line",
            x0=z["x_start"], x1=x_end,
            y0=z["price"],   y1=z["price"],
            line=dict(color=GRN, width=1, dash="dot"),
            row=1, col=1,
        )
        fig.add_annotation(
            x=x_end, y=z["price"],
            text="S  {}".format(_fmt(z["price"])),
            font=dict(color=GRN, size=9, family="monospace"),
            showarrow=False, xanchor="left", xshift=5,
            row=1, col=1,
        )
        if z["touches"]:
            fig.add_trace(
                go.Scatter(
                    x=[t["x"] for t in z["touches"]],
                    y=[t["y"] * 0.9991 for t in z["touches"]],
                    mode="markers",
                    marker=dict(
                        symbol="circle", size=10,
                        color=GRN,
                        line=dict(color="#ffffff", width=1),
                    ),
                    hovertemplate="ЛОНГ зона<br>%{x}<extra></extra>",
                    showlegend=False,
                ),
                row=1, col=1,
            )

    # ── Зоны сопротивления (КРАСНЫЕ = ШОРТ) ──
    for z in res_zones:
        alpha = min(0.12 + z["strength"] * 0.07, 0.38)
        _zone_rect(
            fig, z["x_start"], x_end, z["bottom"], z["top"],
            fill="rgba(239,83,80,{})".format(round(alpha, 2)),
            line=RED,
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
        if z["touches"]:
            fig.add_trace(
                go.Scatter(
                    x=[t["x"] for t in z["touches"]],
                    y=[t["y"] * 1.0009 for t in z["touches"]],
                    mode="markers",
                    marker=dict(
                        symbol="circle", size=10,
                        color=RED,
                        line=dict(color="#ffffff", width=1),
                    ),
                    hovertemplate="ШОРТ зона<br>%{x}<extra></extra>",
                    showlegend=False,
                ),
                row=1, col=1,
            )

    # ── Свечи (поверх зон) ──
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

    # Линия текущей цены
    fig.add_hline(
        y=current,
        line=dict(color="rgba(255,255,255,0.35)", width=1, dash="dot"),
        row=1, col=1,
    )

    # ── RSI ──
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df["rsi"],
            name="RSI",
            line=dict(color="#7b61ff", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(123,97,255,0.05)",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=70, line=dict(color=RED, width=1, dash="dash"), opacity=0.5, row=2, col=1)
    fig.add_hline(y=50, line=dict(color="#555", width=1, dash="dot"), opacity=0.4, row=2, col=1)
    fig.add_hline(y=30, line=dict(color=GRN, width=1, dash="dash"), opacity=0.5, row=2, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor=RED, opacity=0.04, row=2, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor=GRN, opacity=0.04, row=2, col=1)

    # ── MACD ──
    bar_colors = [GRN if v >= 0 else RED for v in df["macd_hist"]]
    fig.add_trace(
        go.Bar(x=df.index, y=df["macd_hist"], name="Hist",
               marker_color=bar_colors, opacity=0.75),
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

    # ── Layout ──
    ax = dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color=TEXT, size=9))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG, plot_bgcolor=BG2,
        title=dict(
            text="<b>{}</b>  ·  {}  ·  SWR Pro".format(symbol, tf),
            font=dict(color=TEXT, size=13), x=0.01,
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

# ──────────────────────────────────────────────────────
#  SIGNAL TEXT
# ──────────────────────────────────────────────────────

def generate_signal_text(df: pd.DataFrame, symbol: str, tf: str,
                          sup_zones: list, res_zones: list) -> str:
    current = float(df["close"].iloc[-1])
    rsi     = float(df["rsi"].iloc[-1])
    macd    = float(df["macd"].iloc[-1])
    macd_s  = float(df["macd_signal"].iloc[-1])

    nearest_sup = max((z["price"] for z in sup_zones if z["price"] < current), default=None)
    nearest_res = min((z["price"] for z in res_zones if z["price"] > current), default=None)

    dist_sup = (current - nearest_sup) / current if nearest_sup else 1
    dist_res = (nearest_res - current) / current if nearest_res else 1

    if rsi < 35 and macd > macd_s and dist_sup < 0.012:
        emoji, signal = "🟢", "ЛОНГ — отскок от зоны поддержки"
    elif rsi > 65 and macd < macd_s and dist_res < 0.012:
        emoji, signal = "🔴", "ШОРТ — отклонение от зоны сопротивления"
    elif rsi < 40 and macd > macd_s:
        emoji, signal = "🟢", "ЛОНГ — RSI разворот вверх"
    elif rsi > 60 and macd < macd_s:
        emoji, signal = "🔴", "ШОРТ — RSI разворот вниз"
    else:
        emoji, signal = "⚪", "Нейтрально — ждём зону"

    rsi_bar = "▓" * int(rsi / 10) + "░" * (10 - int(rsi / 10))
    sup_str = _fmt(nearest_sup) if nearest_sup else "—"
    res_str = _fmt(nearest_res) if nearest_res else "—"
    sup_dist = "  _{:.1f}% ниже_".format(dist_sup * 100) if nearest_sup else ""
    res_dist = "  _{:.1f}% выше_".format(dist_res * 100) if nearest_res else ""

    return "\n".join([
        "*{}* · `{}` · SWR Pro".format(symbol, tf),
        "",
        "💰 Цена:  `{} USDT`".format(_fmt(current)),
        "{} *{}*".format(emoji, signal),
        "",
        "📊 RSI-14:  `{:.1f}`  `{}`".format(rsi, rsi_bar),
        "📈 MACD:  `{:+.2f}` / Signal: `{:+.2f}`".format(macd, macd_s),
        "",
        "🟢 Поддержка:  `{}`{}".format(sup_str, sup_dist),
        "🔴 Сопротивление:  `{}`{}".format(res_str, res_dist),
        "",
        "⏱ `{}`".format(datetime.utcnow().strftime("%Y-%m-%d  %H:%M UTC")),
    ])

# ──────────────────────────────────────────────────────
#  KEYBOARDS
# ──────────────────────────────────────────────────────

def kb_main(symbol: str, tf: str) -> InlineKeyboardMarkup:
    def sym_btn(s):
        label = ("✓ " if s == symbol else "") + s.replace("/USDT", "")
        return InlineKeyboardButton(label, callback_data="sym|{}".format(s))

    def tf_btn(t):
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

# ──────────────────────────────────────────────────────
#  HANDLERS
# ──────────────────────────────────────────────────────

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
        "Я — *SWR Pro Bot*.\n"
        "🟢 Зелёные зоны = поддержка → *ЛОНГ*\n"
        "🔴 Красные зоны = сопротивление → *ШОРТ*\n"
        "Кружки = касания зон\n\n"
        "Выбери пару и таймфрейм ⬇️",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=kb_main(sym, tf),
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*SWR Pro Bot*\n\n"
        "/start  — главное меню\n"
        "/chart  — график\n"
        "/signal — текстовый сигнал\n"
        "/help   — эта справка\n\n"
        "🟢 Зелёная зона — поддержка (потенциальный ЛОНГ)\n"
        "🔴 Красная зона — сопротивление (потенциальный ШОРТ)\n"
        "Кружки — касания зон\n"
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
    try:
        img     = create_chart(df, symbol, tf, sup_zones, res_zones)
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
    text = generate_signal_text(df, symbol, tf, sup_zones, res_zones)
    await msg.edit_text(text, parse_mode=ParseMode.MARKDOWN,
                        reply_markup=kb_main(symbol, tf))


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

    sym, tf = _state(ctx)
    try:
        await q.edit_message_reply_markup(reply_markup=kb_main(sym, tf))
    except Exception:
        pass

# ──────────────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────────────

async def post_init(app: Application) -> None:
    await app.bot.set_my_commands([
        BotCommand("start",  "Главное меню"),
        BotCommand("chart",  "График с SWR-зонами"),
        BotCommand("signal", "Торговый сигнал"),
        BotCommand("help",   "Справка"),
    ])


def main() -> None:
    logger.info("Starting SWR Pro Bot...")
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
