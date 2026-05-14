import logging
import asyncio
from datetime import datetime
import pandas as pd
import numpy as np
import ccxt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

# ------------------- НАСТРОЙКИ -------------------
BOT_TOKEN = "ВАШ_ТОКЕН_БОТА"          # замените на свой токен
EXCHANGE = ccxt.binance()             # используем Binance
SYMBOLS = ['BTC/USDT', 'ETH/USDT']
TIMEFRAMES = {
    '1h': '1h',
    '4h': '4h',
    '1d': '1d'
}
DEFAULT_SYMBOL = 'BTC/USDT'
DEFAULT_TF = '4h'
LIMIT = 150  # сколько свечей загружать (для поиска уровней достаточно)

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------- ПОЛУЧЕНИЕ ДАННЫХ -------------------
def fetch_ohlcv(symbol: str, timeframe: str, limit: int = LIMIT) -> pd.DataFrame:
    """Загружает OHLCV с Binance и возвращает DataFrame"""
    try:
        # Приводим символ к формату Binance (BTC/USDT -> BTC/USDT, но ccxt понимает)
        ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame()

# ------------------- ПОИСК УРОВНЕЙ (WSR-подобный) -------------------
def find_swing_points(df: pd.DataFrame, order: int = 5, tolerance: float = 0.005) -> tuple:
    """
    Находит ключевые уровни поддержки и сопротивления.
    order: окно для поиска локальных экстремумов (чем больше, тем меньше уровней)
    tolerance: расстояние между уровнями (в процентах) для объединения близких
    """
    highs = df['high']
    lows = df['low']
    
    # Локальные максимумы (свинг-хай)
    is_max = (highs == highs.rolling(window=2*order+1, center=True).max())
    swing_highs = highs[is_max].values
    
    # Локальные минимумы (свинг-лоу)
    is_min = (lows == lows.rolling(window=2*order+1, center=True).min())
    swing_lows = lows[is_min].values
    
    # Объединяем близкие уровни (чтобы не было 100 линий)
    def cluster_levels(levels, tol):
        if len(levels) == 0:
            return []
        levels = sorted(levels)
        clusters = []
        current_cluster = [levels[0]]
        for lvl in levels[1:]:
            if (lvl - current_cluster[-1]) / current_cluster[-1] < tol:
                current_cluster.append(lvl)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [lvl]
        clusters.append(np.mean(current_cluster))
        return clusters
    
    # Отбираем только значимые (не слишком старые) – последние 5-6 уровней каждого типа
    supports = cluster_levels(swing_lows, tolerance)
    resistances = cluster_levels(swing_highs, tolerance)
    
    # Оставляем наиболее актуальные (близкие к текущей цене)
    current_price = df['close'].iloc[-1]
    supports = [s for s in supports if s < current_price][-5:]      # 5 ближайших снизу
    resistances = [r for r in resistances if r > current_price][:5] # 5 ближайших сверху
    
    return supports, resistances

# ------------------- ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ -------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет RSI и MACD"""
    # RSI (14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (12,26,9)
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    return df

# ------------------- ГЕНЕРАЦИЯ ГРАФИКА (как на скрине) -------------------
def create_chart(df: pd.DataFrame, symbol: str, timeframe: str, supports: list, resistances: list) -> bytes:
    """Создаёт график с свечами, уровнями S/R, RSI и MACD, возвращает PNG в bytes"""
    df = add_indicators(df)
    current_price = df['close'].iloc[-1]
    
    # Подграфики: свечи + уровни, RSI, MACD
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=(f"{symbol} - {timeframe}", "RSI (14)", "MACD"))
    
    # 1) Свечной график
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['open'], high=df['high'],
                                 low=df['low'], close=df['close'],
                                 name='Price'), row=1, col=1)
    
    # Добавляем уровни поддержки и сопротивления
    for s in supports:
        fig.add_hline(y=s, line_dash="dash", line_color="green", opacity=0.8,
                      annotation_text=f"Support {s:.2f}", annotation_position="bottom right",
                      row=1, col=1)
    for r in resistances:
        fig.add_hline(y=r, line_dash="dash", line_color="red", opacity=0.8,
                      annotation_text=f"Resistance {r:.2f}", annotation_position="top right",
                      row=1, col=1)
    
    # Текущая цена (горизонтальная линия)
    fig.add_hline(y=current_price, line_dash="dot", line_color="white", opacity=0.5,
                  annotation_text=f"Current {current_price:.2f}", row=1, col=1)
    
    # 2) RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    
    # 3) MACD
    fig.add_trace(go.Bar(x=df.index, y=df['macd_hist'], name='MACD Hist', marker_color='grey'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='Signal', line=dict(color='orange')), row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    # Общий стиль (тёмная тема, как на скрине)
    fig.update_layout(
        template="plotly_dark",
        title=f"{symbol} | {timeframe} | WSR Pro Levels",
        xaxis_title="Date",
        yaxis_title="Price (USDT)",
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(rangeslider_visible=False)
    
    # Экспорт в PNG (bytes)
    img_bytes = fig.to_image(format="png", engine="kaleido")
    return img_bytes

# ------------------- ФОРМИРОВАНИЕ ТЕКСТОВОГО СИГНАЛА -------------------
def generate_signal(df: pd.DataFrame, supports: list, resistances: list) -> str:
    """На основе RSI и пробоя уровней даёт рекомендацию"""
    current_price = df['close'].iloc[-1]
    rsi = df['rsi'].iloc[-1]
    nearest_support = max([s for s in supports if s < current_price], default=None)
    nearest_resistance = min([r for r in resistances if r > current_price], default=None)
    
    signal = "Нейтрально"
    reason = []
    
    if rsi < 30:
        signal = "ПОКУПКА (oversold)"
        reason.append(f"RSI = {rsi:.1f} (перепроданность)")
    elif rsi > 70:
        signal = "ПРОДАЖА (overbought)"
        reason.append(f"RSI = {rsi:.1f} (перекупленность)")
    else:
        reason.append(f"RSI = {rsi:.1f} (нейтрально)")
    
    if nearest_support and current_price < nearest_support * 1.01:
        reason.append(f"⚠️ Цена близка к поддержке {nearest_support:.2f} (возможен отскок)")
    if nearest_resistance and current_price > nearest_resistance * 0.99:
        reason.append(f"⚠️ Цена близка к сопротивлению {nearest_resistance:.2f} (возможен откат)")
    
    text = f"📊 *{symbol}* | *{timeframe}*\n"
    text += f"💰 Текущая цена: `{current_price:.2f} USDT`\n"
    text += f"📈 Сигнал: *{signal}*\n"
    text += f"📝 Детали: " + ". ".join(reason) + "\n"
    text += f"🟢 Ближайшая поддержка: {nearest_support if nearest_support else '—'}\n"
    text += f"🔴 Ближайшее сопротивление: {nearest_resistance if nearest_resistance else '—'}\n"
    text += f"⏱ Время: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
    return text

# ------------------- КЛАВИАТУРЫ -------------------
def get_main_keyboard(symbol: str, tf: str) -> InlineKeyboardMarkup:
    """Клавиатура для выбора пары и таймфрейма"""
    keyboard = [
        [InlineKeyboardButton(f"📊 {symbol}", callback_data="noop")],
        [InlineKeyboardButton("1h", callback_data=f"tf_1h"), InlineKeyboardButton("4h", callback_data=f"tf_4h"), InlineKeyboardButton("1d", callback_data=f"tf_1d")],
        [InlineKeyboardButton("BTC/USDT", callback_data="sym_BTC/USDT"), InlineKeyboardButton("ETH/USDT", callback_data="sym_ETH/USDT")],
        [InlineKeyboardButton("🔄 Обновить график", callback_data="chart"), InlineKeyboardButton("📈 Текстовый сигнал", callback_data="signal")]
    ]
    return InlineKeyboardMarkup(keyboard)

# ------------------- ОБРАБОТЧИКИ КОМАНД -------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик /start"""
    user = update.effective_user
    await update.message.reply_text(
        f"Привет, {user.first_name}!\n"
        "Я бот WSR Pro. Анализирую поддержки/сопротивления, RSI, MACD.\n\n"
        "Команды:\n"
        "/chart – график с уровнями\n"
        "/signal – текстовый сигнал\n"
        "Или используй кнопки ниже.",
        reply_markup=get_main_keyboard(DEFAULT_SYMBOL, DEFAULT_TF)
    )

async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет график с параметрами по умолчанию"""
    await send_chart(update, context, DEFAULT_SYMBOL, DEFAULT_TF)

async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Отправляет текстовый сигнал по умолчанию"""
    await send_signal(update, context, DEFAULT_SYMBOL, DEFAULT_TF)

async def send_chart(update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str = None, tf: str = None):
    """Генерирует и отправляет график"""
    if symbol is None:
        symbol = context.user_data.get('symbol', DEFAULT_SYMBOL)
    if tf is None:
        tf = context.user_data.get('timeframe', DEFAULT_TF)
    
    # Сохраняем выбор пользователя
    context.user_data['symbol'] = symbol
    context.user_data['timeframe'] = tf
    
    await update.effective_message.reply_text(f"⏳ Загружаю {symbol} ({tf})...")
    
    df = fetch_ohlcv(symbol, tf, LIMIT)
    if df.empty:
        await update.effective_message.reply_text("❌ Ошибка загрузки данных. Попробуйте позже.")
        return
    
    supports, resistances = find_swing_points(df, order=5, tolerance=0.005)
    try:
        img_bytes = create_chart(df, symbol, tf, supports, resistances)
        await update.effective_message.reply_photo(
            photo=img_bytes,
            caption=f"📉 {symbol} | {tf}\nПоддержки: {', '.join(f'{s:.0f}' for s in supports[-3:])}\nСопротивления: {', '.join(f'{r:.0f}' for r in resistances[:3])}",
            reply_markup=get_main_keyboard(symbol, tf)
        )
    except Exception as e:
        logger.error(f"Ошибка генерации графика: {e}")
        await update.effective_message.reply_text("❌ Не удалось создать график. Ошибка в библиотеках.")

async def send_signal(update: Update, context: ContextTypes.DEFAULT_TYPE, symbol: str = None, tf: str = None):
    """Отправляет текстовый торговый сигнал"""
    if symbol is None:
        symbol = context.user_data.get('symbol', DEFAULT_SYMBOL)
    if tf is None:
        tf = context.user_data.get('timeframe', DEFAULT_TF)
    
    context.user_data['symbol'] = symbol
    context.user_data['timeframe'] = tf
    
    await update.effective_message.reply_text(f"🔍 Анализирую {symbol} ({tf})...")
    
    df = fetch_ohlcv(symbol, tf, LIMIT)
    if df.empty:
        await update.effective_message.reply_text("❌ Ошибка загрузки данных.")
        return
    
    supports, resistances = find_swing_points(df, order=5)
    signal_text = generate_signal(df, supports, resistances)
    await update.effective_message.reply_text(signal_text, parse_mode="Markdown", reply_markup=get_main_keyboard(symbol, tf))

# ------------------- ОБРАБОТЧИК INLINE КНОПОК -------------------
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    
    if data.startswith("tf_"):
        tf = data.split("_")[1]  # tf_1h -> 1h
        tf_key = '1h' if tf == '1h' else ('4h' if tf == '4h' else '1d')
        context.user_data['timeframe'] = tf_key
        await query.edit_message_text(f"✅ Таймфрейм изменён на {tf_key}. Используйте кнопки ниже.")
        # Показываем клавиатуру заново
        await query.message.reply_text("Выберите действие:", reply_markup=get_main_keyboard(context.user_data.get('symbol', DEFAULT_SYMBOL), tf_key))
    
    elif data.startswith("sym_"):
        sym = data.split("_")[1]  # sym_BTC/USDT -> BTC/USDT
        context.user_data['symbol'] = sym
        await query.edit_message_text(f"✅ Пара изменена на {sym}. Используйте кнопки ниже.")
        await query.message.reply_text("Выберите действие:", reply_markup=get_main_keyboard(sym, context.user_data.get('timeframe', DEFAULT_TF)))
    
    elif data == "chart":
        sym = context.user_data.get('symbol', DEFAULT_SYMBOL)
        tf = context.user_data.get('timeframe', DEFAULT_TF)
        await send_chart(update, context, sym, tf)
    
    elif data == "signal":
        sym = context.user_data.get('symbol', DEFAULT_SYMBOL)
        tf = context.user_data.get('timeframe', DEFAULT_TF)
        await send_signal(update, context, sym, tf)
    
    else:
        await query.edit_message_text("Команда не распознана.")

# ------------------- ЗАПУСК БОТА -------------------
def main():
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("chart", chart_command))
    app.add_handler(CommandHandler("signal", signal_command))
    app.add_handler(CallbackQueryHandler(button_callback))
    
    logger.info("Бот WSR Pro запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()
