import os
import re
import sys
import time
import json
import random
import asyncio
import requests
import logging
import unicodedata
from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.errors import FloodWaitError
from telethon.tl.functions.messages import SetTypingRequest
from telethon.tl.types import SendMessageTypingAction
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from phrases import get_random_phrase

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_ID       = int(os.getenv("API_ID"))
API_HASH     = os.getenv("API_HASH")
SESSION_NAME = os.getenv("SESSION_NAME", "userbot")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL")
MODEL        = os.getenv("MODEL", "auto")
TEMPERATURE  = float(os.getenv("TEMPERATURE", "0.7"))
MAX_HISTORY  = int(os.getenv("MAX_HISTORY", "20"))
GROUP_CONTEXT_SIZE = int(os.getenv("GROUP_CONTEXT_SIZE", "20"))
SYSTEM_PROMPT     = os.getenv("SYSTEM_PROMPT", "Ты — полезный ассистент.")
SYSTEM_PROMPT_RAG = os.getenv("SYSTEM_PROMPT_RAG", SYSTEM_PROMPT)
TRIGGER_WORD = os.getenv("TRIGGER_WORD", "лиса")
_TRIGGER_RE  = re.compile(r"\bлис[ауыеойиёцяь]|\bлис\b|\bрыб", re.IGNORECASE)
TRIGGER_TAGS = {"@foxlike_creature", "@foxllke_creature"}
USER_FACTS   = os.getenv("USER_FACTS", "")

IMMEDIATE_CHANCE = 0.2
MIN_DELAY    = 180
MAX_DELAY    = 600
ONLINE_WINDOW = 600
IDLE_BASE    = 10800
IDLE_RANDOM_MAX = 600
SPLIT_CHANCE = 0.20

_LOCATION_KEYWORDS = {
    "где", "живёшь", "живешь", "живу", "находишься", "страна", "город",
    "откуда", "вьетнам", "вьетнаме", "россия", "россию", "рф", "location",
    "место", "переехал", "переехала",
}

client = TelegramClient(os.path.join(BASE_DIR, SESSION_NAME), API_ID, API_HASH)
MY_ID: int | None = None

chat_histories:    dict[int, list[dict]] = {}
message_counters:  dict[int, int] = {}
online_mode_until: dict[int, float] = {}  # chat_id -> timestamp
chat_message_log:  dict[int, list[bool]] = {}
followup_user_id:  dict[int, int] = {}
followup_expires:  dict[int, float] = {}
ladder_bullets:    dict[int, list[str]] = {}
ladder_user_id:    dict[int, int] = {}
ladder_message_id: dict[int, int] = {}
ladder_counter:    dict[int, int] = {}
messages_since_reply: dict[int, int] = {}
pending_tasks:     dict[int, list[asyncio.Task]] = {}
idle_timers:       dict[int, asyncio.Task] = {}

_peer_cache:  dict[int, object] = {}
_user_names:  dict[int, str] = {}
group_message_buffer: dict[int, list[dict]] = {}


def _get_sender_name(event) -> str:
    sid = event.sender_id
    if sid is None:
        return "Аноним"
    if sid not in _user_names:
        sender = event.sender
        if sender:
            # Channel objects have 'title', User objects have 'first_name'
            name = (
                getattr(sender, "first_name", None)
                or getattr(sender, "title", None)
                or getattr(sender, "username", None)
                or str(sid)
            )
            _user_names[sid] = name
    return _user_names.get(sid, str(sid))


def _append_group_ctx(chat_id: int, name: str, text: str) -> None:
    buf = group_message_buffer.setdefault(chat_id, [])
    buf.append({"name": name, "text": text})
    if len(buf) > GROUP_CONTEXT_SIZE:
        del buf[:-GROUP_CONTEXT_SIZE]


def _format_group_ctx(chat_id: int, exclude_text: str) -> str:
    buf = group_message_buffer.get(chat_id, [])
    lines = [f"[{m['name']}]: {m['text']}" for m in buf if m["text"] != exclude_text]
    return "\n".join(lines) if lines else ""


rag_index:      dict | None = None
_rag_matrix     = None
_rag_vectorizer = None
RAG_TOP_K          = 16
RAG_MIN_SCORE      = 0.15
RAG_CONTEXT_MAX_LEN = 5000


def resolve_model() -> None:
    global MODEL
    if MODEL and MODEL != "auto":
        return
    try:
        resp = requests.get(f"{LM_STUDIO_URL}/v1/models", timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if not models:
            logger.error("LM Studio не вернул ни одной модели")
            sys.exit(1)
        MODEL = models[0]["id"]
        logger.info(f"Автовыбор модели: {MODEL}")
    except Exception as e:
        logger.error(f"Не удалось получить модель из LM Studio: {e}")
        sys.exit(1)


def load_rag_index() -> None:
    global rag_index, _rag_matrix, _rag_vectorizer
    index_path = os.path.join(BASE_DIR, "rag_index.json")
    if not os.path.exists(index_path):
        logger.warning(f"RAG-индекс не найден: {index_path}")
        return
    with open(index_path, "r", encoding="utf-8") as f:
        rag_index = json.load(f)
    td = rag_index["tfidf"]
    _rag_matrix = csr_matrix(
        (td["data"], td["indices"], td["indptr"]), shape=td["shape"]
    )
    _rag_vectorizer = TfidfVectorizer(vocabulary=rag_index["vocabulary"])
    _rag_vectorizer.idf_ = np.array(rag_index["idf"])
    count = rag_index.get("pair_count", rag_index.get("chunk_count", 0))
    logger.info(f"RAG загружен: {count} записей, матрица {_rag_matrix.shape}")


def enrich_with_rag(query: str) -> str:
    if not rag_index or _rag_matrix is None or _rag_vectorizer is None:
        return ""
    nq = unicodedata.normalize("NFKC", query).lower().strip()
    qv = _rag_vectorizer.transform([nq])
    sims = cosine_similarity(qv, _rag_matrix).flatten()
    top = sims.argsort()[-RAG_TOP_K:][::-1]
    version = rag_index.get("version", 1)
    parts, seen = [], set()
    for idx in top:
        if sims[idx] < RAG_MIN_SCORE:
            continue
        if version >= 2:
            pair = rag_index["pairs"][idx]
            text = f"[{pair['query']}] → {pair['response']}"
        else:
            text = rag_index["chunks"][idx]["text"]
        key = text[:60].lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(text)
    if not parts:
        return ""
    ctx = "\n---\n".join(parts)
    return ctx[:RAG_CONTEXT_MAX_LEN] + ("..." if len(ctx) > RAG_CONTEXT_MAX_LEN else "")


def personal_fact_hint(query: str) -> str:
    if not USER_FACTS:
        return ""
    if set(query.lower().split()) & _LOCATION_KEYWORDS:
        return USER_FACTS
    return ""


def get_chat_history(chat_id: int) -> list[dict]:
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    return chat_histories[chat_id]


def get_message_counter(chat_id: int) -> int:
    return message_counters.get(chat_id, 0)


def increment_counter(chat_id: int) -> int:
    message_counters[chat_id] = message_counters.get(chat_id, 0) + 1
    return message_counters[chat_id]


def query_lm_studio(chat_id: int, user_message: str) -> str:
    history = get_chat_history(chat_id)
    rag_query = user_message
    if len(user_message.split()) < 6 and history:
        last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
        if last_user:
            rag_query = f"{last_user} {user_message}"
    rag_context = enrich_with_rag(rag_query)
    system_content = SYSTEM_PROMPT_RAG if rag_context else SYSTEM_PROMPT
    if rag_context:
        system_content += (
            "\n\n---\nПримеры твоих ответов в похожих ситуациях — ориентируйся на эту лексику и тон:\n\n"
            + rag_context + "\n---"
        )
    fact = personal_fact_hint(user_message)
    if fact:
        system_content += f"\n\n[факт о себе: {fact}]"
    group_ctx = _format_group_ctx(chat_id, user_message)
    if group_ctx:
        system_content += (
            "\n\n--- Последние сообщения в чате ---\n"
            + group_ctx
            + "\n---"
        )
    messages = [{"role": "system", "content": system_content}]
    messages.extend(history[-MAX_HISTORY:])
    messages.append({"role": "user", "content": user_message})
    try:
        resp = requests.post(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            json={"model": MODEL, "messages": messages, "temperature": TEMPERATURE},
            timeout=120,
        )
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"]
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": reply})
        if len(history) > MAX_HISTORY * 2:
            history[:] = history[-MAX_HISTORY * 2:]
        return reply
    except Exception as e:
        logger.warning(f"Ошибка LM Studio для чата {chat_id}: {e}")
        return ""


def _query_split_reply(chat_id: int, first_reply: str) -> str:
    """Короткое продолжение первого ответа строго по теме разговора."""
    history = get_chat_history(chat_id)
    last_user_msg = next(
        (m["content"] for m in reversed(history) if m["role"] == "user"), ""
    )
    continuation_prompt = (
        f"Ты только что написала: «{first_reply}»\n"
        f"Собеседник говорил о: «{last_user_msg[:120]}»\n\n"
        "Если хочется добавить одну короткую мысль строго по этой теме — напиши её (3–8 слов). "
        "Если добавить нечего по существу — ответь только: -"
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": continuation_prompt})
    try:
        resp = requests.post(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            json={"model": MODEL, "messages": messages, "temperature": 0.4},
            timeout=60,
        )
        resp.raise_for_status()
        second = resp.json()["choices"][0]["message"]["content"].strip()
        if not second or second in ("-", "–", "—") or len(second) < 3:
            return ""
        if history and history[-1]["role"] == "assistant":
            history[-1]["content"] = first_reply + "\n" + second
        return second
    except Exception as e:
        logger.warning(f"Ошибка split-reply для чата {chat_id}: {e}")
        return ""


def _query_idle_message(chat_id: int) -> str:
    """Генерирует спонтанное сообщение через LLM с учётом последних сообщений чата."""
    buf = group_message_buffer.get(chat_id, [])
    ctx_lines = [f"[{m['name']}]: {m['text']}" for m in buf[-5:]]
    ctx = "\n".join(ctx_lines)
    idle_prompt = (
        "Посмотри на последние сообщения в чате. "
        "Напиши одну короткую мысль — своё мнение, реакцию или наблюдение. "
        "Не обращайся ни к кому напрямую, не отвечай на конкретный вопрос. "
        "Пиши как будто вспомнила что-то или захотела поделиться. "
        "5–15 слов, без точки в конце."
    )
    if ctx:
        idle_prompt = f"Последние сообщения в чате:\n{ctx}\n\n{idle_prompt}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": idle_prompt},
    ]
    try:
        resp = requests.post(
            f"{LM_STUDIO_URL}/v1/chat/completions",
            json={"model": MODEL, "messages": messages, "temperature": TEMPERATURE},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Ошибка генерации idle-сообщения: {e}")
        return get_random_phrase()


def online_chance(chat_id: int) -> float:
    n = messages_since_reply.get(chat_id, 99)
    if n <= 1: return 0.50
    if n == 2: return 0.10
    return 0.05


def estimate_typing_time(text: str) -> float:
    return max(1.5, len(text) / 390 * 60)


def calculate_group_delay(trigger_type: str, chat_id: int) -> float:
    if trigger_type in ("tag", "question", "followup"):
        return 0.0
    if trigger_type == "random_online":
        return 0.0
    if online_mode_until.get(chat_id, 0) > time.time():
        return 0.0
    if random.random() < IMMEDIATE_CHANCE:
        return 0.0
    return random.uniform(MIN_DELAY, MAX_DELAY)


def cancel_pending_tasks(chat_id: int) -> None:
    for task in pending_tasks.get(chat_id, []):
        task.cancel()
    pending_tasks.pop(chat_id, None)


def cancel_idle_timer(chat_id: int) -> None:
    if chat_id in idle_timers:
        idle_timers[chat_id].cancel()
        idle_timers.pop(chat_id, None)


def schedule_idle_message(chat_id: int) -> None:
    cancel_idle_timer(chat_id)
    delay = IDLE_BASE + random.uniform(0, IDLE_RANDOM_MAX)

    async def _task() -> None:
        await asyncio.sleep(delay)
        await send_idle_message(chat_id)

    idle_timers[chat_id] = asyncio.create_task(_task())


async def send_idle_message(chat_id: int) -> None:
    log = chat_message_log.get(chat_id, [])
    if log and log[-1] is True:
        logger.info(f"Idle пропущено — последнее уже от меня, ждём ещё {IDLE_BASE}с")
        schedule_idle_message(chat_id)
        return
    phrase = await asyncio.to_thread(_query_idle_message, chat_id)
    logger.info(f"Idle-сообщение в чат {chat_id}: {phrase[:60]}")
    peer = _peer_cache.get(chat_id, chat_id)
    try:
        await client.send_message(peer, phrase)
        online_mode_until[chat_id] = time.time() + ONLINE_WINDOW
        if chat_id not in chat_message_log:
            chat_message_log[chat_id] = []
        chat_message_log[chat_id].append(True)
        if len(chat_message_log[chat_id]) > 5:
            chat_message_log[chat_id] = chat_message_log[chat_id][-5:]
    except FloodWaitError as e:
        logger.warning(f"FloodWait {e.seconds}с при idle для чата {chat_id}")
        await asyncio.sleep(e.seconds)
    except Exception as e:
        logger.error(f"Ошибка idle для чата {chat_id}: {e}")
    schedule_idle_message(chat_id)


async def ladder_wait(chat_id: int, messages: list[str]) -> list[str]:
    last_len = len(messages)
    for _ in range(6):
        if chat_id not in ladder_user_id:
            break
        await asyncio.sleep(10)
        current_len = len(ladder_bullets.get(chat_id, []))
        if current_len > last_len:
            last_len = current_len
            messages = list(ladder_bullets[chat_id])
        else:
            break
    return messages


async def keep_typing(chat_id: int, duration: float, interval: float = 4.0) -> None:
    if chat_id not in _peer_cache:
        try:
            _peer_cache[chat_id] = await client.get_input_entity(chat_id)
        except Exception as e:
            logger.warning(f"Не удалось получить peer для typing {chat_id}: {e}")
            return
    peer = _peer_cache[chat_id]
    elapsed = 0.0
    while elapsed < duration:
        try:
            await client(SetTypingRequest(peer=peer, action=SendMessageTypingAction()))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Typing send error {chat_id}: {e}")
        sleep_time = min(interval, duration - elapsed)
        await asyncio.sleep(sleep_time)
        elapsed += sleep_time


def is_followup(sender_id: int, chat_id: int) -> bool:
    if chat_id not in followup_user_id:
        return False
    if time.time() > followup_expires.get(chat_id, 0):
        followup_user_id.pop(chat_id, None)
        followup_expires.pop(chat_id, None)
        return False
    return sender_id == followup_user_id[chat_id]


async def process_message(event, user_text: str, trigger_type: str, counter_snapshot: int) -> None:
    chat_id  = event.chat_id
    is_group = not event.is_private
    task = asyncio.current_task()
    if chat_id not in pending_tasks:
        pending_tasks[chat_id] = []
    pending_tasks[chat_id].append(task)

    try:
        delay = calculate_group_delay(trigger_type, chat_id) if is_group else 0.0
        if delay > 0:
            logger.info(f"Задержка {delay:.1f}с для чата {chat_id}")
            await asyncio.sleep(delay)

        if trigger_type == "random_online" and online_mode_until.get(chat_id, 0) <= time.time():
            logger.info(f"random_online отменён — онлайн-режим истёк для чата {chat_id}")
            return

        if is_group and trigger_type == "followup":
            ladder_bullets[chat_id] = [user_text]
            ladder_message_id[chat_id] = event.message.id
            ladder_counter[chat_id] = 1

        if is_group and trigger_type == "followup" and chat_id in ladder_bullets:
            ladder_user_id[chat_id] = event.sender_id
            ladder_bullets[chat_id] = await ladder_wait(chat_id, list(ladder_bullets[chat_id]))
            ladder_user_id.pop(chat_id, None)

        if is_group:
            final_text   = "\n".join(ladder_bullets.get(chat_id, [user_text]))
            final_msg_id = ladder_message_id.get(chat_id, event.message.id)
            if ladder_counter.get(chat_id, 0) > 1:
                logger.info(f"Лесенка: {ladder_counter[chat_id]} сообщений склеены")
            ladder_bullets.pop(chat_id, None)
            ladder_message_id.pop(chat_id, None)
            ladder_counter.pop(chat_id, None)
        else:
            final_text   = user_text
            final_msg_id = event.message.id

        await asyncio.sleep(random.uniform(2.5, 4.5))

        model_task  = asyncio.create_task(asyncio.to_thread(query_lm_studio, chat_id, final_text))
        typing_task = asyncio.create_task(keep_typing(chat_id, 300))
        try:
            reply = await model_task
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass

        if not reply:
            logger.info(f"Пустой ответ для чата {chat_id}")
            return

        await asyncio.sleep(estimate_typing_time(reply))

        peer = _peer_cache.get(chat_id, chat_id)
        chunks = [reply[i:i + 4000] for i in range(0, len(reply), 4000)]
        for chunk in chunks:
            has_new = get_message_counter(chat_id) > counter_snapshot
            try:
                if is_group and has_new:
                    await client.send_message(peer, chunk, reply_to=final_msg_id)
                else:
                    await client.send_message(peer, chunk)
            except FloodWaitError as e:
                logger.warning(f"FloodWait {e.seconds}с — ждём")
                await asyncio.sleep(e.seconds)
                await client.send_message(peer, chunk)

        if trigger_type != "random_online":
            online_mode_until[chat_id] = time.time() + ONLINE_WINDOW

        # Split-сообщение с шансом 30%
        if random.random() < SPLIT_CHANCE:
            second = await asyncio.to_thread(_query_split_reply, chat_id, reply)
            if second:
                await asyncio.sleep(random.uniform(1.0, 3.0))
                typing2 = asyncio.create_task(keep_typing(chat_id, 30))
                await asyncio.sleep(estimate_typing_time(second))
                typing2.cancel()
                try:
                    await typing2
                except asyncio.CancelledError:
                    pass
                try:
                    await client.send_message(peer, second)
                except FloodWaitError as e:
                    await asyncio.sleep(e.seconds)
                    await client.send_message(peer, second)
                if is_group:
                    _append_group_ctx(chat_id, "лиса", second)
                logger.info(f"Split-reply в чат {chat_id}: {second[:60]}")

        if is_group:
            schedule_idle_message(chat_id)
            _append_group_ctx(chat_id, "лиса", reply)

        if chat_id not in chat_message_log:
            chat_message_log[chat_id] = []
        chat_message_log[chat_id].append(True)
        if len(chat_message_log[chat_id]) > 5:
            chat_message_log[chat_id] = chat_message_log[chat_id][-5:]

        messages_since_reply[chat_id] = 0
        if event.sender_id is not None:
            followup_user_id[chat_id] = event.sender_id
            followup_expires[chat_id] = time.time() + 60

    except asyncio.CancelledError:
        logger.info(f"Задача отменена для чата {chat_id}")
    finally:
        if chat_id in pending_tasks and task in pending_tasks[chat_id]:
            pending_tasks[chat_id].remove(task)
            if not pending_tasks[chat_id]:
                del pending_tasks[chat_id]


@client.on(events.ChatAction)
async def handle_chat_action(event):
    if MY_ID is None:
        return
    joined = False
    if event.user_joined and event.sender_id == MY_ID:
        joined = True
    elif event.user_added:
        try:
            users = await event.get_users()
            if any(u.id == MY_ID for u in users):
                joined = True
        except Exception:
            pass
    if joined:
        chat_id = event.chat_id
        online_mode_until[chat_id] = time.time() + ONLINE_WINDOW
        logger.info(f"[{chat_id}] Вступление в чат — онлайн-режим активирован")


@client.on(events.NewMessage(incoming=True))
async def handle_message(event):
    if not event.message.text:
        return
    if event.sender_id == MY_ID:
        return
    # Анонимный пост (sender_id=None, напр. канал без аккаунта) —
    # обрабатываем только реплаи, всё остальное игнорируем
    if event.sender_id is None and not event.message.is_reply:
        return

    chat_id   = event.chat_id
    sender_id = event.sender_id
    text      = event.message.text
    is_private = event.is_private
    is_group   = not is_private

    # Кешируем InputPeer из входящего события — гарантированно резолвится,
    # в отличие от client.get_input_entity(user_id) для новых пользователей
    if chat_id not in _peer_cache:
        try:
            _peer_cache[chat_id] = await event.get_input_chat()
        except Exception as e:
            logger.warning(f"Не удалось закешировать peer для {chat_id}: {e}")

    # Отметить сообщение как прочитанное (двойная галочка у собеседника)
    try:
        await client.send_read_acknowledge(
            _peer_cache.get(chat_id, chat_id), event.message
        )
    except Exception:
        pass

    increment_counter(chat_id)
    messages_since_reply[chat_id] = messages_since_reply.get(chat_id, 0) + 1

    if chat_id not in chat_message_log:
        chat_message_log[chat_id] = []
    chat_message_log[chat_id].append(False)
    if len(chat_message_log[chat_id]) > 5:
        chat_message_log[chat_id] = chat_message_log[chat_id][-5:]

    logger.info(f"Сообщение от {sender_id} в чате {chat_id}: {text[:60]}")

    if is_group and chat_id not in idle_timers:
        schedule_idle_message(chat_id)

    # Записать в групповой буфер контекста
    if is_group:
        _append_group_ctx(chat_id, _get_sender_name(event), text)

    trigger_type = None
    user_text    = text

    # Абсолютный приоритет: тег аккаунта (@Foxlike_creature / @Foxllke_creature)
    text_lower = text.lower()
    for tag in TRIGGER_TAGS:
        if tag in text_lower:
            cleaned = text
            for t in TRIGGER_TAGS:
                cleaned = cleaned.replace(t, "").replace(t.upper(), "").replace(t.capitalize(), "")
            user_text    = cleaned.strip() if cleaned.strip() else "шо?"
            trigger_type = "tag"
            break

    # ЛС — всегда отвечаем
    if not trigger_type and is_private:
        trigger_type = "reply"

    # Реплай на моё сообщение — включить цитируемый текст в контекст
    if not trigger_type and event.message.is_reply:
        reply_msg = await event.message.get_reply_message()
        if reply_msg and reply_msg.sender_id == MY_ID:
            trigger_type = "reply"
            if reply_msg.text:
                user_text = f"[Ты ранее написала: {reply_msg.text}]\n\n{text}"

    # Слово-триггер: лиса/рыба в любой форме, в любом месте сообщения
    if not trigger_type and _TRIGGER_RE.search(text):
        user_text    = text
        trigger_type = "word"

    # Вопрос пока онлайн
    if not trigger_type and (
        text.endswith("?")
        and is_group
        and not event.message.is_reply
        and any(chat_message_log.get(chat_id, []))
    ):
        trigger_type = "question"

    # Followup — продолжение от того же пользователя
    if not trigger_type and is_followup(sender_id, chat_id):
        trigger_type = "followup"

    # Лесенка — докидываем в буфер если активна
    if not trigger_type and is_group and chat_id in ladder_user_id and sender_id == ladder_user_id[chat_id]:
        ladder_bullets[chat_id].append(text)
        ladder_counter[chat_id] = ladder_counter.get(chat_id, 0) + 1
        return

    # Случайный ответ в онлайн-режиме
    if not trigger_type:
        if (
            is_group
            and online_mode_until.get(chat_id, 0) > time.time()
            and random.random() < online_chance(chat_id)
        ):
            trigger_type = "random_online"
        else:
            return

    logger.info(f"[{chat_id}] Триггер {trigger_type}: {user_text[:60]}")

    # Сбросить лесенку и followup при новом триггере
    ladder_bullets.pop(chat_id, None)
    ladder_user_id.pop(chat_id, None)
    ladder_message_id.pop(chat_id, None)
    ladder_counter.pop(chat_id, None)
    followup_user_id.pop(chat_id, None)
    followup_expires.pop(chat_id, None)

    cancel_pending_tasks(chat_id)
    counter_snapshot = get_message_counter(chat_id)
    asyncio.create_task(process_message(event, user_text, trigger_type, counter_snapshot))


async def main() -> None:
    resolve_model()
    load_rag_index()

    await client.connect()
    try:
        me = await client.get_me()
        if me is None:
            raise RuntimeError("get_me() вернул None")
    except Exception as e:
        logger.error(f"Сессия не авторизована: {e} — запусти auth.py вручную")
        sys.exit(1)

    global MY_ID
    MY_ID = me.id
    logger.info(f"Юзербот запущен: {me.first_name} (@{me.username}), ID: {me.id}")

    await client.run_until_disconnected()


asyncio.run(main())
