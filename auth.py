#!/usr/bin/env python3
"""
Первичная авторизация юзербота. Запустить один раз вручную:
    python auth.py
Telethon запросит номер телефона и код из Telegram.
После успешного входа создаётся файл userbot.session — больше auth.py не нужен.
"""
import os
from dotenv import load_dotenv
from telethon.sync import TelegramClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
SESSION_NAME = os.getenv("SESSION_NAME", "userbot")

with TelegramClient(os.path.join(BASE_DIR, SESSION_NAME), API_ID, API_HASH) as client:
    me = client.get_me()
    print(f"Авторизация успешна!")
    print(f"  Аккаунт: {me.first_name} (@{me.username})")
    print(f"  ID: {me.id}")
    print(f"  Сессия: {SESSION_NAME}.session")
