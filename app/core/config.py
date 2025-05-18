import os
from pathlib import Path
import logging

# Корень проекта (директория ALLMAX_WRITER)
# __file__ -> ALLMAX_WRITER/app/core/config.py
# .parent -> ALLMAX_WRITER/app/core/
# .parent -> ALLMAX_WRITER/app/
# .parent -> ALLMAX_WRITER/
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Директория для хранения данных проектов пользователей
PROJECTS_DATA_DIR_NAME = "projects_data"
PROJECTS_DATA_DIR = BASE_DIR / PROJECTS_DATA_DIR_NAME

# Директория для глобальных конфигураций приложения
APP_CONFIG_DIR_NAME = "config"
APP_CONFIG_DIR = BASE_DIR / APP_CONFIG_DIR_NAME

# Создаем директории, если они не существуют
os.makedirs(PROJECTS_DATA_DIR, exist_ok=True)
os.makedirs(APP_CONFIG_DIR, exist_ok=True) # Создаем папку config/ на уровне проекта

# Настройки логирования
LOG_LEVEL = logging.INFO # Можно изменить на logging.DEBUG для разработки
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Настройки LLM (Gemini)
# Ключи API по-прежнему управляются через llm_clients/gemini/config.py
# Это просто путь для информации или если понадобится проверить существование
GEMINI_CLIENT_CONFIG_PATH = BASE_DIR / "app" / "llm_clients" / "gemini" / "config.py"

# Настройки генератора DOCX
DOCX_GENERATOR_DIR = BASE_DIR / "app" / "docx_generator"
PANDOC_EXECUTABLE = "pandoc"  # Убедитесь, что pandoc в системном PATH или укажите полный путь
REFERENCE_DOCX_TEMPLATE_DIR = DOCX_GENERATOR_DIR / "reference_docs"
DEFAULT_REFERENCE_DOCX = REFERENCE_DOCX_TEMPLATE_DIR / "template.docx" # Из FORMATTER/template.docx

MD_SYSTEM_PROMPTS_DIR = DOCX_GENERATOR_DIR / "system_prompts"
MD_SYSTEM_PROMPT_COURSEWORK = MD_SYSTEM_PROMPTS_DIR / "coursework_md_prompt.txt"
MD_SYSTEM_PROMPT_DIPLOMA = MD_SYSTEM_PROMPTS_DIR / "diploma_md_prompt.txt"

# Путь к файлу с настройками форматирования DOCX по умолчанию
DEFAULT_FORMATTING_CONFIG_FILE = APP_CONFIG_DIR / "default_formatting.json"

# Статические файлы
STATIC_FILES_DIR = BASE_DIR / "app" / "static"

ALLOWED_UPLOAD_MIME_TYPES = [
    "image/jpeg",
    "image/png",
    "image/webp",
    # "image/heic", # Может потребовать доп. обработки для отображения или конвертации
    # "image/heif", # Может потребовать доп. обработки
    "application/pdf",
    "text/plain",
    "text/markdown", # .md
    # Аудио (если pydub/ffmpeg настроены или если LLM их принимает напрямую)
    "audio/mpeg",  # mp3
    "audio/wav",
    "audio/aac",
    "audio/ogg",   # Если решим поддерживать OGG без конвертации (например, для хранения)
    "audio/flac",
    # Видео (аналогично аудио)
    "video/mp4",
    "video/mpeg",
    "video/quicktime", # .mov
    # Документы, которые Gemini может напрямую обрабатывать через File API (если не конвертируем)
    # "application/msword", # .doc - лучше конвертировать в PDF
    # "application/vnd.openxmlformats-officedocument.wordprocessingml.document", # .docx - лучше конвертировать в PDF
    "application/rtf", # .rtf
    "text/html",       # .html
    "application/json"
]# .json

# Пример вывода путей для проверки (можно закомментировать позже)
if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"PROJECTS_DATA_DIR: {PROJECTS_DATA_DIR}")
    print(f"APP_CONFIG_DIR: {APP_CONFIG_DIR}")
    print(f"DEFAULT_REFERENCE_DOCX: {DEFAULT_REFERENCE_DOCX}")
    print(f"MD_SYSTEM_PROMPT_COURSEWORK: {MD_SYSTEM_PROMPT_COURSEWORK}")
    print(f"STATIC_FILES_DIR: {STATIC_FILES_DIR}")