from typing import Optional, List, Dict

import uvicorn
import logging
from pathlib import Path
import shutil
import base64
import uuid
import tempfile
import os
from contextlib import asynccontextmanager  # Импортируем для lifespan

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.core.config import (
    STATIC_FILES_DIR,
    LOG_LEVEL,
    LOG_FORMAT,
    PROJECTS_DATA_DIR
)

from app.api import projects as projects_router
from app.api import generation as generation_api # <--- ДОБАВЬ ЭТУ СТРОКУ

from app.llm_clients.gemini.gemini import GenaiRequest
from app.llm_clients.gemini.config import ApiKeysManager
from app.llm_clients.gemini.system_instruction import prompt_diplom, prompt_kyrsovie

# Настройка логирования
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)  # Используем __name__ для корректного имени логгера модуля


# --- Lifespan менеджер для событий startup/shutdown ---
@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    # Код, который выполняется перед запуском приложения (эквивалент startup)
    logger.info("Приложение запускается (lifespan)...")
    if not PROJECTS_DATA_DIR.exists():
        logger.warning(f"Директория для данных проектов {PROJECTS_DATA_DIR} не существует. "
                       "Сервис проектов может создать ее при первом вызове.")
    # Здесь можно инициализировать соединения с базами данных, если они появятся, и т.д.

    yield  # Это точка, где приложение будет работать

    # Код, который выполняется после остановки приложения (эквивалент shutdown)
    logger.info("Приложение останавливается (lifespan)...")


# --- Инициализация FastAPI App с lifespan ---
app = FastAPI(
    title="ALLMAX Academic Writer PRO",
    description="Веб-приложение для генерации и форматирования академических работ.",
    version="0.1.0",
    lifespan=lifespan  # Передаем lifespan менеджер
)

# --- Подключение Роутеров ---
app.include_router(projects_router.router, prefix="/api/v1")
app.include_router(generation_api.router, prefix="/api/v1")

# --- Статические файлы ---
if STATIC_FILES_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static")
    logger.info(f"Статические файлы монтируются из: {STATIC_FILES_DIR}")
else:
    logger.warning(
        f"Директория статических файлов '{STATIC_FILES_DIR}' не найдена. "
        "CSS, JS и изображения фронтенда не будут доступны."
    )


# --- Основной маршрут для фронтенда ---
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root(request: Request):
    index_path = STATIC_FILES_DIR / "index.html"
    if not index_path.exists():
        logger.error(f"Файл index.html не найден в директории '{STATIC_FILES_DIR}'.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Основной файл интерфейса (index.html) не найден."
        )
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Ошибка чтения index.html: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка загрузки основного интерфейса."
        )


# --- Старый эндпоинт /generate (для совместимости) ---
class LegacyPromptRequest(BaseModel):
    prompt: str
    image_data: Optional[List[str]] = None
    mode: str = "standard"


class LegacyGeminiResponse(BaseModel):
    response: str


class LegacyErrorResponse(BaseModel):
    detail: str


@app.post("/generate",
          response_model=LegacyGeminiResponse,
          responses={
              status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": LegacyErrorResponse},
              status.HTTP_400_BAD_REQUEST: {"model": LegacyErrorResponse}
          },
          tags=["Legacy LLM Generation"],
          summary="Легаси эндпоинт для генерации текста (без привязки к проекту)",
          deprecated=True
          )
async def legacy_generate_response(request_data: LegacyPromptRequest):
    logger.info(f"Legacy /generate: Получен запрос. Режим: {request_data.mode}, промпт: {request_data.prompt[:50]}...")
    if request_data.image_data:
        logger.info(f"Legacy /generate: Получено изображений: {len(request_data.image_data)}")

    if not ApiKeysManager().get_working_key():
        logger.error("Legacy /generate: Не найдены рабочие API ключи Gemini.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка конфигурации сервера: API ключи Gemini недоступны."
        )

    final_prompt = request_data.prompt
    system_instruction_text: Optional[str] = "You are an assistant."

    if request_data.mode == "coursework":
        system_instruction_text = prompt_kyrsovie
    elif request_data.mode == "diploma":
        system_instruction_text = prompt_diplom

    temp_dir_path_obj: Optional[tempfile.TemporaryDirectory] = None
    attachments_for_gemini: List[Dict[str, str]] = []

    try:
        if request_data.image_data:
            temp_dir_path_obj = tempfile.TemporaryDirectory()
            temp_dir_path = Path(temp_dir_path_obj.name)
            logger.info(f"Legacy /generate: Создана временная директория: {temp_dir_path}")

            for i, img_data_url in enumerate(request_data.image_data):
                try:
                    header, encoded_data = img_data_url.split(',', 1)
                    mime_type = header.split(';')[0].split(':')[1]
                    if not mime_type.startswith("image/"):
                        logger.warning(f"Legacy /generate: Пропущен не-изображенческий тип MIME: {mime_type}")
                        continue

                    extension = mime_type.split('/')[-1]
                    if extension not in ["jpeg", "jpg", "png"]:  # Простая проверка расширений
                        logger.warning(
                            f"Legacy /generate: Неподдерживаемое расширение {extension} для base64 изображения, пропуск.")
                        continue

                    img_bytes = base64.b64decode(encoded_data)
                    # Временное имя файла должно иметь правильное расширение для корректной обработки MIME-типа в GenaiRequest
                    filename = f"upload_{uuid.uuid4()}.{extension if extension != 'jpeg' else 'jpg'}"
                    filepath = temp_dir_path / filename
                    with open(filepath, "wb") as f:
                        f.write(img_bytes)

                    attachments_for_gemini.append({"path": str(filepath), "mime_type": mime_type})
                    logger.info(
                        f"Legacy /generate: Временный файл изображения сохранен и добавлен в аттачи: {filepath}")
                except Exception as e:
                    logger.error(f"Legacy /generate: Ошибка обработки изображения #{i + 1}: {e}", exc_info=True)
                    continue

        request_handler = GenaiRequest(
            prompt=final_prompt,
            attachments=attachments_for_gemini if attachments_for_gemini else None,
            system_instruction=system_instruction_text
        )

        gemini_result = await request_handler.full_response()
        logger.debug(f"Legacy /generate: Сырой ответ от Gemini: {gemini_result}")

        if gemini_result.get("status") == "complete" and "body" in gemini_result:
            response_text = gemini_result["body"].get("text", "В ответе не найден текст.")
            return LegacyGeminiResponse(response=response_text)
        elif gemini_result.get("status") == "ERROR":
            error_body = gemini_result.get("body", {})
            error_message = error_body.get("message", "Неизвестная ошибка Gemini")
            error_type = gemini_result.get("type", "UNKNOWN_GEMINI_ERROR")
            logger.error(f"Legacy /generate: Ошибка API Gemini (тип: {error_type}): {error_message}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail=f"Ошибка API Gemini: {error_type} - {error_message}")
        else:
            logger.error(f"Legacy /generate: Неожиданная структура ответа от библиотеки Gemini: {gemini_result}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Неожиданный ответ от сервиса Gemini.")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Legacy /generate: Произошла непредвиденная ошибка: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Внутренняя ошибка сервера: {str(e)}")
    finally:
        if temp_dir_path_obj:
            try:
                temp_dir_path_obj.cleanup()
                logger.info(f"Legacy /generate: Временная директория {temp_dir_path_obj.name} удалена.")
            except Exception as cleanup_error:
                logger.error(
                    f"Legacy /generate: Ошибка при удалении временной директории {temp_dir_path_obj.name}: {cleanup_error}",
                    exc_info=True)


# --- Запуск сервера ---
if __name__ == "__main__":
    port = 8110
    host = "0.0.0.0"

    log_level_str = logging.getLevelName(LOG_LEVEL).lower()

    logger.info(f"Запуск сервера Uvicorn на http://{host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level=log_level_str
    )