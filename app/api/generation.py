# Файл: acad_writer_pro/app/api/generation.py
from fastapi import APIRouter, HTTPException, Depends, Body, Path as FastApiPath, status
from typing import List, Optional, Dict, Any
import uuid
import logging

from pydantic import BaseModel, Field

from app.core.config import MD_SYSTEM_PROMPT_COURSEWORK, MD_SYSTEM_PROMPT_DIPLOMA
from app.llm_clients.gemini.config import ApiKeysManager
from app.llm_clients.gemini.gemini import GenaiRequest
from app.services.llm_service import LLMService
from app.services.project_service import ProjectService
from app.models.generation import ProjectGenerationRequest, ProjectGenerationResponse

# ProjectGenerationRequest теперь содержит selected_project_files

logger = logging.getLogger(__name__)
router = APIRouter(
    tags=["LLM Generation (Project-specific)"],
)




# --- Зависимости ---
def get_project_service():
    return ProjectService()


def get_llm_service(project_service: ProjectService = Depends(get_project_service)):
    return LLMService(project_service=project_service)


@router.post(
    "/projects/{project_id}/generate",
    response_model=ProjectGenerationResponse,
    summary="Сгенерировать текст для проекта и обновить историю чата"
)
async def generate_for_project(
        project_id: uuid.UUID = FastApiPath(..., description="ID проекта"),
        request_data: ProjectGenerationRequest = Body(...),  # Модель теперь включает selected_project_files
        llm_service: LLMService = Depends(get_llm_service),
        project_service: ProjectService = Depends(get_project_service)
):
    project_meta = project_service.get_project_meta(str(project_id))
    if not project_meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Проект с ID {project_id} не найден.")

    # selected_project_files теперь приходят напрямую из request_data
    model_response_content, updated_history = await llm_service.generate_text_for_project(
        project_id=str(project_id),
        user_prompt_text=request_data.prompt,
        work_mode=request_data.work_mode,
        selected_project_files=request_data.selected_project_files  # Передаем список имен файлов
    )

    # Формируем ProjectGenerationResponse, который включает project_id
    response_payload = ProjectGenerationResponse(
        project_id=project_id,
        model_response=None,
        chat_history=updated_history,  # <--- Присваиваем ее здесь
        error=None
    )

    if model_response_content and "error" in model_response_content:
        response_payload.error = model_response_content["error"]
    elif not model_response_content:  # Если сервис вернул None для ответа модели (например, критическая ошибка до вызова LLM)
        response_payload.error = "Сервис генерации не вернул ответ модели."
    else:
        response_payload.model_response = model_response_content

    return response_payload


class MarkdownFormatRequest(BaseModel):
    raw_text: str
    document_type: str = Field(..., pattern="^(coursework|diploma)$")


class MarkdownFormatResponse(BaseModel):
    formatted_markdown: str
    error: Optional[str] = None


@router.post(  # Используем тот же 'router', который уже определен в этом файле
    "/llm/format-to-markdown",
    response_model=MarkdownFormatResponse,
    summary="Отформатировать сырой текст в Markdown с помощью LLM"
)
async def format_text_to_markdown(
        request_data: MarkdownFormatRequest,
        # llm_service: LLMService = Depends(get_llm_service) # Можно использовать LLMService, если хочешь инкапсулировать вызов Gemini
        # Но для одиночного вызова можно и напрямую GenaiRequest
):
    logger.info(f"Запрос на форматирование в Markdown для типа: {request_data.document_type}")
    system_prompt_path = None
    if request_data.document_type == "coursework":
        system_prompt_path = MD_SYSTEM_PROMPT_COURSEWORK
    elif request_data.document_type == "diploma":
        system_prompt_path = MD_SYSTEM_PROMPT_DIPLOMA
    else:
        # Эта проверка избыточна, если Pydantic валидирует pattern
        logger.error(f"Неподдерживаемый document_type: {request_data.document_type}")
        raise HTTPException(status_code=400, detail="Неподдерживаемый тип документа для Markdown форматирования")

    if not system_prompt_path or not system_prompt_path.exists():
        logger.error(
            f"Системный промпт для Markdown ({request_data.document_type}) не найден по пути: {system_prompt_path}")
        return MarkdownFormatResponse(formatted_markdown="", error="Ошибка сервера: не найден файл системного промпта.")

    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt_content = f.read()
    except Exception as e:
        logger.error(f"Ошибка чтения системного промпта {system_prompt_path}: {e}", exc_info=True)
        return MarkdownFormatResponse(formatted_markdown="",
                                      error="Ошибка сервера: не удалось прочитать системный промпт.")

    if not ApiKeysManager().get_working_key():
        logger.error("LLM format-to-markdown: Не найдены рабочие API ключи Gemini.")
        return MarkdownFormatResponse(formatted_markdown="", error="Ошибка конфигурации сервера: API ключи недоступны.")

    # Прямой вызов GenaiRequest
    request_handler = GenaiRequest(
        prompt=request_data.raw_text,  # Сырой текст как основной промпт
        history=[],  # Без истории для этого специфичного запроса
        attachments=None,  # Без вложений для этого запроса
        system_instruction=system_prompt_content  # Наш специальный Markdown-промпт
    )

    try:
        gemini_result = await request_handler.full_response()
        logger.debug(f"Ответ Gemini (format-to-markdown): {gemini_result}")

        if gemini_result.get("status") == "complete" and "body" in gemini_result:
            markdown_text = gemini_result["body"].get("text", "")
            if not markdown_text.strip():  # Если Gemini вернул пустой текст
                logger.warning("LLM вернул пустой Markdown после форматирования.")
                # Можно вернуть ошибку или пустую строку, в зависимости от желаемого поведения
                return MarkdownFormatResponse(formatted_markdown="",
                                              error="LLM вернул пустой результат форматирования.")
            return MarkdownFormatResponse(formatted_markdown=markdown_text)
        else:
            error_msg_detail = gemini_result.get("body", {}).get("message",
                                                                 "Неизвестная ошибка LLM при форматировании в Markdown")
            try:  # Попытка извлечь более детальную ошибку, если она в JSON-строке
                import json
                error_detail_json = json.loads(error_msg_detail)
                actual_error = error_detail_json.get("error", {}).get("message", error_msg_detail)
            except:
                actual_error = error_msg_detail

            logger.error(f"Ошибка LLM при форматировании в Markdown: {actual_error}")
            return MarkdownFormatResponse(formatted_markdown="", error=f"Ошибка LLM: {actual_error}")
    except Exception as e:
        logger.error(f"Исключение при вызове LLM для форматирования в Markdown: {e}", exc_info=True)
        return MarkdownFormatResponse(formatted_markdown="",
                                      error=f"Внутренняя ошибка сервера при форматировании: {str(e)}")