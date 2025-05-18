from fastapi import APIRouter, HTTPException, Depends, status, Body, Path as FastApiPath, UploadFile, File
from fastapi import Query
from typing import List, Optional, Dict, Any
import uuid  # для валидации project_id в пути
import json  # для загрузки/сохранения JSON
import mimetypes

from pydantic import BaseModel

from app.api.generation import get_project_service
# Сервисы и модели
from app.services.project_service import ProjectService
from app.models.projects import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ProjectListResponse,
    FormattingConfiguration,
    FormattingConfigurationResponse,
    FormattingConfigurationUpdate,
    ChatHistoryResponse, ChatMessage,  # ChatMessage для тела запроса на добавление
    ProjectFileListResponse, ProjectFileContentResponse, ProjectFileEntry
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/projects",  # Добавил /api/v1 для версионирования
    tags=["Projects"],  # Группировка эндпоинтов в Swagger UI
)

# api/projects.py
# ... другие импорты
from fastapi.responses import FileResponse # Для возможной отправки файла напрямую
from app.models.projects import FormattingConfiguration # Нужна для типа current_formatting_options
from app.services.docx_service import DocxGenerationService # НОВЫЙ СЕРВИС
import tempfile # Для временных файлов
from pathlib import Path

# ... существующий router ...

class DocxGenerationRequest(BaseModel):
    markdown_content: str
    # Передаем текущие настройки форматирования, чтобы сервер мог их учесть
    # Это может быть полная модель FormattingConfiguration или только ее часть
    formatting_options: Optional[FormattingConfiguration] = None
    output_filename: Optional[str] = "generated_document.docx"

# Зависимость для DocxGenerationService
def get_docx_generation_service(project_service: ProjectService = Depends(get_project_service)):
    # DocxGenerationService может зависеть от ProjectService для доступа к файлам проекта
    return DocxGenerationService(project_service=project_service)

@router.post(
    "/{project_id}/generate-docx",
    # response_model=ProjectFileEntry, # Или другая модель, описывающая созданный файл
    summary="Сгенерировать DOCX файл из Markdown для проекта"
)
async def generate_docx_for_project(
    project_id: uuid.UUID,
    request_data: DocxGenerationRequest,
    project_service: ProjectService = Depends(get_project_service), # Уже есть
    docx_service: DocxGenerationService = Depends(get_docx_generation_service) # Новый сервис
):
    project_meta = project_service.get_project_meta(str(project_id))
    if not project_meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Проект не найден")

    logger.info(f"Запрос на генерацию DOCX для проекта {project_id}. Имя файла: {request_data.output_filename}")

    try:
        # DocxGenerationService должен инкапсулировать логику вызова main_processor
        # Он может принимать project_id, markdown_content, formatting_options, output_filename
        generated_file_entry: Optional[ProjectFileEntry] = await docx_service.generate_docx_from_markdown_string(
            project_id=str(project_id),
            markdown_string=request_data.markdown_content,
            formatting_options=request_data.formatting_options, # Передаем настройки
            output_filename=request_data.output_filename
        )

        if generated_file_entry:
            logger.info(f"DOCX файл '{generated_file_entry.name}' успешно сгенерирован для проекта {project_id}.")
            # Возвращаем информацию о файле, чтобы клиент мог его запросить/скачать
            return generated_file_entry
        else:
            logger.error(f"Не удалось сгенерировать DOCX для проекта {project_id}.")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Ошибка генерации DOCX файла на сервере.")

    except Exception as e:
        logger.error(f"Исключение при генерации DOCX для проекта {project_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Внутренняя ошибка сервера при генерации DOCX: {str(e)}")

# --- Зависимость для ProjectService ---
# Это позволит FastAPI автоматически создавать экземпляр сервиса для каждого запроса
# или использовать один и тот же экземпляр, если он реализован как синглтон.
def get_project_service():
    return ProjectService()


# === Эндпоинты для управления проектами ===

@router.post(
    "/",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Создать новый проект"
)
async def create_new_project(
        project_data: ProjectCreate = Body(...,
                                           description="Данные для создания нового проекта. Имя проекта, если не указано или пустое, будет сгенерировано автоматически."),
        service: ProjectService = Depends(get_project_service)
):
    """
    Создает новый проект с указанным именем или именем по умолчанию,
    если имя не предоставлено или пустое.
    """
    try:
        # Если project_data.name пустая строка или None, сервис сгенерирует имя по умолчанию
        project_name_to_create = project_data.name if (project_data.name and project_data.name.strip()) else None
        created_project_meta = service.create_project(
            project_name=project_name_to_create,
            description=project_data.description  # <--- ДОБАВИТЬ ЭТУ СТРОКУ
        )
        return ProjectResponse.model_validate(created_project_meta)  # Pydantic v2
        # Для Pydantic v1: return ProjectResponse(**created_project_meta)
    except FileExistsError as e:  # Это исключение может выбросить наш сервис, если ID уже существует
        logger.error(f"Ошибка создания проекта: Директория уже существует (очень редкий случай с UUID): {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка сервера при создании проекта: конфликт ID."
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка при создании проекта: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутренняя ошибка сервера при создании проекта: {str(e)}"
        )


@router.get(
    "/",
    response_model=ProjectListResponse,
    summary="Получить список всех проектов"
)
async def get_all_projects(
        service: ProjectService = Depends(get_project_service)
):
    """
    Возвращает список всех существующих проектов с их метаданными.
    """
    try:
        projects_meta_list = service.get_projects_list()
        response_projects = [ProjectResponse.model_validate(p_meta) for p_meta in projects_meta_list]  # Pydantic v2
        # Для Pydantic v1: response_projects = [ProjectResponse(**p_meta) for p_meta in projects_meta_list]
        return ProjectListResponse(projects=response_projects, total=len(response_projects))
    except Exception as e:
        logger.error(f"Ошибка при получении списка проектов: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка сервера при получении списка проектов."
        )


@router.get(
    "/{project_id}",
    response_model=ProjectResponse,
    summary="Получить метаданные конкретного проекта"
)
async def get_project_details(
        project_id: uuid.UUID = FastApiPath(..., description="ID проекта для получения деталей"),
        service: ProjectService = Depends(get_project_service)
):
    """
    Возвращает метаданные проекта по его ID.
    """
    project_meta = service.get_project_meta(str(project_id))
    if not project_meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Проект с ID '{project_id}' не найден."
        )
    return ProjectResponse.model_validate(project_meta)  # Pydantic v2
    # Для Pydantic v1: return ProjectResponse(**project_meta)


@router.put(
    "/{project_id}",
    response_model=ProjectResponse,
    summary="Обновить метаданные проекта"
)
async def update_project_details(
        project_id: uuid.UUID = FastApiPath(..., description="ID проекта для обновления"),
        project_data: ProjectUpdate = Body(...,
                                           description="Данные для обновления проекта (имя, описание). Обновляются только переданные поля."),
        service: ProjectService = Depends(get_project_service)
):
    """
    Обновляет метаданные проекта (например, имя, описание).
    Обновляются только те поля, которые переданы в теле запроса.
    """
    updates_to_apply = project_data.model_dump(exclude_unset=True)  # Pydantic V2
    # Для Pydantic V1: updates_to_apply = project_data.dict(exclude_unset=True)

    if not updates_to_apply:
        # Хотя модель ProjectUpdate позволяет пустые данные, если все поля Optional,
        # осмысленно это или нет - решает логика. Сейчас разрешим, но можно и 400.
        # Если ничего не передано на обновление, можно просто вернуть текущее состояние.
        current_meta = service.get_project_meta(str(project_id))
        if not current_meta:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Проект с ID '{project_id}' не найден.")
        return ProjectResponse.model_validate(current_meta)

    updated_meta = service.update_project_meta(str(project_id), updates_to_apply)
    if not updated_meta:
        # Проверяем, существует ли проект вообще, чтобы вернуть корректную ошибку
        if not service.get_project_meta(str(project_id)):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Проект с ID '{project_id}' не найден."
            )
        else:  # Если проект существует, но обновить не удалось по другой причине
            logger.error(f"Не удалось обновить метаданные для проекта {project_id}, хотя он существует.")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Ошибка сервера при обновлении метаданных проекта."
            )
    return ProjectResponse.model_validate(updated_meta)  # Pydantic v2
    # Для Pydantic v1: return ProjectResponse(**updated_meta)




# === Эндпоинты для управления файлами конфигурации форматирования проекта ===

@router.get(
    "/{project_id}/formatting-config",
    response_model=FormattingConfigurationResponse,
    summary="Получить конфигурацию форматирования для проекта"
)
async def get_project_formatting_config(
        project_id: uuid.UUID = FastApiPath(..., description="ID проекта"),
        service: ProjectService = Depends(get_project_service)
):
    """
    Возвращает текущую конфигурацию форматирования DOCX для указанного проекта.
    """
    config_content_str = service.read_project_file(str(project_id), "root", "formatting_config.json")
    if config_content_str is None:
        # Проверяем, существует ли проект
        if not service.get_project_meta(str(project_id)):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Проект с ID '{project_id}' не найден."
            )
        else:  # Проект есть, но файла конфигурации нет (маловероятно, если create_project работает корректно)
            logger.error(f"Файл formatting_config.json не найден для существующего проекта {project_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Файл конфигурации форматирования не найден для этого проекта."
            )
    try:
        config_data = json.loads(config_content_str)
        validated_config = FormattingConfiguration.model_validate(config_data)  # Pydantic v2
        # Для Pydantic v1: validated_config = FormattingConfiguration(**config_data)
        return FormattingConfigurationResponse(project_id=project_id, **validated_config.model_dump())
    except json.JSONDecodeError:
        logger.error(f"Ошибка декодирования JSON из formatting_config.json для проекта {project_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Ошибка чтения конфигурации форматирования: невалидный JSON.")
    except Exception as e:  # Включая ValidationError от Pydantic
        logger.error(f"Ошибка валидации или другая ошибка при получении конфига форматирования для {project_id}: {e}",
                     exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Ошибка обработки конфигурации: {str(e)}")


@router.put(
    "/{project_id}/formatting-config",
    response_model=FormattingConfigurationResponse,
    summary="Обновить конфигурацию форматирования для проекта"
)
async def update_project_formatting_config(
        project_id: uuid.UUID = FastApiPath(..., description="ID проекта"),
        config_update_data: FormattingConfigurationUpdate = Body(...,
                                                                 description="Данные для обновления конфигурации форматирования. Обновляются только переданные поля."),
        service: ProjectService = Depends(get_project_service)
):
    """
    Обновляет конфигурацию форматирования DOCX для указанного проекта.
    Позволяет обновлять только указанные поля (частичное обновление).
    """
    current_config_str = service.read_project_file(str(project_id), "root", "formatting_config.json")
    if current_config_str is None:
        if not service.get_project_meta(str(project_id)):  # Проверяем, существует ли проект
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Проект с ID '{project_id}' не найден.")
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                detail="Файл конфигурации форматирования не найден для обновления.")

    try:
        current_config_dict = json.loads(current_config_str)
        # Валидируем текущую конфигурацию для полноты
        current_formatting_config = FormattingConfiguration.model_validate(current_config_dict)  # Pydantic v2
        # Для Pydantic v1: current_formatting_config = FormattingConfiguration(**current_config_dict)

        # Применяем обновления
        update_data_dict = config_update_data.model_dump(exclude_unset=True)  # Pydantic V2
        # Для Pydantic V1: update_data_dict = config_update_data.dict(exclude_unset=True)

        # "Умное" глубокое обновление для Pydantic v2
        # model_copy создаст копию, а update применит изменения рекурсивно, если deep=True
        updated_config = current_formatting_config.model_copy(update=update_data_dict, deep=True)

        # Перед сохранением, убедимся, что результат все еще валиден (хотя model_copy с deep=True должен это обеспечить)
        # validated_updated_config = FormattingConfiguration.model_validate(updated_config.model_dump())
        # Этот шаг избыточен, если мы доверяем model_copy

        if not service.write_project_file(str(project_id), "root", "formatting_config.json",
                                          updated_config.model_dump_json(indent=4)):
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Не удалось сохранить обновленную конфигурацию форматирования.")

        return FormattingConfigurationResponse(project_id=project_id, **updated_config.model_dump())

    except json.JSONDecodeError:
        logger.error(f"Ошибка декодирования JSON для formatting_config.json проекта {project_id} при обновлении.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Ошибка чтения текущей конфигурации форматирования для обновления.")
    except Exception as e:  # Включая ошибки валидации Pydantic
        logger.error(f"Ошибка при обновлении конфигурации форматирования для {project_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Ошибка данных в запросе или при обработке конфигурации: {str(e)}")


@router.get(
    "/{project_id}/chat-history",
    response_model=ChatHistoryResponse,
    summary="Получить историю чата для проекта"
)
async def get_project_chat_history(
        project_id: uuid.UUID = FastApiPath(..., description="ID проекта"),
        service: ProjectService = Depends(get_project_service)
):
    """
    Возвращает историю сообщений для указанного проекта.
    """
    if not service.get_project_meta(str(project_id)):  # Проверка существования проекта
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Проект с ID '{project_id}' не найден.")

    history_messages = service.get_chat_history(str(project_id))
    if history_messages is None:  # Если файл не найден или ошибка чтения/декодирования
        logger.info(
            f"История чата для проекта {project_id} не найдена или не может быть прочитана. Возвращаем пустую историю.")
        # Вместо 404 для отсутствующего файла истории, вернем пустую историю, т.к. проект существует
        return ChatHistoryResponse(project_id=project_id, messages=[])

    try:
        # Валидируем каждое сообщение перед отправкой (если get_chat_history не делает этого)
        validated_messages = [ChatMessage.model_validate(msg) for msg in history_messages]
        return ChatHistoryResponse(project_id=project_id, messages=validated_messages)
    except Exception as e:  # Например, ValidationError от Pydantic
        logger.error(f"Ошибка валидации истории чата для проекта {project_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Ошибка обработки данных истории чата.")


@router.put(
    "/{project_id}/chat-history",
    response_model=ChatHistoryResponse,  # Возвращаем обновленную историю
    summary="Сохранить (перезаписать) всю историю чата для проекта"
)
async def save_project_chat_history(
        project_id: uuid.UUID = FastApiPath(..., description="ID проекта"),
        chat_data: List[ChatMessage] = Body(..., description="Полный список сообщений истории чата"),
        service: ProjectService = Depends(get_project_service)
):
    """
    Полностью перезаписывает историю чата для проекта.
    Принимает список сообщений.
    """
    if not service.get_project_meta(str(project_id)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Проект с ID '{project_id}' не найден.")

    # Преобразуем Pydantic модели обратно в словари для сохранения в JSON
    messages_to_save = [msg.model_dump() for msg in chat_data]

    if service.save_chat_history(str(project_id), messages_to_save):
        # Возвращаем сохраненные и заново провалидированные данные
        return ChatHistoryResponse(project_id=project_id, messages=chat_data)
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Не удалось сохранить историю чата.")


@router.post(
    "/{project_id}/chat-history/messages",
    response_model=ChatMessage,  # Возвращаем добавленное сообщение
    status_code=status.HTTP_201_CREATED,
    summary="Добавить новое сообщение в историю чата проекта"
)
async def add_message_to_project_chat_history(
        project_id: uuid.UUID = FastApiPath(..., description="ID проекта"),
        message_data: ChatMessage = Body(..., description="Данные нового сообщения"),
        service: ProjectService = Depends(get_project_service)
):
    """
    Добавляет одно новое сообщение в конец истории чата проекта.
    """
    if not service.get_project_meta(str(project_id)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Проект с ID '{project_id}' не найден.")

    if service.add_message_to_chat_history(str(project_id), message_data.model_dump()):
        return message_data  # Возвращаем принятое и (предположительно) сохраненное сообщение
    else:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Не удалось добавить сообщение в историю чата.")


# === Эндпоинты для файлов проекта ===

@router.get(
    "/{project_id}/files",
    response_model=ProjectFileListResponse,
    summary="Получить список файлов из директории проекта"
)
async def list_files_in_project_directory(
        project_id: uuid.UUID = FastApiPath(..., description="ID проекта"),
        directory: str = Query(..., description="Тип директории: 'source_materials' или 'generated_texts'",
                               pattern="^(source_materials|generated_texts)$"),
        service: ProjectService = Depends(get_project_service)
):
    """
    Возвращает список файлов и папок из указанной поддиректории проекта.
    """
    if not service.get_project_meta(str(project_id)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Проект с ID '{project_id}' не найден.")

    files = service.list_project_files(str(project_id), directory)
    if files is None:  # Сервис может вернуть None, если сама директория проекта не найдена
        # Эта проверка дублируется с проверкой get_project_meta, но на всякий случай
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Проект или его файловая структура для '{directory}' не найдены.")

    return ProjectFileListResponse(project_id=project_id, file_type_dir=directory, files=files)


class FileUploadResponse(BaseModel):
    successful_uploads: List[ProjectFileEntry] = []
    failed_uploads: List[Dict[str, str]] = [] # [{"filename": "...", "error": "..."}]


@router.post(
    "/{project_id}/files/upload",
    response_model=FileUploadResponse,  # Используем новую модель ответа
    status_code=status.HTTP_200_OK,  # Меняем на 200, т.к. может быть частичный успех
    summary="Загрузить один или несколько файлов в проект"
)
async def upload_files_to_project(
        project_id: uuid.UUID = FastApiPath(..., description="ID проекта"),
        files: List[UploadFile] = File(..., description="Список файлов для загрузки"),
        directory: str = Query("source_materials",
                               description="Тип директории для сохранения: 'source_materials'",
                               # Пока только source_materials
                               pattern="^(source_materials)$"),  # Убрал generated_texts для загрузки
        service: ProjectService = Depends(get_project_service)
):
    """
    Загружает один или несколько файлов в директорию 'source_materials' проекта.
    Файлы с неподдерживаемым MIME-типом будут отклонены.
    """
    project_meta = service.get_project_meta(str(project_id))
    if not project_meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Проект с ID '{project_id}' не найден."
        )

    if directory != "source_materials":  # На всякий случай, если pattern не сработает или кто-то обойдет Query
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Загрузка разрешена только в директорию 'source_materials'."
        )

    response_data = FileUploadResponse()

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Не предоставлено файлов для загрузки."
        )

    for file_to_upload in files:
        original_filename = file_to_upload.filename if file_to_upload.filename else "unknown_file"

        logger.info(f"Проект {project_id}: Попытка загрузки файла '{original_filename}' в директорию '{directory}'")

        saved_entry, error_message = service.save_uploaded_file(str(project_id), directory, file_to_upload)

        if saved_entry:
            response_data.successful_uploads.append(saved_entry)
        else:
            response_data.failed_uploads.append(
                {"filename": original_filename, "error": error_message or "Неизвестная ошибка сохранения файла."})
            logger.error(
                f"Проект {project_id}: Не удалось обработать файл '{original_filename}'. Причина: {error_message}")

    # Если ни один файл не был успешно загружен, но были попытки, можно вернуть ошибку 400 или 500
    # Но сейчас мы возвращаем 200 OK с отчетом об успешных и неудачных загрузках.
    # Это дает фронтенду больше информации.
    if not response_data.successful_uploads and response_data.failed_uploads:
        logger.warning(f"Проект {project_id}: Ни один из {len(files)} файлов не был успешно загружен.")
        # Можно установить другой код ответа, если это важно
        # Например, если все файлы не удались:
        # if len(files) == len(response_data.failed_uploads):
        #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=response_data.model_dump())

    return response_data

@router.get(
    "/{project_id}/files/content",
    response_model=ProjectFileContentResponse,  # или StreamingResponse для бинарных
    summary="Получить содержимое текстового файла проекта"
)
async def get_project_file_content(
        project_id: uuid.UUID = FastApiPath(..., description="ID проекта"),
        directory: str = Query(..., description="Тип директории: 'source_materials' или 'generated_texts'",
                               pattern="^(source_materials|generated_texts)$"),
        filename: str = Query(..., description="Имя файла для чтения"),
        service: ProjectService = Depends(get_project_service)
):
    """
    Возвращает содержимое указанного текстового файла из проекта.
    Для бинарных файлов (например, изображений) этот эндпоинт не подходит,
    для них лучше использовать отдельный эндпоинт для скачивания (возвращающий FileResponse).
    """
    if not service.get_project_meta(str(project_id)):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Проект с ID '{project_id}' не найден.")

    # Простая проверка безопасности имени файла (предотвращение выхода из директории)
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Некорректное имя файла.")

    content = service.read_project_file(str(project_id), directory, filename)
    if content is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Файл '{filename}' не найден в '{directory}' проекта '{project_id}'.")

    # Пытаемся определить MIME-тип для ответа (хотя для текстовых это менее критично)
    file_path = service.get_project_file_path(str(project_id), directory, filename)
    mime_type = None
    if file_path:
        mime_type, _ = mimetypes.guess_type(file_path)

    # Сейчас мы предполагаем, что читаем только текстовые файлы.
    # Если бы это были бинарные, нужно было бы использовать FileResponse или StreamingResponse.
    return ProjectFileContentResponse(
        project_id=project_id,
        filename=filename,
        file_type_dir=directory,
        content=content,
        mime_type=mime_type or "text/plain"
    )

@router.get(
    "/{project_id}/files/view", # Можно назвать "preview" или "display"
    # response_class=FileResponse, # FileResponse для прямого возврата файла
    summary="Получить файл для просмотра в браузере (если возможно)"
)
async def view_project_file(
    project_id: uuid.UUID = FastApiPath(..., description="ID проекта"),
    directory: str = Query(..., description="Тип директории: 'source_materials' или 'generated_texts'",
                           pattern="^(source_materials|generated_texts|root)$"), # Добавил 'root' если нужно
    filename: str = Query(..., description="Имя файла для просмотра"),
    service: ProjectService = Depends(get_project_service)
):
    """
    Возвращает файл для отображения в браузере.
    Для PDF, изображений, текста браузер попытается их отобразить.
    Для других типов может предложить скачивание.
    """
    from fastapi.responses import FileResponse # Импорт здесь, чтобы не загромождать верх модуля

    project_meta = service.get_project_meta(str(project_id))
    if not project_meta:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Проект с ID '{project_id}' не найден.")

    if ".." in filename or filename.startswith("/"): # Базовая безопасность
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Некорректное имя файла.")

    file_path = service.get_project_file_path(str(project_id), directory, filename)

    if not file_path or not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"Файл '{filename}' не найден в '{directory}' проекта '{project_id}'.")

    # Определяем MIME-тип для корректного Content-Disposition и Content-Type
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "application/octet-stream" # Общий тип, если не удалось определить

    # Для просмотра (inline) вместо скачивания (attachment)
    # Браузер сам решит, может ли он отобразить этот mime_type
    # Важно: Content-Disposition: inline может не всегда работать для всех типов/браузеров,
    # но это стандартный способ попросить браузер отобразить, а не скачать.
    return FileResponse(
        path=str(file_path),
        filename=filename, # Имя, которое увидит пользователь при скачивании (если браузер решит скачать)
        media_type=mime_type,
        content_disposition_type="inline" # Указываем "inline" для просмотра
    )

# TODO:
# - POST /{project_id}/files/upload?type=source_materials (для загрузки файлов)
# - GET /{project_id}/files/download?type=[source_materials|generated_texts]&filename=... (для скачивания, используя FileResponse)
# - DELETE /{project_id}/files?type=[source_materials|generated_texts]&filename=...
# - DELETE /{project_id} (для удаления всего проекта)