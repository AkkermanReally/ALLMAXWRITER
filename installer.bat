@echo off
echo ============================================================
echo =            Установщик зависимостей для ALLMAX WRITER     =
echo ============================================================
echo.
echo Убедитесь, что Python 3 уже установлен и добавлен в PATH.
echo Нажмите любую клавишу для продолжения или Ctrl+C для отмены.
pause >nul
echo.

REM Устанавливаем текущую директорию на ту, где лежит скрипт
pushd "%~dp0"

echo Создание виртуального окружения 'venv'...
python -m venv venv
if %errorlevel% neq 0 (
    echo Ошибка: Не удалось создать виртуальное окружение.
    echo Убедитесь, что Python установлен и команда 'python -m venv' работает.
    popd
    pause
    exit /b 1
)
echo Виртуальное окружение создано.
echo.

echo Активация виртуального окружения и установка библиотек...
call "venv\Scripts\activate.bat"
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Ошибка: Не удалось установить библиотеки из requirements.txt.
    popd
    pause
    exit /b 1
)
echo Библиотеки успешно установлены.
echo.

echo Деактивация виртуального окружения (не обязательно, просто для чистоты)
call "venv\Scripts\deactivate.bat"

echo.
echo ============================================================
echo =         Установка зависимостей завершена!               =
echo =      Теперь вы можете запустить run.bat            =
echo ============================================================
echo.

popd
pause
exit /b 0