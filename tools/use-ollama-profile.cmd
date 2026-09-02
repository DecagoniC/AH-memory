@echo off
setlocal EnableExtensions
cd /d "%~dp0.."
set "ROOT=%CD%"
set "PROFILE=%~1"
if "%PROFILE%"=="" set "PROFILE=8b"

if /I "%PROFILE%"=="8b" goto :ok
if /I "%PROFILE%"=="moe" goto :ok
if /I "%PROFILE%"=="30b" (
  set "PROFILE=moe"
  goto :ok
)
echo Usage: %~nx0 [8b^|moe]
echo   8b  - qwen3:8b dense (fast, fits 12GB VRAM)
echo   moe - qwen3:30b-a3b MoE (heavier, RAM offload on 12GB)
exit /b 1

:ok
set "SRC=%ROOT%\config\profiles\ollama-%PROFILE%.yaml"
if not exist "%SRC%" (
  echo Profile not found: %SRC%
  exit /b 1
)

copy /Y "%SRC%" "%ROOT%\config.local.yaml" >nul
if errorlevel 1 exit /b 1

if /I "%PROFILE%"=="8b" (
  set "MODEL=qwen3:8b"
  set "CTX=4096"
  set "TO=120"
) else (
  set "MODEL=qwen3:30b-a3b"
  set "CTX=2048"
  set "TO=300"
)

(
  echo LLM_PROVIDER=ollama
  echo OLLAMA_BASE_URL=http://127.0.0.1:11434
  echo OLLAMA_CHAT_MODEL=%MODEL%
  echo OLLAMA_EMBEDDING_MODEL=nomic-embed-text
  echo OLLAMA_TIMEOUT_SEC=%TO%
  echo OLLAMA_NUM_CTX=%CTX%
  echo OLLAMA_MODELS=%ROOT%\.ollama\models
) > "%ROOT%\.env"

echo [ah-profile] active=%PROFILE% model=%MODEL% num_ctx=%CTX%
echo [ah-profile] wrote config.local.yaml and .env
echo [ah-profile] restart web UI: python -m web.app
exit /b 0
