@echo off
setlocal
set "ROOT=%~dp0.."
set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
set "OLLAMA_MODELS=%ROOT%\.ollama\models"
set "OLLAMA_HOST=127.0.0.1:11434"

if not exist "%OLLAMA_EXE%" (
  echo Ollama not found at %OLLAMA_EXE%
  exit /b 1
)

echo [ah-ollama] models dir: %OLLAMA_MODELS%
"%OLLAMA_EXE%" pull qwen3:8b
if errorlevel 1 exit /b 1
"%OLLAMA_EXE%" pull qwen3:30b-a3b
if errorlevel 1 exit /b 1
"%OLLAMA_EXE%" pull nomic-embed-text
if errorlevel 1 exit /b 1
"%OLLAMA_EXE%" list
echo [ah-ollama] done — switch with: tools\use-ollama-profile.cmd 8b ^| moe
