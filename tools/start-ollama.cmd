@echo off
setlocal
set "ROOT=%~dp0.."
set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
set "OLLAMA_MODELS=%ROOT%\.ollama\models"
set "OLLAMA_HOST=127.0.0.1:11434"

if not exist "%OLLAMA_EXE%" (
  echo Ollama not found at %OLLAMA_EXE%
  echo Run tools\OllamaSetup.exe first.
  exit /b 1
)

if not exist "%OLLAMA_MODELS%" mkdir "%OLLAMA_MODELS%"

echo [ah-ollama] models dir: %OLLAMA_MODELS%
echo [ah-ollama] starting serve on %OLLAMA_HOST%
"%OLLAMA_EXE%" serve
