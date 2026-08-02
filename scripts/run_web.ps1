# Restart AH Memory web UI (frees :8000 if busy).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONPATH = "src"
python -m web.app
