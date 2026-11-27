# Build Windows standalone executable (dist/LocalLLMRunner.exe)
$ErrorActionPreference = "Stop"

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = "python"
}

& $py -m PyInstaller --clean --noconfirm packaging\LocalLLMRunner.spec
Write-Output "Built Windows executable at: dist\LocalLLMRunner.exe"
