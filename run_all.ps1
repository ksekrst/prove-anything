# Windows PowerShell runner. Same as run_all.sh.
#
# One-time setup, in PowerShell, from this folder:
#   python -m venv .venv
#   .\.venv\Scripts\Activate.ps1
#   pip install torch --index-url https://download.pytorch.org/whl/cu130
#   pip install -r requirements.txt
#
# (Plain "pip install torch" on Windows installs a CPU-only build; the line above gets the
#  CUDA build for the RTX 4080. Install torch first, then the requirements.)
#
# Every run:
#   .\.venv\Scripts\Activate.ps1
#   $env:HF_TOKEN = "hf_..."          # for the gated Gemma models
#   .\run_all.ps1                     # all models in config.py
#   .\run_all.ps1 Gemma_2_2B_instruct # one model by its output name
#
# If PowerShell refuses to run scripts, run this once in the same window:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# UTF-8 console output (token printouts include non-Latin characters)
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# Hugging Face warns about symlinks on Windows without Developer Mode; harmless.
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

uv run python -c "import torch, sys; ok = torch.cuda.is_available(); print('torch', torch.__version__, '| CUDA:', ok, '|', torch.cuda.get_device_name(0) if ok else 'no GPU found'); sys.exit(0 if ok else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host "PyTorch does not see the GPU. Reinstall torch with:"
    Write-Host "  pip install torch --index-url https://download.pytorch.org/whl/cu130 --force-reinstall"
    exit 1
}

$steps = @(
    "01_extract_activations.py",
    "02_vectors_and_validation.py",
    "03_unembedding.py",
    "04_steering_ladder.py",
    "05_injection_detection.py",
    "06_button_task.py"
)
foreach ($s in $steps) {
    Write-Host "`n===== $s =====" -ForegroundColor Cyan
    uv run python $s @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "$s failed (exit $LASTEXITCODE). Fix and rerun; finished work is kept." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}
uv run python 08_lexical_baseline.py
uv run python 07_report.py
Write-Host "`nDone. Open results\REPORT.md" -ForegroundColor Green
