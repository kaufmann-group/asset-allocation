# run as `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\setup.ps1`

$ErrorActionPreference = "Stop"

Set-Location $REPO

python -m venv "$REPO\ocean"

& "$REPO\ocean\Scripts\Activate.ps1"

python -m pip install --upgrade pip
pip install -r "$REPO\scripts\requirements.txt"

python -m ipykernel install --user --name=repo-env --display-name "Asset Allocation Repository Environment"

Write-Host ""
Write-Host "Setup complete."
Write-Host "Activate the environment from the project root with:"
Write-Host ".\ocean\Scripts\Activate.ps1"