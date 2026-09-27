# Add "Algorithmic Intelligence Lab" to the Windows Start menu.
# Run from the repository folder after `uv sync`:
#   powershell -ExecutionPolicy Bypass -File tools\install_shortcut.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root ".venv\Scripts\ailab-gui.exe"
if (-not (Test-Path $exe)) { throw "Run 'uv sync' in $root first." }
$menu = [Environment]::GetFolderPath("Programs")
$shell = New-Object -ComObject WScript.Shell
$lnk = $shell.CreateShortcut((Join-Path $menu "Algorithmic Intelligence Lab.lnk"))
$lnk.TargetPath = $exe
$lnk.WorkingDirectory = $root
$lnk.Description = "Learn deterministic intelligent algorithms by playing with them"
$lnk.Save()
Write-Host "Installed. Look for 'Algorithmic Intelligence Lab' in the Start menu."
