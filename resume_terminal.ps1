$ErrorActionPreference = 'Stop'
$taskWorkspace = $PSScriptRoot
$taskCheckpoint = 'D:\vietfood\runs\detect\vietfood68_5070-2\weights\last.pt'
$taskLog = 'D:\vietfood\runs\detect\vietfood68_5070-2\resume-terminal.log'
$taskPython = Join-Path $taskWorkspace '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $taskCheckpoint)) { throw "Checkpoint missing: $taskCheckpoint" }
Set-Location -LiteralPath $taskWorkspace
& $taskPython -c "import psutil; from pathlib import Path; active=[p.pid for p in psutil.process_iter(['name','cmdline']) if 'python' in (p.info['name'] or '').lower() and any(Path(a).name=='train.py' for a in (p.info['cmdline'] or []))]; assert not active, f'Training already running: {active}'"
if ($LASTEXITCODE -ne 0) { throw 'Training is already running; refusing to start a duplicate.' }
$env:PYTHONIOENCODING = 'utf-8'
$taskUtf8 = New-Object System.Text.UTF8Encoding($false)
$OutputEncoding = $taskUtf8
[Console]::OutputEncoding = $taskUtf8
Write-Host "Resuming training on D from: $taskCheckpoint"
Write-Host "Live output is also saved to: $taskLog"
# PowerShell 5 represents native stderr as ErrorRecord; stringify it so the
# training progress is displayed as ordinary text instead of stopping the pipe.
$ErrorActionPreference = 'Continue'
& $taskPython -u (Join-Path $taskWorkspace 'train.py') --resume $taskCheckpoint --workers 3 --prefetch 2 --cache off 2>&1 |
    ForEach-Object { $_.ToString() } |
    Tee-Object -FilePath $taskLog -Append
$taskExitCode = $LASTEXITCODE
Write-Host "Training exited with code $taskExitCode"
