$py = if (Test-Path "$PSScriptRoot\..\.venv\Scripts\python.exe") {
    "$PSScriptRoot\..\.venv\Scripts\python.exe"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    "py"
} else {
    "python"
}

if ($py -eq "py") {
    & py -3 "$PSScriptRoot\..\nativa.py" $args
} else {
    & $py "$PSScriptRoot\..\nativa.py" $args
}

