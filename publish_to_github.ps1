# Usage: .\publish_to_github.ps1 -RepoUrl https://github.com/ayuushi18/Self-RAG-style-Q-A-System-over-Indian-Railways-Manuals-.git
param(
    [Parameter(Mandatory = $true)]
    [string]$RepoUrl
)

$ErrorActionPreference = "Stop"

git --version | Out-Null

if (-not (git config user.email)) {
    Write-Host "Set your git identity first (git config --global user.email ...)." -ForegroundColor Red
    exit 1
}

# 1. .gitignore: keeps keys, large data and copied manual text out of the repo
$ignore = @(
    ".env",
    "venv/",
    "__pycache__/",
    "*.pyc",
    ".streamlit/secrets.toml",
    "data/raw/",
    "data/chroma/",
    "data/batches/",
    "data/verdicts/",
    "data/chunks.jsonl",
    "data/eval_retrieved.json",
    "data/hybrid_drafts.json",
    "data/sft_*.jsonl",
    "data/*.backup.jsonl",
    "azure_test.py",
    "test_embed.py",
    "debug.py",
    "apply_verdicts.py"
)
Set-Content -Path .gitignore -Value $ignore -Encoding ascii

# 2. Start the repository
if (-not (Test-Path .git)) {
    git init | Out-Null
}
git symbolic-ref HEAD refs/heads/main

# 3. Safety check: no secret from .env may appear in any file that will be committed
$secrets = @()
if (Test-Path .env) {
    $secrets = Get-Content .env |
        Where-Object { $_ -match "=" } |
        ForEach-Object { ($_ -split "=", 2)[1].Trim() } |
        Where-Object { $_.Length -ge 12 }
}

$files = git ls-files --others --exclude-standard
$leaks = @()
foreach ($f in $files) {
    if (-not (Test-Path $f)) { continue }
    foreach ($s in $secrets) {
        if (Select-String -Path $f -SimpleMatch -Pattern $s -Quiet) {
            $leaks += $f
        }
    }
}

if ($leaks.Count -gt 0) {
    Write-Host "STOP: a secret from .env appears in these files:" -ForegroundColor Red
    $leaks | Sort-Object -Unique | ForEach-Object { Write-Host "  $_" }
    exit 1
}
Write-Host "Secret check passed." -ForegroundColor Green

# 4. Show what will be committed
git add .
git status --short
Read-Host "Check the list above. Press Enter to commit and push (Ctrl+C to cancel)"

# 5. Commit and push
git commit -m "Self-RAG-style QA over Indian Railways manuals"
if ((git remote) -contains "origin") {
    git remote remove origin
}
git remote add origin $RepoUrl
git push -u origin main

Write-Host "Done. Open your repository on github.com to check it." -ForegroundColor Green