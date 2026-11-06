# Build + kiểm tra image nộp bài. Chạy từ GỐC repo:  .\docker\build.ps1
#
# Mọi thứ do script này sinh ra nằm trong docker\out\ (đã gitignore). Nó không ghi file nào
# ra chỗ khác -- trừ chính image, thứ nằm trong kho của Docker chứ không phải trong repo.
param(
    [string]$Tag  = "lifemate:submit",
    [string]$Data = "data\train+public_test\train+public_test",
    [int]$Samples = 3,
    [switch]$SkipBuild,
    [switch]$Full
)
$ErrorActionPreference = "Stop"

# Console PowerShell 5.1 la cp1252 va bop meo tieng Viet, nen moi thong bao o day la ASCII.
# Kiem engine TRUOC khi lam gi: thieu no thi `docker build` nem mot traceback PowerShell dai
# ma khong noi ro phai lam gi.
docker info --format "{{.ServerVersion}}" 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Docker Engine chua chay. CLI co, nhung daemon thi khong." -ForegroundColor Yellow
    Write-Host "Mo Docker Desktop, cho den khi icon bao Running, roi chay lai lenh nay."
    Write-Host "Kiem bang: docker info"
    exit 1
}
$root = (Get-Location).Path
$out  = Join-Path $root "docker\out"
New-Item -ItemType Directory -Force $out | Out-Null

if (-not $SkipBuild) {
    Write-Host "== build $Tag (context = goc repo, Dockerfile = docker\Dockerfile) =="
    docker build -f docker\Dockerfile -t $Tag . 2>&1 | Tee-Object -FilePath (Join-Path $out "build.log")
    if ($LASTEXITCODE -ne 0) { throw "build that bai; xem docker\out\build.log" }
}
docker image inspect $Tag --format 'image {{.Id}}  {{.Size}} byte' | Tee-Object -FilePath (Join-Path $out "image.txt")

# Smoke: mount public_test hệt cách BTC mount private_test, rồi đọc submission.jsonl.
$split   = Join-Path $root "$Data\public_test"
$resdir  = Join-Path $out "smoke"
Remove-Item -Recurse -Force $resdir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $resdir | Out-Null
$extra = if ($Full) { "" } else { "--max-samples $Samples" }
Write-Host "== smoke: mount $split vao /data =="
docker run --rm --gpus all `
    -v "${split}:/data:ro" `
    -v "${resdir}:/app/result" `
    -e "INFER_ARGS=$extra" `
    $Tag 2>&1 | Tee-Object -FilePath (Join-Path $out "smoke.log")

$sub = Join-Path $resdir "submission.jsonl"
if (-not (Test-Path $sub)) { throw "KHONG co submission.jsonl; xem docker\out\smoke.log" }
$n = (Get-Content $sub | Measure-Object -Line).Lines
Write-Host "== submission.jsonl: $n dong =="
Get-Content $sub -TotalCount 1
