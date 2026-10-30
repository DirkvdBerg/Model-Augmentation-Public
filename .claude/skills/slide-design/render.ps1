param([string]$pptx, [string]$outdir)
New-Item -ItemType Directory -Force $outdir | Out-Null
$app = New-Object -ComObject PowerPoint.Application
$p = $app.Presentations.Open($pptx, $true, $false, $false)
foreach ($s in $p.Slides) {
  $s.Export((Join-Path $outdir ("{0:D2}.png" -f $s.SlideIndex)), "PNG", 1920, 1080)
}
$p.Close()
$app.Quit()
Write-Output "rendered"
