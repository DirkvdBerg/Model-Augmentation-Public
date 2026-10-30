Add-Type -AssemblyName System.Drawing
$root = Join-Path $pwd 'tasks/handoffs/2026-10-29-codex-ps-figure-review'
$files = @(Get-ChildItem "$root/attempts" -Recurse -Filter *.png | Where-Object { $_.Name -ne 'psb_steps.png' -and $_.DirectoryName -notlike '*01_deck*' }) + @(Get-ChildItem "$root/reference" -Recurse -Filter *.png)
for ($start=0; $start -lt $files.Count; $start+=4) {
 $bmp = New-Object System.Drawing.Bitmap 1800,1400
 $g=[System.Drawing.Graphics]::FromImage($bmp); $g.Clear([System.Drawing.Color]::White)
 $font=New-Object System.Drawing.Font 'Arial',14
 for ($j=0; $j -lt 4 -and ($start+$j) -lt $files.Count; $j++) {
  $f=$files[$start+$j];$im=[System.Drawing.Image]::FromFile($f.FullName)
  $x=($j%2)*900; $y=[math]::Floor($j/2)*700
  $g.DrawString($f.Name,$font,[System.Drawing.Brushes]::Black,$x+10,$y+5)
  $scale=[math]::Min(880/$im.Width,660/$im.Height)
  $g.DrawImage($im,[int]($x+10),[int]($y+30),[int]($im.Width*$scale),[int]($im.Height*$scale));$im.Dispose()
 }
 $bmp.Save("$root/review-evidence/sheet-$start.png");$g.Dispose();$bmp.Dispose();$font.Dispose()
}
