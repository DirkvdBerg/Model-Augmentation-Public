<#
.SYNOPSIS
Extracts reviewable text and visual-presence metadata from PowerPoint files.

.DESCRIPTION
Reads PPTX files as Open XML archives without requiring PowerPoint. For each deck it emits slide
text, speaker-note text, image counts/targets and chart labels. Exact duplicate files below the root
are identified by SHA-256 and emitted once with a duplicate marker.

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/thesis/extract_pptx_text.ps1 `
  -Root "docs/Thesis-documentation/Meetings ASMPT"
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Root,
    [string]$NamePattern = '*.pptx'
)

Add-Type -AssemblyName System.IO.Compression.FileSystem

function Read-ZipXmlText {
    param(
        [System.IO.Compression.ZipArchive]$Archive,
        [string]$EntryName,
        [string]$XPath = "//*[local-name()='t']"
    )

    $entry = $Archive.GetEntry($EntryName)
    if ($null -eq $entry) {
        return @()
    }

    $stream = $entry.Open()
    try {
        $reader = [System.IO.StreamReader]::new($stream)
        try {
            [xml]$xml = $reader.ReadToEnd()
        }
        finally {
            $reader.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }

    return @($xml.SelectNodes($XPath) | ForEach-Object { $_.InnerText })
}

function Read-ZipXml {
    param(
        [System.IO.Compression.ZipArchive]$Archive,
        [string]$EntryName
    )

    $entry = $Archive.GetEntry($EntryName)
    if ($null -eq $entry) {
        return $null
    }

    $stream = $entry.Open()
    try {
        $reader = [System.IO.StreamReader]::new($stream)
        try {
            return [xml]$reader.ReadToEnd()
        }
        finally {
            $reader.Dispose()
        }
    }
    finally {
        $stream.Dispose()
    }
}

function Resolve-PptxTarget {
    param(
        [string]$SourceEntry,
        [string]$Target
    )

    $base = [Uri]::new("https://pptx.local/$SourceEntry")
    $resolved = [Uri]::new($base, $Target)
    return $resolved.AbsolutePath.TrimStart('/')
}

$resolvedRoot = (Resolve-Path -LiteralPath $Root).Path
$files = Get-ChildItem -LiteralPath $resolvedRoot -Recurse -File -Filter $NamePattern | Sort-Object FullName
$seenHashes = @{}

foreach ($file in $files) {
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    $relativeName = $file.FullName.Substring($resolvedRoot.Length + 1)
    if ($seenHashes.ContainsKey($hash)) {
        "===== DUPLICATE: $relativeName ====="
        "Same content as: $($seenHashes[$hash])"
        continue
    }
    $seenHashes[$hash] = $relativeName

    "===== DECK: $relativeName ====="
    $archive = [System.IO.Compression.ZipFile]::OpenRead($file.FullName)
    try {
        $slides = @($archive.Entries |
            Where-Object { $_.FullName -match '^ppt/slides/slide([0-9]+)\.xml$' } |
            Sort-Object { [int]([regex]::Match($_.FullName, 'slide([0-9]+)\.xml$').Groups[1].Value) })

        "Slide count: $($slides.Count)"
        foreach ($slide in $slides) {
            $slideNumber = [int]([regex]::Match($slide.FullName, 'slide([0-9]+)\.xml$').Groups[1].Value)
            "--- SLIDE $slideNumber ---"
            $slideText = Read-ZipXmlText -Archive $archive -EntryName $slide.FullName
            if ($slideText.Count -gt 0) {
                $slideText -join ' | '
            }
            else {
                '[no extractable slide text]'
            }

            $relsName = "ppt/slides/_rels/slide$slideNumber.xml.rels"
            $rels = Read-ZipXml -Archive $archive -EntryName $relsName
            if ($null -eq $rels) {
                continue
            }

            $relationships = @($rels.SelectNodes("//*[local-name()='Relationship']"))
            $imageTargets = @($relationships | Where-Object { $_.Type -match '/image$' } | ForEach-Object { $_.Target })
            if ($imageTargets.Count -gt 0) {
                "[images: $($imageTargets.Count); targets: $($imageTargets -join ', ')]"
            }

            foreach ($relationship in $relationships | Where-Object { $_.Type -match '/notesSlide$' }) {
                $notesEntry = Resolve-PptxTarget -SourceEntry $slide.FullName -Target $relationship.Target
                $notesText = @(Read-ZipXmlText -Archive $archive -EntryName $notesEntry | Where-Object { $_ -notmatch '^\s*[0-9]+\s*$' })
                if ($notesText.Count -gt 0) {
                    "[speaker notes] $($notesText -join ' | ')"
                }
            }

            foreach ($relationship in $relationships | Where-Object { $_.Type -match '/chart$' }) {
                $chartEntry = Resolve-PptxTarget -SourceEntry $slide.FullName -Target $relationship.Target
                $chartText = Read-ZipXmlText -Archive $archive -EntryName $chartEntry -XPath "//*[local-name()='tx']//*[local-name()='v'] | //*[local-name()='strCache']//*[local-name()='v']"
                if ($chartText.Count -gt 0) {
                    "[chart labels] $($chartText -join ' | ')"
                }
            }
        }
    }
    finally {
        $archive.Dispose()
    }
}
