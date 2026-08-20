Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$resultFiles = Get-ChildItem -Path (Join-Path $PSScriptRoot "..\output") -Recurse -Filter "results.csv"

foreach ($resultFile in $resultFiles) {
    $destination = Join-Path $resultFile.DirectoryName "results_closest_with_vehicle_type.csv"

    Import-Csv -LiteralPath $resultFile.FullName |
        Select-Object `
            @{Name = "vehicle_id"; Expression = { $_.track_id }},
            vehicle_type,
            class,
            closest_frame,
            timestamp_sec,
            timestamp_hms,
            x1,
            y1,
            x2,
            y2,
            center_x,
            center_y |
        Export-Csv -LiteralPath $destination -NoTypeInformation -Encoding utf8

    Write-Output $destination
}
