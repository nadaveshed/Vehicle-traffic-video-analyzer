Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$source = Join-Path $PSScriptRoot "..\output\job_0002\results.csv"
$destination = Join-Path $PSScriptRoot "..\output\job_0002\results_closest.csv"

Import-Csv -LiteralPath $source |
    Where-Object { $_.entered_zone -eq "True" -and $_.exits_view -eq "True" } |
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
