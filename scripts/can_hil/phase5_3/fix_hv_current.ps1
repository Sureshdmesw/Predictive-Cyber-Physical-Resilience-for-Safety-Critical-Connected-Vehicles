$ErrorActionPreference = "Stop"

Set-Location "E:\Predictive Cyber-Physical Resilience for Safety-Critical Connected Vehicles"

Write-Host ""
Write-Host "=== PHASE 5.3 HV CURRENT FIX ===" -ForegroundColor Cyan

$path = "data\schemas\can_hil\ev\virtual_ev.dbc"

$text = [System.IO.File]::ReadAllText(
    (Resolve-Path $path),
    [System.Text.Encoding]::UTF8
)

$old = 'SG_ HVCurrent : 32|16@1- (0.1,-3276.8) [-3276.8|3276.7] "A" BMS'

$new = 'SG_ HVCurrent : 32|16@1- (0.1,0) [-3276.8|3276.7] "A" BMS'

if (-not $text.Contains($old)) {
    Write-Host "Expected HVCurrent definition not found." -ForegroundColor Red
    Write-Host "The DBC may already be corrected." -ForegroundColor Yellow
}
else {
    $text = $text.Replace($old, $new)

    $encoding = New-Object System.Text.UTF8Encoding($false)

    [System.IO.File]::WriteAllText(
        (Resolve-Path $path),
        $text,
        $encoding
    )

    Write-Host "HVCurrent corrected." -ForegroundColor Green
}

Write-Host ""
Write-Host "=== TESTING DBC ===" -ForegroundColor Cyan

python -c "import cantools; db=cantools.database.load_file(r'data\schemas\can_hil\ev\virtual_ev.dbc'); m=db.get_message_by_name('BATTERY_STATUS'); x=m.encode({'BatterySOC':80,'BatterySOH':98,'HVVoltage':400,'HVCurrent':80,'BatteryTemperature':28,'BMSState':2}); print('BATTERY_STATUS encode PASS:', len(x), 'bytes'); print('Decoded:', m.decode(x))"

if ($LASTEXITCODE -ne 0) {
    throw "BATTERY_STATUS validation failed."
}

Write-Host ""
Write-Host "=== HV CURRENT FIX PASSED ===" -ForegroundColor Green
Write-Host "The PowerShell window will remain open."
Write-Host ""