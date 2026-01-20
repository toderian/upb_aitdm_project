# Script PowerShell pentru rularea serverului și clienților secvențial

$ServerCmd = "python -m model_side.federated.server --num_rounds 1 --num_clients 1 --local_epochs 2"
$ClientCmdBase = "python -m model_side.federated.client --server_address 127.0.0.1:8080 --client_id"
$ClientIDs = @(0, 1, 2)

Write-Host "`n===== PORNESC SERVERUL =====`n"

# Pornire server în background
$ServerProcess = Start-Process powershell -ArgumentList "-NoLogo -NoProfile -Command $ServerCmd" -PassThru

Write-Host "Aștept serverul să se inițializeze..."
Start-Sleep -Seconds 3

Write-Host "`n===== PORNESC CLIENȚII SECVENȚIAL =====`n"

foreach ($cid in $ClientIDs) {
    $ClientCmd = "$ClientCmdBase $cid"

    Write-Host "`n=== Pornesc clientul $cid ===`n"

    # Pornire client și așteptare finalizare
    $ClientProcess = Start-Process powershell -ArgumentList "-NoLogo -NoProfile -Command $ClientCmd" -Wait

    Write-Host "`n=== Clientul $cid a terminat execuția ===`n"
    Start-Sleep -Seconds 1
}

Write-Host "`n===== Toți clienții au rulat ====="
Write-Host "Închid serverul...`n"

# Terminăm serverul
if ($ServerProcess -ne $null) {
    Stop-Process -Id $ServerProcess.Id -Force
}

Write-Host "Server oprit.`n"
