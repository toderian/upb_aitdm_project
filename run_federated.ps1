# Pornește serverul
Write-Host "Pornesc serverul..."
$server = Start-Process -FilePath "python" -ArgumentList "-m model_side.federated.server --num_rounds 1 --num_clients 3 --local_epochs 2" -PassThru

# Așteaptă ca serverul să se inițializeze
Start-Sleep -Seconds 3

# Pornește cei 3 clienți
$clients = @()
for ($i = 0; $i -lt 3; $i++) {
    Write-Host "Pornesc clientul $i..."
    $client = Start-Process -FilePath "python" -ArgumentList "-m model_side.federated.client --client_id $i --server_address 127.0.0.1:8080" -PassThru
    $clients += $client
}

# Așteaptă ca serverul să termine
Wait-Process -Id $server.Id

# Când serverul se oprește, închide clienții
foreach ($client in $clients) {
    Write-Host "Oprire client $($client.Id)..."
    Stop-Process -Id $client.Id -Force
}

Write-Host "Toate procesele au fost oprite."
