import subprocess
import time
import sys

SERVER_CMD = [
    "python", "-m", "model_side.federated.server",
    "--num_rounds", "1",
    "--num_clients", "1",
    "--local_epochs", "2"
]

CLIENT_CMD_TEMPLATE = [
    "python", "-m", "model_side.federated.client",
    "--server_address", "127.0.0.1:8080",
    "--client_id"  # ultimul element va fi înlocuit cu ID-ul clientului
]

CLIENT_IDS = [0, 1, 2]  # lista clienților disponibili


def run_process(cmd):
    """Rulează un proces și așteaptă să se termine."""
    try:
        print(f"\n=== Pornesc: {' '.join(cmd)} ===\n")
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        print("Oprire manuală detectată. Închid procesele...")
        process.terminate()
        sys.exit(1)


def main():
    print("\n===== PORNESC SERVERUL =====\n")
    server_proc = subprocess.Popen(SERVER_CMD)

    # așteptăm ca serverul să pornească
    print("Aștept serverul să se inițializeze...")
    time.sleep(3)

    print("\n===== PORNESC CLIENȚII SECVENȚIAL =====\n")

    for cid in CLIENT_IDS:
        client_cmd = CLIENT_CMD_TEMPLATE + [str(cid)]
        run_process(client_cmd)
        print(f"\n=== Clientul {cid} a terminat execuția ===\n")
        time.sleep(1)

    print("\n===== Toți clienții au rulat =====")
    print("Închid serverul...")

    server_proc.terminate()
    print("Server oprit.\n")


if __name__ == "__main__":
    main()
