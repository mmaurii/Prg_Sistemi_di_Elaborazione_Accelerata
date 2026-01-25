import subprocess
import os
import time

# ================= CONFIGURAZIONE =================
EXECUTABLES_CONFIG = [
    # (Nome Eseguibile, Ignorato_in_questa_versione)
    ("./main_naif_gpu_path_dependent_1", True),
    ("./main_gpu_path_dependent_2", True),
    ("./main_gpu_path_dependent_3", True),
    ("./main_naif_gpu_0", True),
    ("./main_naif_gpu_1", True),
    ("./main_gpu_3", True),
    ("./main_gpu_4", True),
    ("./main_gpu_5", True),
    ("./main_gpu_6", True),
    ("./main_gpu_7", True),
    ("./main_gpu_8", True),
    ("./main_gpu_9", True),
    ("./main_gpu_10", True),
    ("./main_gpu_11", True),
    ("./main_naif_cpu_0", False),
    ("./main_naif_cpu_1", False),
    ("./main_naif_cpu_2", False),
    ("./main_naif_cpu_path_dependent_1", False),
    ("./main_simd", False)
]

POW_START = 5  # 10^5
POW_END = 9    # 10^9
OUTPUT_DIR = "benchmark_results_native" # Ho cambiato nome cartella per non mischiare con nsys

# Limite di tempo in secondi per l'esecuzione
TIMEOUT_SEC = 120 
# ==================================================

def run_benchmark():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Cartella risultati: {OUTPUT_DIR}")
        print(f"⏱️  Timeout impostato a: {TIMEOUT_SEC} secondi")

    for p in range(POW_START, POW_END + 1):
        n_sim = 10**p
        print(f"\n{'='*60}")
        print(f"🚀 N_SIMULATIONS = 10^{p} ({n_sim})")
        print(f"{'='*60}")

        # Nota: '_' ignora il booleano True/False che non serve più
        for exe_path, _ in EXECUTABLES_CONFIG:
            exe_name = os.path.basename(exe_path)
            
            if not os.path.isfile(exe_path):
                print(f"❌ Errore: Eseguibile '{exe_path}' non trovato.")
                continue

            base_output_name = f"{exe_name}_1e{p}"
            log_path = os.path.join(OUTPUT_DIR, f"{base_output_name}_LOG.txt")

            # --- CONTROLLO SE IL FILE ESISTE GIÀ ---
            if os.path.exists(log_path):
                print(f"   ⏩ {exe_name}: Log presente. Skippo.")
                continue
            
            print(f"   ▶️  Eseguendo: {exe_name}...", end=" ", flush=True)

            # Comando di esecuzione diretta
            cmd = [exe_path, str(n_sim)]

            with open(log_path, "w") as log_file:
                log_file.write(f"=== NATIVE RUN: {exe_name} @ 10^{p} ===\n\n")
                
                start_time_python = time.time()
                
                try:
                    # Esecuzione pura dell'eseguibile
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=TIMEOUT_SEC
                    )
                    
                    elapsed_python = time.time() - start_time_python

                    # Scriviamo Output (std::cout) e Errori (std::cerr) nel file
                    log_file.write(result.stdout)
                    if result.stderr:
                        log_file.write("\n--- STDERR ---\n")
                        log_file.write(result.stderr)

                    if result.returncode == 0:
                        print(f"✅ Fatto ({elapsed_python:.2f}s)")
                        # Aggiungiamo un footer con il tempo misurato da Python per verifica
                        log_file.write(f"\n\n[Python Timer]: Total Execution Time: {elapsed_python:.4f} s\n")
                    else:
                        print(f"⚠️ Errore (Code: {result.returncode})")
                        log_file.write(f"\n!!! PROCESSO TERMINATO CON CODICE {result.returncode} !!!\n")

                except subprocess.TimeoutExpired:
                    print(f"⏰ TIMEOUT!")
                    log_file.write(f"\n\n!!! ABORTITO: TIMEOUT DI {TIMEOUT_SEC} SECONDI RAGGIUNTO !!!\n")
                
                except Exception as e:
                    print(f"❌ Exception!")
                    log_file.write(f"\n\n!!! ERRORE PYTHON: {e} !!!\n")

    print(f"\n🎉 TUTTI I TEST COMPLETATI. Risultati in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    # Assicura permessi di esecuzione
    for exe, _ in EXECUTABLES_CONFIG:
        if os.path.exists(exe):
            os.chmod(exe, 0o755)
    run_benchmark()