import subprocess
import os
import time

# ================= CONFIGURAZIONE =================
EXECUTABLES_CONFIG = [
    # (Nome Eseguibile, Is_GPU_Code)
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
    ]

POW_START = 5  # 10^5
POW_END = 9    # 10^9
OUTPUT_DIR = "benchmark_results"

# Limite di tempo in secondi per OGNI singolo comando (nsys o ncu)
TIMEOUT_SEC = 60* 6
# ==================================================

def run_benchmark():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Cartella risultati: {OUTPUT_DIR}")
        print(f"⏱️  Timeout impostato a: {TIMEOUT_SEC} secondi per operazione")

    for p in range(POW_START, POW_END + 1):
        n_sim = 10**p
        print(f"\n{'='*60}")
        print(f"🚀 N_SIMULATIONS = 10^{p} ({n_sim})")
        print(f"{'='*60}")

        for exe_path, is_gpu in EXECUTABLES_CONFIG:
            exe_name = os.path.basename(exe_path)
            
            if not os.path.isfile(exe_path):
                print(f"❌ Errore: Eseguibile '{exe_path}' non trovato.")
                continue

            base_output_name = f"{exe_name}_1e{p}"
            full_rep_path = os.path.join(OUTPUT_DIR, base_output_name)
            log_path = os.path.join(OUTPUT_DIR, f"{base_output_name}_LOG.txt")

            # --- 1. CONTROLLO SE IL FILE ESISTE GIÀ ---
            if os.path.exists(log_path):
                print(f"   ⏩ {exe_name}: Risultati già presenti. Skippo.")
                continue
            
            print(f"\n🔹 Benchmarking: {exe_name}")
            
            # Flag per sapere se dobbiamo saltare NCU (es. se NSYS fallisce o va in timeout)
            skip_ncu_run = False

            with open(log_path, "w") as log_file:
                log_file.write(f"=== BENCHMARK REPORT: {exe_name} @ 10^{p} ===\n\n")

                # --- A. Esecuzione NSIGHT SYSTEMS ---
                print(f"   ⏳ Running nsys...")
                cmd_nsys = [
                    "nsys", "profile",
                    "--trace=cuda,osrt,nvtx",
                    "--stats=true",
                    "--force-overwrite=true",
                    "-o", full_rep_path,
                    exe_path, str(n_sim)
                ]
                
                log_file.write("--------------------------------------------------\n")
                log_file.write(">>> SEZIONE NSYS (Profilazione Sistema) <<<\n")
                log_file.write("--------------------------------------------------\n")
                
                try:
                    # Eseguiamo con TIMEOUT
                    result_nsys = subprocess.run(
                        cmd_nsys, 
                        capture_output=True, 
                        text=True, 
                        timeout=TIMEOUT_SEC
                    )
                    
                    log_file.write(result_nsys.stdout)
                    log_file.write(result_nsys.stderr)
                    
                    if result_nsys.returncode == 0:
                        print(f"   ✅ nsys completato.")
                    else:
                        print(f"   ⚠️ Errore nsys (vedi log).")
                        
                except subprocess.TimeoutExpired as e:
                    print(f"   ⏰ TIMEOUT nsys (> {TIMEOUT_SEC}s). Skippo questa run.")
                    log_file.write(f"\n\n!!! ABORTITO: TIMEOUT DI {TIMEOUT_SEC} SECONDI RAGGIUNTO !!!\n")
                    if e.stdout: log_file.write(e.stdout.decode('utf-8') if isinstance(e.stdout, bytes) else e.stdout)
                    if e.stderr: log_file.write(e.stderr.decode('utf-8') if isinstance(e.stderr, bytes) else e.stderr)
                    skip_ncu_run = True # Se nsys va in timeout, inutile provare ncu

                except Exception as e:
                    print(f"   ❌ Eccezione: {e}")
                    log_file.write(f"\n\n!!! ERRORE CRITICO: {e} !!!\n")
                    skip_ncu_run = True

                # --- B. Esecuzione NSIGHT COMPUTE ---
                if is_gpu and not skip_ncu_run:
                    print(f"   ⏳ Running ncu...")
                    
                    cmd_ncu = [
                        "ncu",
                        "--set", "detailed", # Usa 'detailed' per evitare timeout facili
                        "--force-overwrite",
                        "-o", full_rep_path,
                        exe_path, str(n_sim)
                    ]

                    log_file.write("\n\n")
                    log_file.write("--------------------------------------------------\n")
                    log_file.write(">>> SEZIONE NCU (Profilazione Kernel) <<<\n")
                    log_file.write("--------------------------------------------------\n")

                    try:
                        # Eseguiamo con TIMEOUT
                        result_ncu = subprocess.run(
                            cmd_ncu, 
                            capture_output=True, 
                            text=True, 
                            timeout=TIMEOUT_SEC
                        )
                        
                        log_file.write(result_ncu.stdout)
                        log_file.write(result_ncu.stderr)

                        if result_ncu.returncode == 0:
                            print(f"   ✅ ncu completato.")
                        else:
                            print(f"   ⚠️ Errore ncu (vedi log).")

                    except subprocess.TimeoutExpired as e:
                        print(f"   ⏰ TIMEOUT ncu (> {TIMEOUT_SEC}s).")
                        log_file.write(f"\n\n!!! ABORTITO: TIMEOUT DI {TIMEOUT_SEC} SECONDI RAGGIUNTO SU NCU !!!\n")
                        # Tentiamo di salvare quello che ha catturato prima di morire
                        if e.stdout: log_file.write(e.stdout.decode('utf-8') if isinstance(e.stdout, bytes) else e.stdout)
                
                elif is_gpu and skip_ncu_run:
                    print(f"   ⏭️  Skipping ncu (timeout precedente).")
                    log_file.write("\n\n>>> NCU SALTATO A CAUSA DI TIMEOUT/ERRORE SU NSYS <<<\n")
                
                else:
                    log_file.write("\n\n>>> NCU SALTATO (CPU ONLY) <<<\n")

    print(f"\n🎉 TUTTI I TEST COMPLETATI. Risultati in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    # Assicura permessi di esecuzione
    for exe, _ in EXECUTABLES_CONFIG:
        if os.path.exists(exe):
            os.chmod(exe, 0o755)
    run_benchmark()