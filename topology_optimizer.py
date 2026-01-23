"""
Optymalizacja topologii sieci przez wyłączanie linii
Uruchamiać PO optymalizacji generatorów
"""

import sys
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Ścieżki
SCRIPT_DIR = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Wysokie napiecie\PYTHON"
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from run_optimization import (
    Logger, log, find_element_multi_method,
    load_export_config, collect_results_parametrized, save_results_to_excel,
    OUT_DIR, EXCEL_FILE, PROJECT_NAME, USER
)

from PSO import PSO

# ==========================================
# KONFIGURACJA
# ==========================================

# Folder na wyniki TOPOLOGII (osobny!)
TOPOLOGY_OUT_DIR = os.path.join(OUT_DIR, "Topologia")

# Utwórz folder jeśli nie istnieje
if not os.path.exists(TOPOLOGY_OUT_DIR):
    try:
        os.makedirs(TOPOLOGY_OUT_DIR)
        print(f"✅ Utworzono folder:{TOPOLOGY_OUT_DIR}")
    except Exception as e:
        print(f"⚠️ Nie można utworzyć folderu:{e}")
        TOPOLOGY_OUT_DIR = OUT_DIR  # Fallback do głównego folderu

RECONFIG_PARAMS = {
    'n_particles':10,      # PSO:ilość cząstek
    'max_iter':5,          # PSO:iteracje
    'w':0.7,
    'c1':1.5,
    'c2':1.5,
    'autosave_every':5,
}

# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================

def load_reconfiguration_config(excel_file):
    """Wczytaj konfigurację rekonfiguracji - WERSJA NAPRAWIONA"""
    
    print("\n" + "="*80)
    print("🔍 WCZYTYWANIE KONFIGURACJI REKONFIGURACJI")
    print("="*80)
    
    try:
        df = pd.read_excel(excel_file, sheet_name="Rekonfiguracja")
        
        print(f"✓ Arkusz wczytany:{len(df)} wierszy")
        print(f"  Kolumny:{list(df.columns)}")
        
        # === WCZYTAJ LINIE ===
        if 'Line_Name' not in df.columns or 'Can_Disable' not in df.columns:
            raise ValueError("Brak kolumn Line_Name lub Can_Disable")
        
        # Filtruj Can_Disable = 1
        candidate_df = df[df['Can_Disable'] == 1]
        
        # KONWERTUJ DO STRING! 
        candidate_lines = candidate_df['Line_Name'].astype(str).tolist()
        
        # Usuń NaN, puste, 'nan'
        candidate_lines = [
            x.strip() for x in candidate_lines 
            if x and x.lower() != 'nan' and str(x).strip()
        ]
        
        # Priority (opcjonalnie)
        if 'Priority' in df.columns:
            priorities = candidate_df['Priority'].tolist()
            sorted_pairs = sorted(
                zip(candidate_lines, priorities), 
                key=lambda x:x[1] if pd.notna(x[1]) else 999
            )
            candidate_lines = [line for line, _ in sorted_pairs]
            print(f"  ✓ Linie posortowane według priorytetu")
        
        # === PARAMETRY ===
        max_lines_out = 3
        min_lines_out = 1
        
        if 'Parameter' in df.columns and 'Value' in df.columns:
            params = df[['Parameter', 'Value']].dropna()
            
            for _, row in params.iterrows():
                param = str(row['Parameter']).strip()
                value = row['Value']
                
                if param == 'Max_Lines_Out':
                    max_lines_out = int(value)
                elif param == 'Min_Lines_Out':
                    min_lines_out = int(value)
        else:
            print("  ⚠️ Brak parametrów - domyślne (Max=3, Min=1)")
        
        print(f"\n✅ Konfiguracja rekonfiguracji:")
        print(f"   Linii kandydujących:{len(candidate_lines)}")
        print(f"   Max do wyłączenia:{max_lines_out}")
        print(f"   Min do wyłączenia:{min_lines_out}")
        print(f"   Lista linii:")
        for i, line in enumerate(candidate_lines[:10], 1):
            print(f"      {i}.'{line}' (type:{type(line).__name__})")
        if len(candidate_lines) > 10:
            print(f"      ...i {len(candidate_lines) - 10} więcej")
        print("="*80)
        
        return candidate_lines, max_lines_out, min_lines_out
    
    except Exception as e:
        print(f"❌ Błąd wczytywania Rekonfiguracja:{e}")
        import traceback
        traceback.print_exc()
        return [], 3, 1

def check_island_formation(app, ldf):
    """
    Sprawdź czy nie powstały wyspy - WERSJA DEBUG
    """
    
    print("    [ISLAND_CHECK] Sprawdzam wyspy...")
    
    try:
        print("    [ISLAND_CHECK] Uruchamianie LF...")
        code = ldf.Execute()
        
        print(f"    [ISLAND_CHECK] LF code:{code}")
        
        if code != 0:
            print(f"    [ISLAND_CHECK] LF nie zbiegł → WYSPA")
            return True
        
        print("    [ISLAND_CHECK] LF OK - sprawdzam napięcia...")
        
        buses = app.GetCalcRelevantObjects("*.ElmTerm")
        total_buses = len(buses)
        
        print(f"    [ISLAND_CHECK] Liczba węzłów:{total_buses}")
        
        low_voltage_count = 0
        zero_voltage_buses = []
        
        for bus in buses:
            try:
                u_pu = bus.GetAttribute("m:u")
                if u_pu is None or u_pu < 0.01:
                    low_voltage_count += 1
                    zero_voltage_buses.append(bus.loc_name)
            except:
                pass
        
        print(f"    [ISLAND_CHECK] Węzłów z U≈0:{low_voltage_count}/{total_buses}")
        
        if low_voltage_count > 0:
            print(f"    [ISLAND_CHECK] Przykłady:{', '.join(zero_voltage_buses[:5])}")
        
        threshold = total_buses * 0.05
        print(f"    [ISLAND_CHECK] Próg (5%):{threshold:.1f}")
        
        if low_voltage_count > threshold:
            print(f"    [ISLAND_CHECK] {low_voltage_count} > {threshold:.1f} → WYSPA")
            return True
        
        print(f"    [ISLAND_CHECK] OK - brak wysp")
        return False
    
    except Exception as e:
        print(f"    [ISLAND_CHECK] EXCEPTION:{e}")
        import traceback
        traceback.print_exc()
        return True

# ==========================================
# KLASA FUNKCJI CELU DLA REKONFIGURACJI
# ==========================================

class TopologyObjective:
    """
    Funkcja celu:optymalizacja topologii przez wyłączanie linii
    x = wektor binarny [0/1] dla każdej linii kandydującej
    """
    
    def __init__(self, app, ldf, candidate_lines, max_lines_out, min_lines_out, base_objective_func, debug_file=None):
        """
        app:PowerFactory application
        ldf:Load Flow object
        candidate_lines:lista nazw linii (stringi)
        max_lines_out:max liczba linii do wyłączenia
        min_lines_out:min liczba linii do wyłączenia
        base_objective_func:funkcja obliczająca f_celu (np.przeciążenia)
        debug_file:plik do logowania
        """
        self.app = app
        self.ldf = ldf
        self.candidate_lines = candidate_lines
        self.max_lines_out = max_lines_out
        self.min_lines_out = min_lines_out
        self.base_objective = base_objective_func
        self.debug_file = debug_file
        
        self.eval_count = 0
        self.best_value = np.inf
        self.best_config = None
        
        # Cache linii
        self._line_cache = {}
        self._cache_lines()
    
    def _cache_lines(self):
        """Znajdź i zachowaj referencje do linii"""
        print(f"\n🔍 Cachowanie {len(self.candidate_lines)} linii...")
        
        cached_count = 0
        for line_name in self.candidate_lines:
            line = find_element_multi_method(self.app, line_name, "ElmLne")
            if line:
                self._line_cache[line_name] = line
                cached_count += 1
            else:
                print(f"  ⚠️ Nie znaleziono:{line_name}")
        
        print(f"✅ Cached:{cached_count}/{len(self.candidate_lines)} linii")
    
    def _decode_binary_vector(self, x):
        """
        Dekoduj wektor x na listę linii do wyłączenia
        x[i] > 0.5 → linia i wyłączona
        """
        lines_to_disable = []
        
        for i, val in enumerate(x):
            if i < len(self.candidate_lines):
                if val > 0.5:# Traktuj jako binarny
                    lines_to_disable.append(self.candidate_lines[i])
        
        # Ogranicz do max_lines_out
        if len(lines_to_disable) > self.max_lines_out:
            lines_to_disable = lines_to_disable[:self.max_lines_out]
        
        # Wymuszenie min_lines_out (jeśli za mało, dodaj losowe)
        if len(lines_to_disable) < self.min_lines_out:
            available = [ln for ln in self.candidate_lines if ln not in lines_to_disable]
            needed = self.min_lines_out - len(lines_to_disable)
            if len(available) >= needed:
                import random
                additional = random.sample(available, needed)
                lines_to_disable.extend(additional)
        
        return lines_to_disable
    
    def _set_lines_state(self, lines_to_disable, state):
        """
        Ustaw stan linii (outserv)
        state:1 = wyłącz, 0 = włącz
        """
        for line_name in lines_to_disable:
            line = self._line_cache.get(line_name)
            if line:
                try:
                    line.outserv = state
                except Exception as e:
                    print(f"  ⚠️ Błąd ustawienia {line_name}:{e}")
    
    def __call__(self, x):
        """Główna funkcja celu"""
        self.eval_count += 1
        
        try:
            msg = f"\n{'='*60}\n🔧 TOPO EVAL #{self.eval_count}\n{'='*60}"
            print(msg)
            
            if self.debug_file:
                with open(self.debug_file, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
            
            # Dekoduj które linie wyłączyć
            lines_to_disable = self._decode_binary_vector(x)
            
            print(f"  Wyłączam {len(lines_to_disable)} linii:")
            for ln in lines_to_disable:
                print(f"    - {str(ln)}")
            
            if self.debug_file:
                with open(self.debug_file, 'a', encoding='utf-8') as f:
                    lines_str = [str(ln) for ln in lines_to_disable]
                    f.write(f"  Lines OUT:{', '.join(lines_str)}\n")
            
            # Wyłącz linie
            self._set_lines_state(lines_to_disable, state=1)
            
            # Sprawdź wyspy
            is_island = check_island_formation(self.app, self.ldf)
            
            if is_island:
                island_msg = "  ⚠️ WYSPA wykryta → f=inf"
                print(island_msg)
                
                if self.debug_file:
                    with open(self.debug_file, 'a', encoding='utf-8') as f:
                        f.write(island_msg + "\n")
                
                # Przywróć linie
                self._set_lines_state(lines_to_disable, state=0)
                return np.inf
            
            # Oblicz f_celu
            f_value = self.base_objective(None)
            
            result_msg = f"  → f_total = {f_value:.6f}"
            print(result_msg)
            
            if self.debug_file:
                with open(self.debug_file, 'a', encoding='utf-8') as f:
                    f.write(result_msg + "\n")
            
            # Best tracking
            if f_value < self.best_value:
                self.best_value = f_value
                self.best_config = lines_to_disable.copy()
                
                lines_str = [str(ln) for ln in lines_to_disable]
                best_msg = f"  ✅ NEW BEST:{f_value:.6f}, Lines OUT:{', '.join(lines_str)}"
                print(best_msg)
                
                if self.debug_file:
                    with open(self.debug_file, 'a', encoding='utf-8') as f:
                        f.write(best_msg + "\n")
            
            # Przywróć linie
            self._set_lines_state(lines_to_disable, state=0)
            
            return f_value
        
        except Exception as e:
            error_msg = f"\n❌ EXCEPTION:{e}"
            print(error_msg)
            
            import traceback
            tb = traceback.format_exc()
            print(tb)
            
            if self.debug_file:
                with open(self.debug_file, 'a', encoding='utf-8') as f:
                    f.write(error_msg + "\n")
                    f.write(tb + "\n")
            
            # Przywróć linie
            try:
                self._set_lines_state(self.candidate_lines, state=0)
            except:
                pass
            
            return np.inf

# ==========================================
# GŁÓWNA FUNKCJA OPTYMALIZACJI TOPOLOGII
# ==========================================

def run_topology_optimization(app, ldf, excel_file, out_dir, scenario_name="TOPO"):
    """
    Uruchom optymalizację topologii
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topo_dir = TOPOLOGY_OUT_DIR
    log_file = os.path.join(topo_dir, f"topology_log_{timestamp}.txt")
    
    global _logger
    _logger = Logger(log_file, app)
    
    try:
        log("="*80)
        log("OPTYMALIZACJA TOPOLOGII SIECI")
        log("="*80)
        log(f"Scenariusz:{scenario_name}")
        log(f"Folder wyników:{topo_dir}")
        log("="*80)
        
        # Wczytaj konfigurację
        candidate_lines, max_lines_out, min_lines_out = load_reconfiguration_config(excel_file)
        
        if not candidate_lines:
            log("❌ Brak linii kandydujących - przerywam")
            return None
        
        # WŁĄCZ WSZYSTKIE LINIE (reset stanu)
        log("\n🔄 Resetowanie stanu sieci (włączanie wszystkich linii)...")
        all_lines = app.GetCalcRelevantObjects("*.ElmLne")
        enabled_count = 0
        for line in all_lines:
            try:
                line.outserv = 0
                enabled_count += 1
            except:
                pass
        log(f"✓ Włączono {enabled_count}/{len(all_lines)} linii")
        
        # Test Load Flow
        log("\n🔍 Test Load Flow przed optymalizacją...")
        code = ldf.Execute()
        if code == 0:
            log("✓ Load Flow OK")
        else:
            log(f"⚠️ Load Flow błąd:{code} - kontynuuję mimo to")
        
        # === PRZYGOTUJ FUNKCJĘ BAZOWĄ z DEBUG ===
        def base_objective_func(x_dummy):
            """Oblicz przeciążenia w obecnym stanie sieci - WERSJA DEBUG"""
            try:
                print("\n    [BASE_OBJ] Uruchamianie Load Flow...")
                code = ldf.Execute()
                
                print(f"    [BASE_OBJ] LF code:{code}")
                
                if code != 0:
                    print(f"    [BASE_OBJ] LF nie zbiegł → inf")
                    return np.inf
                
                print(f"    [BASE_OBJ] LF OK - obliczam przeciążenia...")
                
                overload = 0.0
                lines = app.GetCalcRelevantObjects("*.ElmLne")
                
                # UŻYJ NAZW zamiast indeksów! 
                observed_line_names = candidate_lines  # Obserwuj te same linie które kandydują
                
                print(f"    [BASE_OBJ] Liczba linii:{len(lines)}")
                print(f"    [BASE_OBJ] Obserwowane nazwy:{observed_line_names[:5]}...")
                
                overloaded_lines = []
                
                for line in lines:
                    if line.loc_name in observed_line_names:
                        try:
                            loading = line.GetAttribute("c:loading")
                            
                            print(f"    [BASE_OBJ]   {line.loc_name}:{loading:.2f}%")
                            
                            if loading and loading > 100:
                                excess = loading - 100
                                overload += excess
                                overloaded_lines.append(f"{line.loc_name}:{loading:.1f}%")
                        except Exception as e:
                            print(f"    [BASE_OBJ]   {line.loc_name}:ERROR - {e}")
                
                print(f"    [BASE_OBJ] Suma przeciążeń:{overload:.3f}")
                if overloaded_lines:
                    print(f"    [BASE_OBJ] Przeciążone:{', '.join(overloaded_lines)}")
                else:
                    print(f"    [BASE_OBJ] Brak przeciążeń")
                
                return overload
            
            except Exception as e:
                print(f"    [BASE_OBJ] EXCEPTION:{e}")
                import traceback
                traceback.print_exc()
                return np.inf
        
        # Debug file
        debug_file = os.path.join(topo_dir, f"DEBUG_TOPOLOGY_{timestamp}.txt")
        
        # === FUNKCJA CELU TOPOLOGII ===
        topo_objective = TopologyObjective(
            app, ldf,
            candidate_lines,
            max_lines_out,
            min_lines_out,
            base_objective_func,
            debug_file=debug_file
        )
        
        # PSO
        dim = len(candidate_lines)
        lb = np.zeros(dim)
        ub = np.ones(dim)
        
        log(f"\nUruchamianie PSO:")
        log(f"  Zmiennych (linii):{dim}")
        log(f"  Cząstek:{RECONFIG_PARAMS['n_particles']}")
        log(f"  Iteracji:{RECONFIG_PARAMS['max_iter']}")
        
        checkpoint_path = os.path.join(topo_dir, f"topo_checkpoint_{timestamp}.npz")
        
        pso = PSO(
            func=topo_objective,
            n_particles=RECONFIG_PARAMS['n_particles'],
            dim=dim,
            lb=lb,
            ub=ub,
            max_iter=RECONFIG_PARAMS['max_iter'],
            w=RECONFIG_PARAMS['w'],
            c1=RECONFIG_PARAMS['c1'],
            c2=RECONFIG_PARAMS['c2'],
            autosave_every_iters=RECONFIG_PARAMS['autosave_every'],
            autosave_path=checkpoint_path,
            early_stop_threshold=0.0,      # ✅ NOWE
            early_stop_patience=1000         # ✅ NOWE - mniej bo topologia ma mniej iteracji
        )
        
        time_start = time.time()
        result = pso.optimize()
        time_end = time.time()
        
        # ✅ Informacja o early stopping
        if result.get('early_stopped'):
            log(f"\n⏸️  Early stop:{result.get('reason')}")
            log(f"   At iteration:{result.get('stopped_at_iter')}")
        
        log(f"\n✅ Optymalizacja zakończona w {time_end - time_start:.2f}s")
        log(f"Najlepsza wartość:{result['gbest_val']:.6f}")
        log(f"Najlepsza konfiguracja:")
        
        best_lines = topo_objective._decode_binary_vector(result['gbest'])
        for ln in best_lines:
            log(f"  - {ln} (OUT)")
        
        # Zapisz wyniki
        output_file = os.path.join(topo_dir, f"TOPOLOGY_{scenario_name}_{timestamp}.xlsx")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Podsumowanie
            summary = {
                'Scenariusz':[scenario_name],
                'Timestamp':[timestamp],
                'F_best':[result['gbest_val']],
                'Lines_disabled':[', '.join([str(ln) for ln in best_lines])],
                'Num_lines_disabled':[len(best_lines)],
            }
            pd.DataFrame(summary).to_excel(writer, sheet_name='Podsumowanie', index=False)
            
            # Historia
            history = pd.DataFrame({
                'Iteration':range(len(result['best_per_iter'])),
                'Best_Value':result['best_per_iter']
            })
            history.to_excel(writer, sheet_name='History', index=False)
            
            # Status linii
            lines_status = []
            for line_name in candidate_lines:
                status = 'DISABLED' if line_name in best_lines else 'ACTIVE'
                lines_status.append({
                    'Line_Name':line_name,
                    'Status':status
                })
            pd.DataFrame(lines_status).to_excel(writer, sheet_name='Lines_Status', index=False)
        
        log(f"\n✅ Wyniki zapisane:{output_file}")
        log(f"✅ Folder wyników:{topo_dir}")
        
        return result
    
    except Exception as e:
        log(f"\n❌ BŁĄD w run_topology_optimization():")
        log(f"  {e}")
        import traceback
        log(traceback.format_exc())
        return None
    
    finally:
        if _logger:
            _logger.close()

# ==========================================
# MAIN (do uruchamiania z PF)
# ==========================================

def main():
    """Główna funkcja z pełną obsługą błędów"""
    
    print("\n" + "="*80)
    print("🚀 URUCHAMIANIE TOPOLOGY_OPTIMIZER.PY")
    print("="*80)
    
    try:
        # === KROK 1:Import PowerFactory ===
        print("\n[1/7] Import PowerFactory...")
        try:
            import powerfactory
            print("  ✓ Moduł powerfactory zaimportowany")
        except ImportError as e:
            print(f"  ❌ Błąd importu powerfactory:{e}")
            print("  Sprawdź czy skrypt uruchamiasz z PowerFactory!")
            return
        
        # === KROK 2:Połączenie z PF ===
        print("\n[2/7] Połączenie z PowerFactory...")
        try:
            app = powerfactory.GetApplicationExt()
            if app is None:
                print("  ❌ Nie można połączyć z PowerFactory")
                print("  Upewnij się, że skrypt uruchamiasz z poziomu PF (Execute Python Script)")
                return
            print(f"  ✓ Połączono z PowerFactory")
        except Exception as e:
            print(f"  ❌ Błąd połączenia:{e}")
            return
        
        # === KROK 3:Aktywacja projektu ===
        print(f"\n[3/7] Aktywacja projektu '{PROJECT_NAME}'...")
        try:
            prj = app.GetActiveProject()
            
            if prj is None:
                print(f"  ⚠️ Brak aktywnego projektu - próbuję aktywować {PROJECT_NAME}")
                app.ActivateProject(PROJECT_NAME)
                prj = app.GetActiveProject()
            
            if prj is None:
                print(f"  ❌ Nie można aktywować projektu {PROJECT_NAME}")
                print(f"  Sprawdź nazwę projektu w konfiguracji")
                return
            
            print(f"  ✓ Projekt aktywny:{prj.loc_name}")
            
            if prj.loc_name != PROJECT_NAME:
                print(f"  ⚠️ Aktywny projekt '{prj.loc_name}' różni się od '{PROJECT_NAME}'")
                print(f"  Kontynuuję z projektem '{prj.loc_name}'")
        
        except Exception as e:
            print(f"  ❌ Błąd aktywacji projektu:{e}")
            import traceback
            traceback.print_exc()
            return
        
        # === KROK 4:Load Flow ===
        print("\n[4/7] Pobieranie obiektu Load Flow...")
        try:
            ldf = app.GetFromStudyCase("ComLdf")
            if ldf is None:
                print("  ❌ Nie znaleziono obiektu ComLdf (Load Flow)")
                print("  Sprawdź czy w Study Case jest Load Flow Calculation")
                return
            print(f"  ✓ Load Flow znaleziony")
        except Exception as e:
            print(f"  ❌ Błąd pobierania Load Flow:{e}")
            return
        
        # === KROK 5:Sprawdź pliki ===
        print(f"\n[5/7] Sprawdzanie plików...")
        print(f"  Excel:{EXCEL_FILE}")
        
        if not os.path.exists(EXCEL_FILE):
            print(f"  ❌ Plik Excel nie istnieje!")
            return
        print(f"  ✓ Plik Excel istnieje")
        
        print(f"  Katalog wyników:{OUT_DIR}")
        if not os.path.exists(OUT_DIR):
            print(f"  ⚠️ Katalog nie istnieje - tworzę...")
            try:
                os.makedirs(OUT_DIR)
                print(f"  ✓ Katalog utworzony")
            except Exception as e:
                print(f"  ❌ Nie można utworzyć katalogu:{e}")
                return
        else:
            print(f"  ✓ Katalog istnieje")
        
        # === KROK 6:Uruchomienie optymalizacji ===
        print("\n[6/7] Uruchamianie optymalizacji topologii...")
        
        scenario_name = "N1_Topology"
        
        try:
            result = run_topology_optimization(
                app, ldf, EXCEL_FILE, OUT_DIR, scenario_name
            )
            
            if result:
                print("\n[7/7] ✅ SUKCES")
                print(f"  Najlepsza wartość:{result['gbest_val']:.6f}")
                print(f"  Wyniki w:{TOPOLOGY_OUT_DIR}")
            else:
                print("\n[7/7] ⚠️ Optymalizacja nie zwróciła wyniku")
                print("  Sprawdź log w folderze Wyniki/Topologia/")
        
        except Exception as e:
            print(f"\n❌ BŁĄD podczas optymalizacji:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"\n❌ KRYTYCZNY BŁĄD w main():")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n" + "="*80)
        print("Skrypt zakończony")
        print("="*80)

if __name__ == "__main__":
    main()