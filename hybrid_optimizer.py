"""
Hybrydowa optymalizacja:Generatory + Rekonfiguracja topologii (PSO + ranking linii)
- Wczytywanie danych bazowych z Excel
- Złagodzona weryfikacja Load Flow (≥70% linii z danymi)
- Early stopping (threshold + patience)
- Pełny eksport ustawień + obciążenia linii
"""

import sys
import os
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np

# === ŚCIEŻKI ===
SCRIPT_DIR = Path(r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Wysokie napiecie\PYTHON")
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from run_optimization import (
        Logger, find_element_multi_method, load_export_config, 
        collect_results_parametrized, save_results_to_excel,
        OUT_DIR, EXCEL_FILE, PROJECT_NAME, USER
    )
    from PSO import PSO
    print("✅ Moduły zaimportowane")
except ImportError as e:
    print(f"❌ Błąd importu:{e}")
    raise

# === KONFIGURACJA ===
HYBRID_OUT_DIR = Path(OUT_DIR) / "Hybrid"
HYBRID_OUT_DIR.mkdir(exist_ok=True)

@dataclass
class HybridConfig:
    n_particles:int = 10
    max_iter:int = 5
    w:float = 0.7
    c1:float = 1.5
    c2:float = 1.5
    autosave_every:int = 10
    early_stop_threshold:float = 0.0
    early_stop_patience:int = 100
    weight_topology:float = 0.0
    weight_overload:float = 1.0
    observed_lines:List[str] = None
    min_valid_lines_pct:float = 0.7  # 70%
    
    def __post_init__(self):
        if self.observed_lines is None:
            self.observed_lines = ['5.73', '8.14', '79.73', '35.77', '6.2', '73.71', '71.72']

CONFIG = HybridConfig()

# === LOGOWANIE ===
def safe_log(msg:str, logger:Optional[Logger] = None, app=None):
    """Jednolita funkcja logowania"""
    msg_str = str(msg)
    if logger:
        try:
            logger.write(msg_str + "\n")
        except:
            pass
    if app:
        try:
            app.PrintPlain(msg_str)
        except:
            pass
    print(msg_str)

# === WCZYTYWANIE DANYCH BAZOWYCH ===
def _load_elements_from_sheet(app, excel_file:str, sheet_name:str, elem_type:str, 
                               attr_p:str, attr_q:str) -> Tuple[int, int]:
    """Uniwersalna funkcja wczytująca elementy z arkusza Excel"""
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        set_count = 0
        not_found = []
        
        for _, row in df.iterrows():
            name = str(row["name"]).strip()
            elem = find_element_multi_method(app, name, elem_type)
            if elem:
                try:
                    elem.SetAttribute(attr_p, float(row["P"]))
                    elem.SetAttribute(attr_q, float(row["Q"]))
                    set_count += 1
                except Exception as e:
                    print(f"  ⚠️ {name}:błąd ustawienia - {e}")
            else:
                not_found.append(name)
        
        total = len(df)
        pct = 100 * set_count / max(1, total)
        print(f"  ✓ {elem_type}:{set_count}/{total} ({pct:.1f}%)")
        if not_found and len(not_found) <= 5:
            print(f"  ⚠️ Nie znaleziono:{', '.join(not_found[:5])}")
        elif len(not_found) > 5:
            print(f"  ⚠️ Nie znaleziono:{len(not_found)} elementów")
        
        return set_count, total
    except Exception as e:
        print(f"  ❌ Błąd wczytywania {sheet_name}:{e}")
        return 0, 0

def load_base_data_from_excel(app, excel_file:str) -> bool:
    """Wczytaj dane bazowe z Excela"""
    print("\n" + "="*80)
    print("📥 WCZYTYWANIE DANYCH BAZOWYCH Z EXCELA")
    print("="*80)
    print(f"Plik:{excel_file}")
    
    try:
        configs = [
            ("Loads", "ElmLod", "plini", "qlini"),
            ("Generators", "ElmSym", "pgini", "qgini"),
            ("PV", "ElmPvsys", "pgini", "qgini"),
            ("StatGen", "ElmGenstat", "pgini", "qgini"),
        ]
        
        for i, (sheet, elem_type, attr_p, attr_q) in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] {sheet} ({elem_type})...")
            _load_elements_from_sheet(app, excel_file, sheet, elem_type, attr_p, attr_q)
        
        print("\n" + "="*80)
        print("✅ DANE BAZOWE WCZYTANE")
        print("="*80)
        return True
    except Exception as e:
        print(f"\n❌ KRYTYCZNY BŁĄD:{e}")
        return False

# === KONFIGURACJA OPTYMALIZACJI ===
def load_hybrid_config(excel_file:str) -> Tuple[List[Dict], List[str], int, int]:
    """Wczytaj konfigurację optymalizacji"""
    print("\n" + "="*80)
    print("🔍 WCZYTYWANIE KONFIGURACJI OPTYMALIZACJI")
    print("="*80)
    
    try:
        # 1.Generatory do optymalizacji
        print("\n[1/2] Generatory do optymalizacji...")
        df_opt = pd.read_excel(excel_file, sheet_name="Optymalizacja")
        opt_variables = df_opt.to_dict('records')
        print(f"  ✓ {len(opt_variables)} elementów")
        
        # 2.Topologia (linie)
        print("\n[2/2] Topologia (linie do rekonfiguracji)...")
        df_reconfig = pd.read_excel(excel_file, sheet_name="Rekonfiguracja")
        
        candidate_df = df_reconfig[df_reconfig['Can_Disable'] == 1]
        candidate_lines = candidate_df['Line_Name'].astype(str).str.strip().tolist()
        candidate_lines = [x for x in candidate_lines if x and x.lower() != 'nan']
        
        # Sortowanie według priorytetów
        if 'Priority' in df_reconfig.columns:
            priorities = candidate_df['Priority'].tolist()
            sorted_pairs = sorted(zip(candidate_lines, priorities), key=lambda x:x[1] if pd.notna(x[1]) else 999)
            candidate_lines = [line for line, _ in sorted_pairs]
        
        print(f"  ✓ {len(candidate_lines)} linii kandydujących")
        
        # Min/Max z arkusza
        max_lines_out, min_lines_out = 3, 0
        if 'Parameter' in df_reconfig.columns and 'Value' in df_reconfig.columns:
            params = df_reconfig[['Parameter', 'Value']].dropna()
            for _, row in params.iterrows():
                param_name = str(row['Parameter']).strip()
                if param_name == 'Max_Lines_Out':
                    max_lines_out = int(row['Value'])
                elif param_name == 'Min_Lines_Out':
                    min_lines_out = int(row['Value'])
        
        print(f"  ✓ Zakres wyłączeń:Min={min_lines_out}, Max={max_lines_out}")
        print("="*80)
        
        return opt_variables, candidate_lines, min_lines_out, max_lines_out
    except Exception as e:
        print(f"❌ BŁĄD:{e}")
        return [], [], 0, 3

# === STATYSTYKI OBSERWOWANYCH LINII ===
def get_observed_lines_stats(app, ldf, observed_lines:List[str]) -> Dict:
    """Pobierz statystyki obserwowanych linii"""
    try:
        ldf.Execute()
        lines = app.GetCalcRelevantObjects("*.ElmLne")
        
        overloaded_count = 0
        total_overload = 0.0
        max_overload = 0.0
        max_line_name = ""
        overloads = []
        
        for line in lines:
            if line.loc_name in observed_lines:
                try:
                    loading = line.GetAttribute("c:loading")
                    if loading and loading > 100:
                        excess = loading - 100
                        overloads.append(excess)
                        total_overload += excess
                        overloaded_count += 1
                        if excess > max_overload:
                            max_overload = excess
                            max_line_name = line.loc_name
                except:
                    pass
        
        avg_overload = total_overload / len(overloads) if overloads else 0.0
        
        return {
            'total_overload':total_overload,
            'overloaded_count':overloaded_count,
            'max_overload':max_overload,
            'max_line_name':max_line_name,
            'avg_overload':avg_overload,
        }
    except:
        return {
            'total_overload':0.0, 'overloaded_count':0,
            'max_overload':0.0, 'max_line_name':'N/A', 'avg_overload':0.0,
        }

# === EKSPORT USTAWIEŃ ===
def _build_element_data(elements, elem_type:str, attrs:Dict[str, str]) -> List[Dict]:
    """Buduj dane dla elementów"""
    data = []
    for elem in elements:
        try:
            row = {
                'Element':elem.loc_name,
                'Type':elem_type,
                'Status':'OUT' if getattr(elem, 'outserv', 0) == 1 else 'IN',
            }
            for key, attr_name in attrs.items():
                try:
                    val = elem.GetAttribute(attr_name)
                    row[key] = f"{val:.2f}" if val is not None and key != 'Tap' else (val if val is not None else 'N/A')
                except:
                    row[key] = 'N/A'
            data.append(row)
        except:
            pass
    return data

def export_all_settings(app, ldf, phase="BEFORE") -> Dict:
    """Eksportuj ustawienia WSZYSTKICH elementów + obciążenia linii"""
    print(f"\n📋 Eksport ustawień - {phase}...")
    
    try:
        code = ldf.Execute()
        print(f"  ✓ Load Flow wykonany (code={code})")
        if code >= 2:
            print(f"  ❌ Load Flow NIE ZBIEGŁ (code={code})!")
        elif code == 1:
            print(f"  ⚠️ Load Flow z ostrzeżeniem (code=1)")
    except Exception as e:
        print(f"  ❌ Load Flow błąd:{e}")
    
    all_settings = {}
    
    # Konfiguracja typów elementów
    elem_configs = [
        ("ElmSym", "*.ElmSym", {'P [MW]':'pgini', 'Q [Mvar]':'qgini'}),
        ("ElmGenstat", "*.ElmGenstat", {'P [MW]':'pgini', 'Q [Mvar]':'qgini'}),
        ("ElmPvsys", "*.ElmPvsys", {'P [MW]':'pgini', 'Q [Mvar]':'qgini'}),
        ("ElmLod", "*.ElmLod", {'P [MW]':'plini', 'Q [Mvar]':'qlini'}),
        ("ElmLne", "*.ElmLne", {'Loading [%]':'c:loading', 'Current [kA]':'m:I:bus1'}),
        ("ElmTr2", "*.ElmTr2", {'Loading [%]':'c:loading', 'Tap':'nntap'}),
    ]
    
    for elem_type, pattern, attrs in elem_configs:
        try:
            elements = app.GetCalcRelevantObjects(pattern)
            data = _build_element_data(elements, elem_type, attrs)
            all_settings[elem_type] = data
            
            # Raportowanie None dla linii
            if elem_type == "ElmLne":
                none_count = sum(1 for row in data if row.get('Loading [%]') == 'N/A')
                if none_count > 0:
                    print(f"  ⚠️ ElmLne:{none_count}/{len(elements)} linii bez obciążenia")
            
            print(f"  ✓ {elem_type}:{len(data)} elementów")
        except Exception as e:
            print(f"  ⚠️ {elem_type} błąd:{e}")
            all_settings[elem_type] = []
    
    return all_settings

# === FUNKCJA CELU ===
class HybridObjective:
    """Hybrydowa funkcja celu - RANKING + ZŁAGODZONA WERYFIKACJA"""
    
    def __init__(self, app, ldf, opt_variables:List[Dict], candidate_lines:List[str],
                 min_lines_out:int, max_lines_out:int, config:HybridConfig, overload_base:float):
        
        print("\n" + "="*80)
        print("🔧 FUNKCJA CELU (RANKING + ZŁAGODZONA WERYFIKACJA)")
        print("="*80)
        
        self.app = app
        self.ldf = ldf
        self.opt_variables = opt_variables
        self.candidate_lines = candidate_lines
        self.min_lines_out = min_lines_out
        self.max_lines_out = max_lines_out
        self.config = config
        self.overload_base = max(overload_base, 0.1)
        
        self.eval_count = 0
        self.island_count = 0
        self.lf_fail_count = 0
        self.best_value = np.inf
        self.best_x = None
        
        # Wymiary
        self.n_gen_vars = sum(
            1 for v in opt_variables for attr in ['Attribute_P', 'Attribute_Q'] if v.get(attr)
        )
        self.n_line_vars = 1 + len(candidate_lines)
        self.total_dim = self.n_gen_vars + self.n_line_vars
        
        print(f"  Gen vars:{self.n_gen_vars}")
        print(f"  Line vars:{self.n_line_vars} (1 liczba + {len(candidate_lines)} ranking)")
        print(f"  Total:{self.total_dim}")
        print(f"  Zakres wyłączeń:[{min_lines_out}, {max_lines_out}]")
        print(f"  Overload_base:{self.overload_base:.3f}")
        print(f"  Wagi:w1={config.weight_topology}, w2={config.weight_overload}")
        print(f"  ✅ Złagodzona weryfikacja:akceptuje ≥{config.min_valid_lines_pct*100:.0f}% linii z danymi")
        
        # Cache
        self._gen_cache = {
            f"{v['Element_Name']}_{v['Element_Type']}":find_element_multi_method(app, v['Element_Name'], v['Element_Type'])
            for v in opt_variables
        }
        self._line_cache = {
            name:find_element_multi_method(app, name, "ElmLne") for name in candidate_lines
        }
        
        gen_count = sum(1 for v in self._gen_cache.values() if v)
        line_count = sum(1 for v in self._line_cache.values() if v)
        print(f"  ✓ Gen:{gen_count}/{len(opt_variables)}")
        print(f"  ✓ Lines:{line_count}/{len(candidate_lines)}")
        print("="*80)
    
    def _decode_vector(self, x:np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Dekoduj wektor - WYMUSZENIE [Min, Max]"""
        gen_values = x[:self.n_gen_vars]
        n_to_disable_raw = x[self.n_gen_vars]
        line_scores = x[self.n_gen_vars + 1:]
        
        n_range = self.max_lines_out - self.min_lines_out
        n_to_disable = (self.min_lines_out + int(round(n_to_disable_raw * n_range))) if n_range > 0 else self.min_lines_out
        n_to_disable = max(self.min_lines_out, min(n_to_disable, self.max_lines_out))
        
        lines_to_disable = []
        if n_to_disable > 0:
            line_ranking = sorted(enumerate(line_scores[:len(self.candidate_lines)]), key=lambda x:x[1], reverse=True)
            lines_to_disable = [self.candidate_lines[idx] for idx, _ in line_ranking[:n_to_disable]]
        
        return gen_values, lines_to_disable
    
    def _apply_generator_settings(self, gen_values:np.ndarray):
        """Ustaw generatory"""
        idx = 0
        for var in self.opt_variables:
            key = f"{var['Element_Name']}_{var['Element_Type']}"
            elem = self._gen_cache.get(key)
            if elem:
                for attr in ['Attribute_P', 'Attribute_Q']:
                    if var.get(attr):
                        try:
                            elem.SetAttribute(var[attr], float(gen_values[idx]))
                        except:
                            pass
                        idx += 1
    
    def _apply_line_settings(self, lines_to_disable:List[str]):
        """Ustaw linie - WŁĄCZ WSZYSTKIE, potem wyłącz wybrane"""
        # Reset wszystkich linii
        try:
            all_lines = self.app.GetCalcRelevantObjects("*.ElmLne")
            for line in all_lines:
                try:
                    line.outserv = 0
                except:
                    pass
        except:
            pass
        
        # Wyłącz wybrane
        for line_name in lines_to_disable:
            line = self._line_cache.get(line_name)
            if line:
                try:
                    line.outserv = 1
                except:
                    pass
    
    def _check_island(self) -> bool:
        """Sprawdź wyspy - ZŁAGODZONA WERSJA"""
        try:
            code = self.ldf.Execute()
            
            if code >= 2:# Nie zbiegł
                return True
            if code == 0:# OK
                return False
            
            # code == 1:sprawdź napięcia
            buses = self.app.GetCalcRelevantObjects("*.ElmTerm")
            total_buses = len(buses)
            if total_buses == 0:
                return True
            
            low_voltage_count = sum(
                1 for bus in buses
                for u_pu in [bus.GetAttribute("m:u")] if u_pu is not None and u_pu < 0.01
            )
            none_count = sum(
                1 for bus in buses
                for u_pu in [bus.GetAttribute("m:u")] if u_pu is None
            )
            
            # ZŁAGODZONE WARUNKI
            if none_count > total_buses * 0.8:# >80%
                return True
            if low_voltage_count > total_buses * 0.1:# >10%
                return True
            
            return False
        except:
            return True
    
    def _calculate_current_overload(self) -> float:
        """Oblicz przeciążenia - ZŁAGODZONA WERSJA (≥70% linii z danymi)"""
        try:
            lines = self.app.GetCalcRelevantObjects("*.ElmLne")
            if not lines:
                return np.inf
            
            overload = 0.0
            valid_readings = 0
            
            for line in lines:
                if line.loc_name in self.config.observed_lines:
                    try:
                        loading = line.GetAttribute("c:loading")
                        if loading is not None:
                            valid_readings += 1
                            if loading > 100:
                                overload += loading - 100
                    except:
                        pass
            
            # ZŁAGODZONY WARUNEK
            required_valid = int(len(self.config.observed_lines) * self.config.min_valid_lines_pct)
            if valid_readings < required_valid:
                return np.inf
            
            return overload
        except:
            return np.inf
    
    def __call__(self, x:np.ndarray) -> float:
        """Funkcja celu"""
        self.eval_count += 1
        
        try:
            # Dekoduj + zastosuj
            gen_values, lines_to_disable = self._decode_vector(x)
            self._apply_generator_settings(gen_values)
            self._apply_line_settings(lines_to_disable)
            
            # Wyspa
            if self._check_island():
                self.island_count += 1
                return np.inf
            
            # Przeciążenia
            overload_current = self._calculate_current_overload()
            if np.isinf(overload_current):
                self.lf_fail_count += 1
                return np.inf
            
            # Funkcja celu
            n_disabled = len(lines_to_disable)
            term1 = n_disabled / self.max_lines_out if self.max_lines_out > 0 else float(n_disabled)
            term2 = overload_current / self.overload_base
            f_total = self.config.weight_topology * term1 + self.config.weight_overload * term2
            
            if np.isnan(f_total) or np.isinf(f_total):
                return np.inf
            
            # Best tracking
            if f_total < self.best_value:
                self.best_value = f_total
                self.best_x = x.copy()
                msg = f"\n✅ BEST #{self.eval_count}:f={f_total:.6f} (n={n_disabled}, topo={term1:.4f}, over={term2:.4f})"
                try:
                    self.app.PrintPlain(msg)
                except:
                    pass
            
            return f_total
        except:
            return np.inf

# === GŁÓWNA FUNKCJA ===
def run_hybrid_optimization(app, ldf, excel_file:str, scenario_name="HYBRID"):
    """Uruchom hybrydową optymalizację"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = HYBRID_OUT_DIR / f"hybrid_log_{timestamp}.txt"
    error_file = HYBRID_OUT_DIR / f"hybrid_ERROR_{timestamp}.txt"
    
    try:
        logger = Logger(str(log_file), app)
    except Exception as e:
        error_file.write_text(f"LOGGER ERROR:{e}\n")
        return None
    
    def log_msg(msg):
        safe_log(msg, logger, app)
    
    try:
        log_msg("="*80)
        log_msg("HYBRYDOWA OPTYMALIZACJA - ZŁAGODZONA WERYFIKACJA")
        log_msg("="*80)
        log_msg(f"Scenariusz:{scenario_name}")
        log_msg(f"Timestamp:{timestamp}")
        log_msg(f"✅ Weryfikacja:akceptuje ≥{CONFIG.min_valid_lines_pct*100:.0f}% linii z danymi")
        log_msg("="*80)
        
        # [0/7] Dane bazowe
        log_msg("\n[0/7] Wczytywanie danych bazowych z Excela...")
        if not load_base_data_from_excel(app, excel_file):
            log_msg("❌ Błąd wczytywania danych bazowych - przerywam")
            return None
        
        # [1/7] Config
        log_msg("\n[1/7] Konfiguracja optymalizacji...")
        opt_variables, candidate_lines, min_lines_out, max_lines_out = load_hybrid_config(excel_file)
        if not opt_variables and not candidate_lines:
            log_msg("❌ Brak konfiguracji optymalizacji")
            return None
        
        log_msg(f"  ✓ Zmiennych optymalizacji:{len(opt_variables)}")
        log_msg(f"  ✓ Linii kandydujących:{len(candidate_lines)}")
        log_msg(f"  ✓ Zakres wyłączeń:[{min_lines_out}, {max_lines_out}]")
        
        # [2/7] Stan bazowy
        log_msg("\n[2/7] Stan bazowy (po wczytaniu danych)...")
        all_lines = app.GetCalcRelevantObjects("*.ElmLne")
        for line in all_lines:
            try:
                line.outserv = 0
            except:
                pass
        
        stats_before = get_observed_lines_stats(app, ldf, CONFIG.observed_lines)
        log_msg(f"\n  📊 Stan bazowy:")
        log_msg(f"     Suma przeciążeń:{stats_before['total_overload']:.3f}")
        log_msg(f"     Linii przeciążonych:{stats_before['overloaded_count']}")
        if stats_before['overloaded_count'] > 0:
            log_msg(f"     Max przeciążenie:{stats_before['max_overload']:.2f}% ({stats_before['max_line_name']})")
        
        settings_before = export_all_settings(app, ldf, "BEFORE")
        overload_base = max(stats_before['total_overload'], 0.1)
        
        # [3/7] Funkcja celu
        log_msg("\n[3/7] Inicjalizacja funkcji celu...")
        objective = HybridObjective(
            app, ldf, opt_variables, candidate_lines, min_lines_out, max_lines_out, CONFIG, overload_base
        )
        
        # [4/7] Granice
        log_msg("\n[4/7] Określanie granic zmiennych...")
        lb, ub, var_names = [], [], []
        
        for var in opt_variables:
            for attr_min, attr_max, suffix in [('Pmin', 'Pmax', '_P'), ('Qmin', 'Qmax', '_Q')]:
                if attr_min in var and attr_max in var:
                    lb.append(float(var[attr_min]))
                    ub.append(float(var[attr_max]))
                    var_names.append(f"{var['Element_Name']}{suffix}")
        
        lb.append(0.0)
        ub.append(1.0)
        var_names.append("N_lines_to_disable")
        
        for line_name in candidate_lines:
            lb.extend([0.0])
            ub.extend([1.0])
            var_names.append(f"Score_{line_name}")
        
        lb, ub = np.array(lb), np.array(ub)
        
        log_msg(f"  ✓ Wymiar przestrzeni:{len(lb)}")
        
        # [5/7] PSO
        log_msg("\n[5/7] Uruchamianie PSO...")
        log_msg(f"  Parametry:")
        log_msg(f"    Cząstek:{CONFIG.n_particles}")
        log_msg(f"    Max iteracji:{CONFIG.max_iter}")
        log_msg(f"    🎯 Early stop threshold:f <= {CONFIG.early_stop_threshold}")
        
        pso = PSO(
            func=objective,
            n_particles=CONFIG.n_particles,
            dim=len(lb),
            lb=lb,
            ub=ub,
            max_iter=CONFIG.max_iter,
            w=CONFIG.w,
            c1=CONFIG.c1,
            c2=CONFIG.c2,
            autosave_every_iters=CONFIG.autosave_every,
            autosave_path=str(HYBRID_OUT_DIR / f"hybrid_checkpoint_{timestamp}.npz"),
            early_stop_threshold=CONFIG.early_stop_threshold,
            early_stop_patience=CONFIG.early_stop_patience
        )
        
        time_start = time.time()
        result = pso.optimize()
        time_end = time.time()
        
        log_msg(f"\n✅ Optymalizacja zakończona:{time_end - time_start:.2f}s")
        
        # Statystyki odrzuceń
        log_msg(f"\n📊 STATYSTYKI ODRZUCEŃ:")
        log_msg(f"  Ewaluacji total:{objective.eval_count}")
        log_msg(f"  Wyspy:{objective.island_count} ({100*objective.island_count/objective.eval_count:.1f}%)")
        log_msg(f"  LF fails:{objective.lf_fail_count} ({100*objective.lf_fail_count/objective.eval_count:.1f}%)")
        accepted = objective.eval_count - objective.island_count - objective.lf_fail_count
        log_msg(f"  Akceptowanych:{accepted} ({100*accepted/objective.eval_count:.1f}%)")
        
        if result.get('early_stopped'):
            reason = result.get('reason')
            stopped_at = result.get('stopped_at_iter')
            log_msg(f"\n⏸️ EARLY STOPPING:")
            log_msg(f"   Powód:{reason}")
            log_msg(f"   Zatrzymano na iteracji:{stopped_at}/{CONFIG.max_iter}")
        
        log_msg(f"\nF_best:{result['gbest_val']:.6f}")
        
        # [6/7] Zastosowanie najlepszego rozwiązania
        log_msg("\n[6/7] Zastosowanie najlepszego rozwiązania...")
        gen_best, lines_best = objective._decode_vector(result['gbest'])
        objective._apply_generator_settings(gen_best)
        objective._apply_line_settings(lines_best)
        
        stats_after = get_observed_lines_stats(app, ldf, CONFIG.observed_lines)
        settings_after = export_all_settings(app, ldf, "AFTER")
        
        log_msg(f"\n📊 PORÓWNANIE:")
        log_msg(f"  Przed:{stats_before['total_overload']:.3f} ({stats_before['overloaded_count']} linii)")
        log_msg(f"  Po:{stats_after['total_overload']:.3f} ({stats_after['overloaded_count']} linii)")
        log_msg(f"  Wyłączonych linii:{len(lines_best)}")
        
        if stats_before['total_overload'] > 0:
            change = stats_after['total_overload'] - stats_before['total_overload']
            change_pct = (change / stats_before['total_overload']) * 100
            if change < 0:
                log_msg(f"  ✅ REDUKCJA:{abs(change):.3f} ({abs(change_pct):.2f}%)")
            elif change > 0:
                log_msg(f"  ⚠️ WZROST:{change:.3f} (+{change_pct:.2f}%)")
            else:
                log_msg(f"  ➡️ BEZ ZMIANY")
        
        # [7/7] Excel
        log_msg("\n[7/7] Zapisywanie wyników do Excel...")
        output_file = HYBRID_OUT_DIR / f"HYBRID_{scenario_name}_{timestamp}.xlsx"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Podsumowanie
            early_stop_info = "Wszystkie iteracje"
            if result.get('early_stopped'):
                reason = result.get('reason')
                early_stop_info = "Cel osiągnięty" if reason == 'threshold_reached' else "Brak poprawy"
            
            summary = {
                'Parametr':[
                    'Scenariusz', 'Timestamp', 'Czas [s]', 'Status', 'Iteracji wykonanych', '',
                    'Ewaluacji total', 'Wyspy', 'LF fails', 'Akceptowanych', 'Acceptance rate', '',
                    'Przeciążenia przed', 'Przeciążenia po', 'Redukcja', '',
                    'Linii wyłączonych', 'Zakres dozwolony', '', 'f_celu',
                ],
                'Wartość':[
                    scenario_name, timestamp, f"{time_end - time_start:.2f}", early_stop_info,
                    f"{result.get('stopped_at_iter', CONFIG.max_iter)}/{CONFIG.max_iter}", '',
                    objective.eval_count,
                    f"{objective.island_count} ({100*objective.island_count/objective.eval_count:.1f}%)",
                    f"{objective.lf_fail_count} ({100*objective.lf_fail_count/objective.eval_count:.1f}%)",
                    f"{accepted} ({100*accepted/objective.eval_count:.1f}%)",
                    f"{100*accepted/objective.eval_count:.1f}%", '',
                    f"{stats_before['total_overload']:.3f}",
                    f"{stats_after['total_overload']:.3f}",
                    f"{(stats_before['total_overload'] - stats_after['total_overload']) / max(stats_before['total_overload'], 0.1) * 100:.2f}%",
                    '', len(lines_best), f"[{min_lines_out}, {max_lines_out}]", '',
                    f"{result['gbest_val']:.6f}",
                ]
            }
            pd.DataFrame(summary).to_excel(writer, sheet_name='Podsumowanie', index=False)
            
            # Historia
            history = pd.DataFrame({
                'Iteration':range(len(result['best_per_iter'])),
                'F_best':result['best_per_iter'],
            })
            history.to_excel(writer, sheet_name='History', index=False)
            
            # Ustawienia PRZED/PO
            for phase, settings in [('BEFORE', settings_before), ('AFTER', settings_after)]:
                all_gen = []
                for key in ['ElmSym', 'ElmGenstat', 'ElmPvsys']:
                    all_gen.extend(settings.get(key, []))
                if all_gen:
                    pd.DataFrame(all_gen).to_excel(writer, sheet_name=f'{phase}_Generators', index=False)
                
                for key, sheet_suffix in [('ElmLod', 'Loads'), ('ElmLne', 'Lines'), ('ElmTr2', 'Transformers')]:
                    if settings.get(key):
                        pd.DataFrame(settings[key]).to_excel(writer, sheet_name=f'{phase}_{sheet_suffix}', index=False)
            
            # Zmiany optymalizowanych elementów
            gen_changes = []
            idx = 0
            for var in opt_variables:
                row = {'Element':var['Element_Name'], 'Type':var['Element_Type']}
                for attr, suffix in [('Attribute_P', 'P_AFTER [MW]'), ('Attribute_Q', 'Q_AFTER [Mvar]')]:
                    if var.get(attr):
                        row[suffix] = f"{gen_best[idx]:.6f}"
                        idx += 1
                gen_changes.append(row)
            pd.DataFrame(gen_changes).to_excel(writer, sheet_name='Optimized_Generators', index=False)
            
            # Linie - status
            lines_status = [
                {'Line':line_name, 'Status_AFTER':'DISABLED' if line_name in lines_best else 'ACTIVE'}
                for line_name in candidate_lines
            ]
            pd.DataFrame(lines_status).to_excel(writer, sheet_name='Optimized_Lines', index=False)
            
            # Porównanie
            comparison = {
                'Metryka':['Suma przeciążeń', 'Linii przeciążonych', 'Max przeciążenie', 'Linii wyłączonych'],
                'Przed':[
                    f"{stats_before['total_overload']:.3f}",
                    stats_before['overloaded_count'],
                    f"{stats_before['max_overload']:.2f}%",
                    0,
                ],
                'Po':[
                    f"{stats_after['total_overload']:.3f}",
                    stats_after['overloaded_count'],
                    f"{stats_after['max_overload']:.2f}%",
                    len(lines_best),
                ],
                'Zmiana':[
                    f"{(stats_after['total_overload'] - stats_before['total_overload']) / max(stats_before['total_overload'], 0.1) * 100:.2f}%",
                    f"{stats_after['overloaded_count'] - stats_before['overloaded_count']:+d}",
                    'N/A',
                    f"+{len(lines_best)}",
                ]
            }
            pd.DataFrame(comparison).to_excel(writer, sheet_name='Comparison', index=False)
        
        log_msg(f"\n✅ Wyniki zapisane:{output_file}")
        log_msg("\n" + "="*80)
        log_msg("✅ OPTYMALIZACJA ZAKOŃCZONA POMYŚLNIE")
        log_msg("="*80)
        
        return result
    except Exception as e:
        error_msg = f"\n❌ BŁĄD:{e}\n"
        import traceback
        error_msg += traceback.format_exc()
        
        try:
            log_msg(error_msg)
        except:
            pass
        
        error_file.write_text(error_msg)
        return None
    finally:
        if logger:
            logger.close()

# === MAIN ===
def main():
    print("\n" + "="*80)
    print("🚀 HYBRID OPTIMIZER - ZŁAGODZONA WERYFIKACJA + DIAGNOSTYKA")
    print("="*80)
    
    try:
        import powerfactory
        app = powerfactory.GetApplicationExt()
        
        if app is None:
            print("❌ Nie można połączyć z PowerFactory")
            return
        
        prj = app.GetActiveProject()
        if not prj:
            print("⚠️ Brak aktywnego projektu")
            return
        print(f"✓ Projekt:{prj.loc_name}")
        
        ldf = app.GetFromStudyCase("ComLdf")
        if not ldf:
            print("❌ Brak Load Flow Calculation w Study Case")
            return
        print(f"✓ Load Flow znaleziony")
        
        print(f"\n📂 Excel:{EXCEL_FILE}")
        print(f"📂 Wyniki:{HYBRID_OUT_DIR}")
        print("="*80)
        
        result = run_hybrid_optimization(app, ldf, EXCEL_FILE, "N1_Hybrid")
        
        if result:
            print("\n" + "="*80)
            print("✅ SUKCES!")
            print("="*80)
            print(f"F_best:{result['gbest_val']:.6f}")
            if result.get('early_stopped'):
                print(f"⏸️ Early stop:{result.get('reason')}")
                print(f"   Iteracja:{result.get('stopped_at_iter')}")
            print(f"📁 Wyniki:{HYBRID_OUT_DIR}")
            print("\n📊 SPRAWDŹ:")
            print("  1.Arkusz 'Podsumowanie' - Acceptance rate (powinno być >5%)")
            print("  2.Arkusz 'AFTER_Lines' - czy obciążenia są liczbami (nie N/A)")
            print("="*80)
        else:
            print("\n❌ Optymalizacja nie powiodła się")
            print("Sprawdź logi w folderze Wyniki/Hybrid/")
    except Exception as e:
        print(f"\n❌ BŁĄD:{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()