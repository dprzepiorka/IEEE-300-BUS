"""
HYBRYDOWA OPTYMALIZACJA:Generatory + Rekonfiguracja topologii jednocześnie
Funkcja celu:f = w1*(N_disabled/N_max) + w2*(Overload_current/Overload_base)
WERSJA Z RANKINGIEM LINII + Min/Max z Excela + EARLY STOPPING + PEŁNY EKSPORT USTAWIEŃ
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

# Usuń cache modułów
if 'run_optimization' in sys.modules:
    del sys.modules['run_optimization']
if 'PSO' in sys.modules:
    del sys.modules['PSO']

try:
    from run_optimization import (
        Logger, find_element_multi_method,
        load_export_config, collect_results_parametrized, save_results_to_excel,
        OUT_DIR, EXCEL_FILE, PROJECT_NAME, USER
    )
    from PSO import PSO
    print("✅ Moduły zaimportowane")
except ImportError as e:
    print(f"❌ Błąd importu:{e}")
    raise

# ==========================================
# KONFIGURACJA
# ==========================================

# Folder wyników
HYBRID_OUT_DIR = os.path.join(OUT_DIR, "Hybrid")
if not os.path.exists(HYBRID_OUT_DIR):
    try:
        os.makedirs(HYBRID_OUT_DIR)
        print(f"✅ Utworzono folder:{HYBRID_OUT_DIR}")
    except:
        HYBRID_OUT_DIR = OUT_DIR

# Parametry PSO
HYBRID_PSO_PARAMS = {
    'n_particles':100,
    'max_iter':500,
    'w':0.7,
    'c1':1.5,
    'c2':1.5,
    'autosave_every':10,
    'early_stop_threshold':0.0,    # ✅ Zatrzymaj gdy f <= 0
    'early_stop_patience':100,      # ✅ Brak poprawy przez 100 iteracji
}

# Wagi funkcji celu
WEIGHT_TOPOLOGY = 0  # Kara za wyłączenia (znormalizowana)
WEIGHT_OVERLOAD = 1  # Waga na przeciążenia

# Linie obserwowane
OBSERVED_LINES = ['5.73', '8.14', '79.73', '35.77', '6.2', '73.71', '71.72']

# ==========================================
# LOGGER GLOBALNY
# ==========================================

_logger = None

def log(message):
    """Funkcja logowania"""
    global _logger
    if _logger:
        _logger.write(str(message) + "\n")
    else:
        print(message)

# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================

def load_hybrid_config(excel_file):
    """Wczytaj konfigurację"""
    
    print("\n" + "="*80)
    print("🔍 WCZYTYWANIE KONFIGURACJI")
    print("="*80)
    
    try:
        # === 1.GENERATORY ===
        print("\n[1/2] Generatory...")
        try:
            df_opt = pd.read_excel(excel_file, sheet_name="Optymalizacja")
            opt_variables = df_opt.to_dict('records')
            print(f"  ✓ {len(opt_variables)} elementów")
        except Exception as e:
            print(f"  ⚠️ Błąd:{e}")
            opt_variables = []
        
        # === 2.TOPOLOGIA ===
        print("\n[2/2] Topologia...")
        try:
            df_reconfig = pd.read_excel(excel_file, sheet_name="Rekonfiguracja")
            
            if 'Line_Name' in df_reconfig.columns and 'Can_Disable' in df_reconfig.columns:
                candidate_df = df_reconfig[df_reconfig['Can_Disable'] == 1]
                candidate_lines = candidate_df['Line_Name'].astype(str).tolist()
                candidate_lines = [x.strip() for x in candidate_lines if x and x.lower() != 'nan']
                
                if 'Priority' in df_reconfig.columns:
                    priorities = candidate_df['Priority'].tolist()
                    sorted_pairs = sorted(zip(candidate_lines, priorities),
                                        key=lambda x:x[1] if pd.notna(x[1]) else 999)
                    candidate_lines = [line for line, _ in sorted_pairs]
                
                print(f"  ✓ {len(candidate_lines)} linii")
            else:
                print(f"  ⚠️ Brak kolumn")
                candidate_lines = []
            
            # WCZYTAJ Min i Max
            max_lines_out = 3
            min_lines_out = 0
            
            if 'Parameter' in df_reconfig.columns and 'Value' in df_reconfig.columns:
                params = df_reconfig[['Parameter', 'Value']].dropna()
                for _, row in params.iterrows():
                    param_name = str(row['Parameter']).strip()
                    if param_name == 'Max_Lines_Out':
                        max_lines_out = int(row['Value'])
                    elif param_name == 'Min_Lines_Out':
                        min_lines_out = int(row['Value'])
            
            print(f"  ✓ Min:{min_lines_out}, Max:{max_lines_out}")
            
        except Exception as e:
            print(f"  ⚠️ Błąd:{e}")
            candidate_lines = []
            max_lines_out = 3
            min_lines_out = 0
        
        print("="*80)
        
        return opt_variables, candidate_lines, min_lines_out, max_lines_out
    
    except Exception as e:
        print(f"❌ BŁĄD:{e}")
        import traceback
        traceback.print_exc()
        return [], [], 0, 3

def get_observed_lines_stats(app, ldf, observed_lines):
    """Pobierz statystyki obserwowanych linii"""
    
    try:
        ldf.Execute()
        
        lines = app.GetCalcRelevantObjects("*.ElmLne")
        
        overloaded_count = 0
        overloads = []
        max_overload = 0.0
        max_line_name = ""
        total_overload = 0.0
        
        for line in lines:
            if line.loc_name in observed_lines:
                try:
                    loading = line.GetAttribute("c:loading")
                    
                    if loading and loading > 100:
                        overloaded_count += 1
                        excess = loading - 100
                        overloads.append(excess)
                        total_overload += excess
                        
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
            'total_overload':0.0,
            'overloaded_count':0,
            'max_overload':0.0,
            'max_line_name':'N/A',
            'avg_overload':0.0,
        }

# ==========================================
# ✅ NOWA FUNKCJA - EKSPORT WSZYSTKICH USTAWIEŃ
# ==========================================

def export_all_settings(app, phase="BEFORE"):
    """
    Eksportuj ustawienia WSZYSTKICH elementów generujących/pobierających moc
    
    Returns:
        dict z kluczami:'ElmSym', 'ElmGenstat', 'ElmPvsys', 'ElmLod', 'ElmLne'
    """
    
    print(f"\n📋 Eksport ustawień - {phase}...")
    
    all_settings = {}
    
    # === 1.GENERATORY SYNCHRONICZNE (ElmSym) ===
    try:
        gens = app.GetCalcRelevantObjects("*.ElmSym")
        gen_data = []
        
        for gen in gens:
            try:
                row = {
                    'Element':gen.loc_name,
                    'Type':'ElmSym',
                    'P [MW]':gen.GetAttribute('pgini') if hasattr(gen, 'pgini') else None,
                    'Q [Mvar]':gen.GetAttribute('qgini') if hasattr(gen, 'qgini') else None,
                    'Status':'OUT' if getattr(gen, 'outserv', 0) == 1 else 'IN',
                }
                gen_data.append(row)
            except:
                pass
        
        all_settings['ElmSym'] = gen_data
        print(f"  ✓ ElmSym:{len(gen_data)} elementów")
    except Exception as e:
        print(f"  ⚠️ ElmSym błąd:{e}")
        all_settings['ElmSym'] = []
    
    # === 2.GENERATORY STATYCZNE (ElmGenstat) ===
    try:
        genstat = app.GetCalcRelevantObjects("*.ElmGenstat")
        genstat_data = []
        
        for gs in genstat:
            try:
                row = {
                    'Element':gs.loc_name,
                    'Type':'ElmGenstat',
                    'P [MW]':gs.GetAttribute('pgini') if hasattr(gs, 'pgini') else None,
                    'Q [Mvar]':gs.GetAttribute('qgini') if hasattr(gs, 'qgini') else None,
                    'Status':'OUT' if getattr(gs, 'outserv', 0) == 1 else 'IN',
                }
                genstat_data.append(row)
            except:
                pass
        
        all_settings['ElmGenstat'] = genstat_data
        print(f"  ✓ ElmGenstat:{len(genstat_data)} elementów")
    except Exception as e:
        print(f"  ⚠️ ElmGenstat błąd:{e}")
        all_settings['ElmGenstat'] = []
    
    # === 3.SYSTEMY PV (ElmPvsys) ===
    try:
        pv = app.GetCalcRelevantObjects("*.ElmPvsys")
        pv_data = []
        
        for p in pv:
            try:
                row = {
                    'Element':p.loc_name,
                    'Type':'ElmPvsys',
                    'P [MW]':p.GetAttribute('pgini') if hasattr(p, 'pgini') else None,
                    'Q [Mvar]':p.GetAttribute('qgini') if hasattr(p, 'qgini') else None,
                    'Status':'OUT' if getattr(p, 'outserv', 0) == 1 else 'IN',
                }
                pv_data.append(row)
            except:
                pass
        
        all_settings['ElmPvsys'] = pv_data
        print(f"  ✓ ElmPvsys:{len(pv_data)} elementów")
    except Exception as e:
        print(f"  ⚠️ ElmPvsys błąd:{e}")
        all_settings['ElmPvsys'] = []
    
    # === 4.OBCIĄŻENIA (ElmLod) ===
    try:
        loads = app.GetCalcRelevantObjects("*.ElmLod")
        load_data = []
        
        for load in loads:
            try:
                row = {
                    'Element':load.loc_name,
                    'Type':'ElmLod',
                    'P [MW]':load.GetAttribute('plini') if hasattr(load, 'plini') else None,
                    'Q [Mvar]':load.GetAttribute('qlini') if hasattr(load, 'qlini') else None,
                    'Status':'OUT' if getattr(load, 'outserv', 0) == 1 else 'IN',
                }
                load_data.append(row)
            except:
                pass
        
        all_settings['ElmLod'] = load_data
        print(f"  ✓ ElmLod:{len(load_data)} elementów")
    except Exception as e:
        print(f"  ⚠️ ElmLod błąd:{e}")
        all_settings['ElmLod'] = []
    
    # === 5.LINIE (ElmLne) - STATUS ===
    try:
        lines = app.GetCalcRelevantObjects("*.ElmLne")
        line_data = []
        
        for line in lines:
            try:
                row = {
                    'Element':line.loc_name,
                    'Type':'ElmLne',
                    'Status':'OUT' if getattr(line, 'outserv', 0) == 1 else 'IN',
                }
                line_data.append(row)
            except:
                pass
        
        all_settings['ElmLne'] = line_data
        print(f"  ✓ ElmLne:{len(line_data)} linii")
    except Exception as e:
        print(f"  ⚠️ ElmLne błąd:{e}")
        all_settings['ElmLne'] = []
    
    # === 6.TRANSFORMATORY (ElmTr2) - STATUS ===
    try:
        trafos = app.GetCalcRelevantObjects("*.ElmTr2")
        trafo_data = []
        
        for trafo in trafos:
            try:
                row = {
                    'Element':trafo.loc_name,
                    'Type':'ElmTr2',
                    'Status':'OUT' if getattr(trafo, 'outserv', 0) == 1 else 'IN',
                    'Tap':trafo.GetAttribute('nntap') if hasattr(trafo, 'nntap') else None,
                }
                trafo_data.append(row)
            except:
                pass
        
        all_settings['ElmTr2'] = trafo_data
        print(f"  ✓ ElmTr2:{len(trafo_data)} transformatorów")
    except Exception as e:
        print(f"  ⚠️ ElmTr2 błąd:{e}")
        all_settings['ElmTr2'] = []
    
    return all_settings

# ==========================================
# KLASA FUNKCJI CELU
# ==========================================

class HybridObjective:
    """Hybrydowa funkcja celu - RANKING + Min/Max"""
    
    def __init__(self, app, ldf, opt_variables, candidate_lines, min_lines_out, max_lines_out,
                 observed_lines, overload_base, w1=0.3, w2=0.7, debug_file=None):
        
        print("\n" + "="*80)
        print("🔧 FUNKCJA CELU (RANKING)")
        print("="*80)
        
        self.app = app
        self.ldf = ldf
        self.opt_variables = opt_variables
        self.candidate_lines = candidate_lines
        self.min_lines_out = min_lines_out
        self.max_lines_out = max_lines_out
        self.observed_lines = observed_lines
        self.overload_base = overload_base
        self.w1 = w1
        self.w2 = w2
        self.debug_file = debug_file
        
        self.eval_count = 0
        self.best_value = np.inf
        self.best_x = None
        
        # Wymiary
        self.n_gen_vars = 0
        for v in opt_variables:
            if 'Attribute_P' in v and v.get('Attribute_P'):
                self.n_gen_vars += 1
            if 'Attribute_Q' in v and v.get('Attribute_Q'):
                self.n_gen_vars += 1
        
        self.n_line_vars = 1 + len(candidate_lines)
        self.total_dim = self.n_gen_vars + self.n_line_vars
        
        print(f"  Gen vars:{self.n_gen_vars}")
        print(f"  Line vars:{self.n_line_vars} (1 liczba + {len(candidate_lines)} ranking)")
        print(f"  Total:{self.total_dim}")
        print(f"  Zakres wyłączeń:[{min_lines_out}, {max_lines_out}]")
        print(f"  Overload_base:{overload_base:.3f}")
        print(f"  Wagi:w1={w1}, w2={w2}")
        
        # Cache
        self._gen_cache = {}
        self._line_cache = {}
        self._cache_elements()
        
        print("="*80)
    
    def _cache_elements(self):
        """Cache elementów"""
        
        print("\n🔍 Cache...")
        
        gen_count = 0
        for var in self.opt_variables:
            key = f"{var['Element_Name']}_{var['Element_Type']}"
            if key not in self._gen_cache:
                elem = find_element_multi_method(self.app, var['Element_Name'], var['Element_Type'])
                if elem:
                    self._gen_cache[key] = elem
                    gen_count += 1
        
        line_count = 0
        for line_name in self.candidate_lines:
            line = find_element_multi_method(self.app, line_name, "ElmLne")
            if line:
                self._line_cache[line_name] = line
                line_count += 1
        
        print(f"  ✓ Gen:{gen_count}/{len(self.opt_variables)}")
        print(f"  ✓ Lines:{line_count}/{len(self.candidate_lines)}")
    
    def _decode_vector(self, x):
        """Dekoduj wektor - WYMUSZENIE [Min, Max]"""
        gen_values = x[:self.n_gen_vars]
        
        # Pierwsza zmienna topologii = liczba [0, 1]
        n_to_disable_raw = x[self.n_gen_vars]
        
        # Pozostałe = ranking
        line_scores = x[self.n_gen_vars + 1:]
        
        # SKALUJ do zakresu [Min, Max]
        n_range = self.max_lines_out - self.min_lines_out
        
        if n_range > 0:
            n_to_disable = self.min_lines_out + int(round(n_to_disable_raw * n_range))
        else:
            n_to_disable = self.min_lines_out
        
        # Ogranicz do [Min, Max]
        n_to_disable = max(self.min_lines_out, min(n_to_disable, self.max_lines_out))
        
        # Wybierz TOP N linii według rankingu
        lines_to_disable = []
        
        if n_to_disable > 0 and len(line_scores) >= len(self.candidate_lines):
            line_ranking = []
            for i in range(len(self.candidate_lines)):
                if i < len(line_scores):
                    line_ranking.append((i, line_scores[i]))
            
            line_ranking.sort(key=lambda x:x[1], reverse=True)
            
            for i in range(min(n_to_disable, len(line_ranking))):
                idx = line_ranking[i][0]
                if idx < len(self.candidate_lines):
                    lines_to_disable.append(self.candidate_lines[idx])
        
        return gen_values, lines_to_disable
    
    def _apply_generator_settings(self, gen_values):
        """Ustaw generatory"""
        idx = 0
        for var in self.opt_variables:
            key = f"{var['Element_Name']}_{var['Element_Type']}"
            elem = self._gen_cache.get(key)
            
            if elem:
                if 'Attribute_P' in var and var.get('Attribute_P'):
                    try:
                        elem.SetAttribute(var['Attribute_P'], float(gen_values[idx]))
                    except:
                        pass
                    idx += 1
                
                if 'Attribute_Q' in var and var.get('Attribute_Q'):
                    try:
                        elem.SetAttribute(var['Attribute_Q'], float(gen_values[idx]))
                    except:
                        pass
                    idx += 1
    
    def _apply_line_settings(self, lines_to_disable):
        """Ustaw linie"""
        for line_name in self.candidate_lines:
            line = self._line_cache.get(line_name)
            if line:
                try:
                    line.outserv = 0
                except:
                    pass
        
        for line_name in lines_to_disable:
            line = self._line_cache.get(line_name)
            if line:
                try:
                    line.outserv = 1
                except:
                    pass
    
    def _check_island(self):
        """Sprawdź wyspy"""
        
        debug_log = os.path.join(HYBRID_OUT_DIR, "hybrid_objective_debug.txt")
        
        def write_debug(msg):
            try:
                with open(debug_log, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
            except:
                pass
        
        try:
            code = self.ldf.Execute()
            write_debug(f"    LF code:{code}")
            
            if code >= 2:
                write_debug(f"    LF nie zbiegł → WYSPA")
                return True
            
            if code == 0:
                write_debug(f"    LF OK → brak wyspy")
                return False
            
            # Kod 1 - sprawdź napięcia
            buses = self.app.GetCalcRelevantObjects("*.ElmTerm")
            total_buses = len(buses)
            
            if total_buses == 0:
                write_debug(f"    Brak węzłów → WYSPA")
                return True
            
            low_voltage_count = 0
            min_voltage = 999.0
            
            for bus in buses:
                try:
                    u_pu = bus.GetAttribute("m:u")
                    if u_pu is not None:
                        min_voltage = min(min_voltage, u_pu)
                        if u_pu < 0.01:
                            low_voltage_count += 1
                except:
                    pass
            
            write_debug(f"    Min U:{min_voltage:.3f}, Low:{low_voltage_count}")
            
            if low_voltage_count > total_buses * 0.05:
                write_debug(f"    → WYSPA")
                return True
            
            if min_voltage > 0.7:
                write_debug(f"    → OK")
                return False
            
            if min_voltage < 0.5:
                write_debug(f"    → WYSPA")
                return True
            
            write_debug(f"    → OK")
            return False
        
        except Exception as e:
            write_debug(f"    EXCEPTION:{e}")
            return True
    
    def _calculate_current_overload(self):
        """Oblicz przeciążenia - TYLKO obserwowanych"""
        
        debug_log = os.path.join(HYBRID_OUT_DIR, "hybrid_objective_debug.txt")
        
        def write_debug(msg):
            try:
                with open(debug_log, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
            except:
                pass
        
        try:
            overload = 0.0
            lines = self.app.GetCalcRelevantObjects("*.ElmLne")
            
            if not lines:
                write_debug("    ⚠️ Brak linii!")
                return np.inf
            
            write_debug(f"    Linii:{len(lines)}")
            
            found = 0
            details = []
            
            for line in lines:
                if line.loc_name in self.observed_lines:
                    found += 1
                    try:
                        loading = line.GetAttribute("c:loading")
                        
                        if loading is None:
                            continue
                        
                        if loading > 100:
                            excess = loading - 100
                            overload += excess
                            details.append(f"{line.loc_name}:{loading:.2f}%")
                        
                    except:
                        pass
            
            write_debug(f"    Znaleziono:{found}, Przeciążonych:{len(details)}")
            write_debug(f"    Suma:{overload:.3f}")
            
            return overload
        
        except Exception as e:
            write_debug(f"    EXCEPTION:{e}")
            return np.inf
    
    def __call__(self, x):
        """Funkcja celu - FINALNA"""
        self.eval_count += 1
        
        debug_log = os.path.join(HYBRID_OUT_DIR, "hybrid_objective_debug.txt")
        
        def write_debug(msg):
            try:
                with open(debug_log, 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
            except:
                pass
        
        show_debug = (self.eval_count <= 10)
        
        if self.eval_count == 1:
            write_debug("="*80)
            write_debug("HYBRID OBJECTIVE - RANKING + Min/Max")
            write_debug("="*80)
            write_debug(f"Overload_base:{self.overload_base:.3f}")
            write_debug(f"Zakres wyłączeń:[{self.min_lines_out}, {self.max_lines_out}]")
            write_debug(f"Wagi:w1={self.w1}, w2={self.w2}")
            write_debug(f"Funkcja:f = w1*(N/Max) + w2*(Overload/Base)")
            write_debug("="*80 + "\n")
        
        try:
            if show_debug:
                write_debug(f"\n{'='*60}")
                write_debug(f"EVAL #{self.eval_count}")
                write_debug(f"{'='*60}")
            
            # Dekoduj
            try:
                gen_values, lines_to_disable = self._decode_vector(x)
                if show_debug:
                    write_debug(f"  N wyłączeń:{len(lines_to_disable)}")
                    write_debug(f"  Linie:{lines_to_disable}")
            except Exception as e:
                write_debug(f"  ❌ Dekodowanie:{e}")
                return np.inf
            
            # Generatory
            try:
                self._apply_generator_settings(gen_values)
            except Exception as e:
                write_debug(f"  ❌ Generatory:{e}")
                return np.inf
            
            # Linie
            try:
                self._apply_line_settings(lines_to_disable)
            except Exception as e:
                write_debug(f"  ❌ Linie:{e}")
                return np.inf
            
            # Wyspa
            try:
                if self._check_island():
                    if show_debug:
                        write_debug(f"  WYSPA → inf")
                    return np.inf
            except Exception as e:
                write_debug(f"  ❌ Wyspa:{e}")
                return np.inf
            
            # Przeciążenia
            try:
                overload_current = self._calculate_current_overload()
            except Exception as e:
                write_debug(f"  ❌ Overload:{e}")
                return np.inf
            
            # Funkcja celu
            try:
                n_disabled = len(lines_to_disable)
                
                # Składnik 1:ZNORMALIZOWANY
                if self.max_lines_out > 0:
                    term1 = n_disabled / self.max_lines_out
                else:
                    term1 = float(n_disabled)
                
                # Składnik 2:Przeciążenia
                if self.overload_base > 0:
                    term2 = overload_current / self.overload_base
                else:
                    term2 = overload_current / 100.0
                
                f_total = self.w1 * term1 + self.w2 * term2
                
                if show_debug:
                    write_debug(f"\n  SKŁADNIKI:")
                    write_debug(f"    n_disabled = {n_disabled}")
                    write_debug(f"    zakres = [{self.min_lines_out}, {self.max_lines_out}]")
                    write_debug(f"    term1 = {n_disabled}/{self.max_lines_out} = {term1:.6f}")
                    write_debug(f"    overload = {overload_current:.3f}/{self.overload_base:.3f}")
                    write_debug(f"    term2 = {term2:.6f}")
                    write_debug(f"    f = {self.w1}*{term1:.6f} + {self.w2}*{term2:.6f} = {f_total:.6f}")
                
                if np.isnan(f_total) or np.isinf(f_total):
                    return np.inf
                
            except Exception as e:
                write_debug(f"  ❌ f_celu:{e}")
                return np.inf
            
            # Best tracking
            if f_total < self.best_value:
                self.best_value = f_total
                self.best_x = x.copy()
                
                msg = f"\n✅ BEST #{self.eval_count}:f={f_total:.6f} (n={n_disabled}, topo={term1:.4f}, over={term2:.4f})"
                write_debug(msg)
                
                try:
                    self.app.PrintPlain(msg)
                except:
                    pass
            
            return f_total
        
        except Exception as e:
            write_debug(f"\n❌ UNEXPECTED:{e}")
            import traceback
            write_debug(traceback.format_exc())
            return np.inf

# ==========================================
# GŁÓWNA FUNKCJA
# ==========================================

def run_hybrid_optimization(app, ldf, excel_file, out_dir, scenario_name="HYBRID"):
    """Uruchom hybrydową optymalizację"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(HYBRID_OUT_DIR, f"hybrid_log_{timestamp}.txt")
    error_file = os.path.join(HYBRID_OUT_DIR, f"hybrid_ERROR_{timestamp}.txt")
    
    global _logger
    
    try:
        _logger = Logger(log_file, app)
    except Exception as e:
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"LOGGER ERROR:{e}\n")
        return None
    
    def safe_log(msg):
        try:
            log(msg)
        except:
            pass
        try:
            app.PrintPlain(str(msg))
        except:
            pass
    
    try:
        safe_log("="*80)
        safe_log("HYBRYDOWA OPTYMALIZACJA - PEŁNY EKSPORT USTAWIEŃ")
        safe_log("="*80)
        safe_log(f"Scenariusz:{scenario_name}")
        safe_log("="*80)
        
        # [1/7] Config
        safe_log("\n[1/7] Konfiguracja...")
        
        opt_variables, candidate_lines, min_lines_out, max_lines_out = load_hybrid_config(excel_file)
        
        if not opt_variables and not candidate_lines:
            safe_log("❌ Brak konfiguracji")
            return None
        
        safe_log(f"  ✓ Gen:{len(opt_variables)}")
        safe_log(f"  ✓ Lines:{len(candidate_lines)}")
        safe_log(f"  ✓ Zakres wyłączeń:[{min_lines_out}, {max_lines_out}]")
        
        # [2/7] Stan bazowy
        safe_log("\n[2/7] Stan bazowy...")
        
        all_lines = app.GetCalcRelevantObjects("*.ElmLne")
        for line in all_lines:
            try:
                line.outserv = 0
            except:
                pass
        
        stats_before = get_observed_lines_stats(app, ldf, OBSERVED_LINES)
        
        safe_log(f"  ✓ Przeciążenia:{stats_before['total_overload']:.3f}")
        safe_log(f"  ✓ Przeciążonych:{stats_before['overloaded_count']}")
        
        # ✅ EKSPORT STANU PRZED
        safe_log("\n📋 Eksport ustawień PRZED optymalizacją...")
        settings_before = export_all_settings(app, "BEFORE")
        
        overload_base = stats_before['total_overload']
        
        # ✅ WALIDACJA - zabezpieczenie przed zerem
        if overload_base <= 0:
            safe_log("\n" + "="*80)
            safe_log("⚠️ UWAGA:BRAK PRZECIĄŻEŃ W STANIE BAZOWYM")
            safe_log("="*80)
            safe_log("Sieć już jest w dobrym stanie - optymalizacja może nie być potrzebna.")
            safe_log("Ustawiam overload_base = 100.0 (wartość domyślna do normalizacji)")
            safe_log("="*80)
            overload_base = 100.0
        
        # [3/7] Funkcja celu
        safe_log("\n[3/7] Funkcja celu...")
        
        objective = HybridObjective(
            app, ldf,
            opt_variables,
            candidate_lines,
            min_lines_out,
            max_lines_out,
            OBSERVED_LINES,
            overload_base,
            w1=WEIGHT_TOPOLOGY,
            w2=WEIGHT_OVERLOAD
        )
        safe_log(f"  ✓ OK")
        
        # [4/7] Granice
        safe_log("\n[4/7] Granice...")
        
        lb = []
        ub = []
        var_names = []
        
        # Generatory
        for var in opt_variables:
            if 'Pmin' in var and 'Pmax' in var:
                lb.append(float(var['Pmin']))
                ub.append(float(var['Pmax']))
                var_names.append(f"{var['Element_Name']}_P")
            if 'Qmin' in var and 'Qmax' in var:
                lb.append(float(var['Qmin']))
                ub.append(float(var['Qmax']))
                var_names.append(f"{var['Element_Name']}_Q")
        
        # Topologia:1 zmienna = liczba + N = ranking
        lb.append(0.0)
        ub.append(1.0)
        var_names.append("N_lines_to_disable")
        
        for line_name in candidate_lines:
            lb.append(0.0)
            ub.append(1.0)
            var_names.append(f"Score_{line_name}")
        
        lb = np.array(lb)
        ub = np.array(ub)
        
        safe_log(f"  ✓ Dim:{len(lb)}")
        safe_log(f"     ({objective.n_gen_vars} gen + {objective.n_line_vars} topo)")
        
        # [5/7] PSO
        safe_log("\n[5/7] PSO z Early Stopping...")
        safe_log(f"  Cząstek:{HYBRID_PSO_PARAMS['n_particles']}")
        safe_log(f"  Iteracji:{HYBRID_PSO_PARAMS['max_iter']}")
        safe_log(f"  🎯 Target:f <= {HYBRID_PSO_PARAMS['early_stop_threshold']}")
        safe_log(f"  ⏸️ Patience:{HYBRID_PSO_PARAMS['early_stop_patience']} iter")
        
        pso = PSO(
            func=objective,
            n_particles=HYBRID_PSO_PARAMS['n_particles'],
            dim=len(lb),
            lb=lb,
            ub=ub,
            max_iter=HYBRID_PSO_PARAMS['max_iter'],
            w=HYBRID_PSO_PARAMS['w'],
            c1=HYBRID_PSO_PARAMS['c1'],
            c2=HYBRID_PSO_PARAMS['c2'],
            autosave_every_iters=HYBRID_PSO_PARAMS['autosave_every'],
            autosave_path=os.path.join(HYBRID_OUT_DIR, f"hybrid_checkpoint_{timestamp}.npz"),
            early_stop_threshold=HYBRID_PSO_PARAMS['early_stop_threshold'],
            early_stop_patience=HYBRID_PSO_PARAMS['early_stop_patience']
        )
        
        time_start = time.time()
        result = pso.optimize()
        time_end = time.time()
        
        # ✅ Raportowanie early stopping
        safe_log(f"\n✅ Zakończono:{time_end - time_start:.2f}s")
        
        if result.get('early_stopped'):
            reason = result.get('reason')
            stopped_at = result.get('stopped_at_iter')
            safe_log(f"\n⏸️ EARLY STOPPING:")
            if reason == 'threshold_reached':
                safe_log(f"   🎯 Osiągnięto cel (f <= {HYBRID_PSO_PARAMS['early_stop_threshold']})")
            elif reason == 'no_improvement':
                safe_log(f"   ⏸️ Brak poprawy przez {HYBRID_PSO_PARAMS['early_stop_patience']} iteracji")
            safe_log(f"   Zatrzymano na iteracji:{stopped_at}/{HYBRID_PSO_PARAMS['max_iter']}")
        
        safe_log(f"F_best:{result['gbest_val']:.6f}")
        
        # [6/7] Wyniki - zastosuj najlepsze rozwiązanie
        safe_log("\n[6/7] Zastosowanie najlepszego rozwiązania...")
        
        gen_best, lines_best = objective._decode_vector(result['gbest'])
        objective._apply_generator_settings(gen_best)
        objective._apply_line_settings(lines_best)
        
        stats_after = get_observed_lines_stats(app, ldf, OBSERVED_LINES)
        
        # ✅ EKSPORT STANU PO
        safe_log("\n📋 Eksport ustawień PO optymalizacji...")
        settings_after = export_all_settings(app, "AFTER")
        
        safe_log(f"\n📊 PORÓWNANIE:")
        safe_log(f"  Przed:{stats_before['total_overload']:.3f} ({stats_before['overloaded_count']} linii)")
        safe_log(f"  Po:{stats_after['total_overload']:.3f} ({stats_after['overloaded_count']} linii)")
        safe_log(f"  Wyłączonych linii:{len(lines_best)}")
        
        # ✅ ZABEZPIECZONA KALKULACJA ZMIANY
        if stats_before['total_overload'] > 0:
            change = stats_after['total_overload'] - stats_before['total_overload']
            change_pct = (change / stats_before['total_overload']) * 100
            
            if change < 0:
                safe_log(f"  ✅ REDUKCJA:{abs(change):.3f} ({abs(change_pct):.2f}%)")
            elif change > 0:
                safe_log(f"  ⚠️ WZROST:{change:.3f} (+{change_pct:.2f}%)")
            else:
                safe_log(f"  ➡️ BEZ ZMIANY:{stats_after['total_overload']:.3f}")
        else:
            if stats_after['total_overload'] > 0:
                safe_log(f"  ⚠️ POJAWIŁY SIĘ PRZECIĄŻENIA:{stats_after['total_overload']:.3f}")
            else:
                safe_log(f"  ✅ BRAK PRZECIĄŻEŃ (przed i po)")
        
        # [7/7] Excel
        safe_log("\n[7/7] Zapisywanie do Excel...")
        output_file = os.path.join(HYBRID_OUT_DIR, f"HYBRID_{scenario_name}_{timestamp}.xlsx")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # ✅ ZABEZPIECZONE PODSUMOWANIE
            if stats_before['total_overload'] > 0:
                reduction_value = (stats_before['total_overload'] - stats_after['total_overload']) / stats_before['total_overload'] * 100
                reduction_str = f"{reduction_value:.2f}%"
            else:
                if stats_after['total_overload'] > 0:
                    reduction_str = f"N/A (przed=0, po={stats_after['total_overload']:.3f})"
                else:
                    reduction_str = "0% (brak przeciążeń)"
            
            # Informacja o early stopping
            early_stop_info = ""
            if result.get('early_stopped'):
                reason = result.get('reason')
                if reason == 'threshold_reached':
                    early_stop_info = "Cel osiągnięty"
                elif reason == 'no_improvement':
                    early_stop_info = "Brak poprawy"
                else:
                    early_stop_info = "Early stop"
            else:
                early_stop_info = "Wszystkie iteracje"
            
            # Podsumowanie
            summary = {
                'Parametr':[
                    'Scenariusz',
                    'Czas [s]',
                    'Status',
                    'Iteracji wykonanych',
                    '',
                    'Przeciążenia przed',
                    'Przeciążenia po',
                    'Redukcja',
                    '',
                    'Linii wyłączonych',
                    'Zakres dozwolony',
                    '',
                    'f_celu',
                ],
                'Wartość':[
                    scenario_name,
                    f"{time_end - time_start:.2f}",
                    early_stop_info,
                    f"{result.get('stopped_at_iter', HYBRID_PSO_PARAMS['max_iter'])}/{HYBRID_PSO_PARAMS['max_iter']}",
                    '',
                    f"{stats_before['total_overload']:.3f}",
                    f"{stats_after['total_overload']:.3f}",
                    reduction_str,
                    '',
                    len(lines_best),
                    f"[{min_lines_out}, {max_lines_out}]",
                    '',
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
            
            # ✅ USTAWIENIA PRZED - WSZYSTKIE ELEMENTY
            safe_log("  Zapisywanie:Ustawienia PRZED...")
            
            # Połącz wszystkie generujące elementy (ElmSym, ElmGenstat, ElmPvsys)
            all_gen_before = []
            all_gen_before.extend(settings_before.get('ElmSym', []))
            all_gen_before.extend(settings_before.get('ElmGenstat', []))
            all_gen_before.extend(settings_before.get('ElmPvsys', []))
            
            if all_gen_before:
                df_gen_before = pd.DataFrame(all_gen_before)
                df_gen_before.to_excel(writer, sheet_name='BEFORE_Generators', index=False)
            
            # Obciążenia PRZED
            if settings_before.get('ElmLod'):
                df_load_before = pd.DataFrame(settings_before['ElmLod'])
                df_load_before.to_excel(writer, sheet_name='BEFORE_Loads', index=False)
            
            # Linie PRZED
            if settings_before.get('ElmLne'):
                df_line_before = pd.DataFrame(settings_before['ElmLne'])
                df_line_before.to_excel(writer, sheet_name='BEFORE_Lines', index=False)
            
            # Transformatory PRZED
            if settings_before.get('ElmTr2'):
                df_trafo_before = pd.DataFrame(settings_before['ElmTr2'])
                df_trafo_before.to_excel(writer, sheet_name='BEFORE_Transformers', index=False)
            
            # ✅ USTAWIENIA PO - WSZYSTKIE ELEMENTY
            safe_log("  Zapisywanie:Ustawienia PO...")
            
            # Połącz wszystkie generujące elementy (ElmSym, ElmGenstat, ElmPvsys)
            all_gen_after = []
            all_gen_after.extend(settings_after.get('ElmSym', []))
            all_gen_after.extend(settings_after.get('ElmGenstat', []))
            all_gen_after.extend(settings_after.get('ElmPvsys', []))
            
            if all_gen_after:
                df_gen_after = pd.DataFrame(all_gen_after)
                df_gen_after.to_excel(writer, sheet_name='AFTER_Generators', index=False)
            
            # Obciążenia PO
            if settings_after.get('ElmLod'):
                df_load_after = pd.DataFrame(settings_after['ElmLod'])
                df_load_after.to_excel(writer, sheet_name='AFTER_Loads', index=False)
            
            # Linie PO
            if settings_after.get('ElmLne'):
                df_line_after = pd.DataFrame(settings_after['ElmLne'])
                df_line_after.to_excel(writer, sheet_name='AFTER_Lines', index=False)
            
            # Transformatory PO
            if settings_after.get('ElmTr2'):
                df_trafo_after = pd.DataFrame(settings_after['ElmTr2'])
                df_trafo_after.to_excel(writer, sheet_name='AFTER_Transformers', index=False)
            
            # ✅ ZMIANY (tylko elementy optymalizowane)
            safe_log("  Zapisywanie:Zmiany optymalizowanych elementów...")
            
            gen_changes = []
            idx = 0
            for var in opt_variables:
                row = {'Element':var['Element_Name'], 'Type':var['Element_Type']}
                if 'Attribute_P' in var and var.get('Attribute_P'):
                    row['P_AFTER [MW]'] = f"{gen_best[idx]:.6f}"
                    idx += 1
                if 'Attribute_Q' in var and var.get('Attribute_Q'):
                    row['Q_AFTER [Mvar]'] = f"{gen_best[idx]:.6f}"
                    idx += 1
                gen_changes.append(row)
            
            pd.DataFrame(gen_changes).to_excel(writer, sheet_name='Optimized_Generators', index=False)
            
            # Linie - status
            lines_status = []
            for line_name in candidate_lines:
                status = 'DISABLED' if line_name in lines_best else 'ACTIVE'
                lines_status.append({'Line':line_name, 'Status_AFTER':status})
            pd.DataFrame(lines_status).to_excel(writer, sheet_name='Optimized_Lines', index=False)
            
            # ✅ ZABEZPIECZONE PORÓWNANIE
            if stats_before['total_overload'] > 0:
                comparison_change = f"{(stats_after['total_overload'] - stats_before['total_overload']) / stats_before['total_overload'] * 100:.2f}%"
            else:
                comparison_change = "N/A (przed=0)"
            
            comparison = {
                'Metryka':[
                    'Suma przeciążeń',
                    'Linii przeciążonych',
                    'Max przeciążenie',
                    'Linii wyłączonych',
                ],
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
                    comparison_change,
                    f"{stats_after['overloaded_count'] - stats_before['overloaded_count']:+d}",
                    'N/A',
                    f"+{len(lines_best)}",
                ]
            }
            pd.DataFrame(comparison).to_excel(writer, sheet_name='Comparison', index=False)
        
        safe_log(f"✅ {output_file}")
        safe_log(f"\n📊 Zapisano arkusze:")
        safe_log(f"   - Podsumowanie, History, Comparison")
        safe_log(f"   - BEFORE_Generators, BEFORE_Loads, BEFORE_Lines, BEFORE_Transformers")
        safe_log(f"   - AFTER_Generators, AFTER_Loads, AFTER_Lines, AFTER_Transformers")
        safe_log(f"   - Optimized_Generators, Optimized_Lines")
        
        return result
    
    except Exception as e:
        error_msg = f"\n❌ BŁĄD:{e}\n"
        import traceback
        error_msg += traceback.format_exc()
        
        try:
            log(error_msg)
        except:
            pass
        
        return None
    
    finally:
        if _logger:
            _logger.close()

# ==========================================
# MAIN
# ==========================================

def main():
    print("\n" + "="*80)
    print("🚀 HYBRID - PEŁNY EKSPORT USTAWIEŃ")
    print("="*80)
    
    try:
        import powerfactory
        app = powerfactory.GetApplicationExt()
        
        if app is None:
            print("❌ Brak PF")
            return
        
        prj = app.GetActiveProject()
        if prj:
            print(f"✓ Projekt:{prj.loc_name}")
        
        ldf = app.GetFromStudyCase("ComLdf")
        if ldf:
            print(f"✓ LF OK")
        
        result = run_hybrid_optimization(app, ldf, EXCEL_FILE, OUT_DIR, "N1_Hybrid")
        
        if result:
            print(f"\n✅ F_best:{result['gbest_val']:.6f}")
            if result.get('early_stopped'):
                print(f"⏸️ Stopped early:{result.get('reason')}")
            print(f"📁 {HYBRID_OUT_DIR}")
    
    except Exception as e:
        print(f"\n❌ BŁĄD:{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()