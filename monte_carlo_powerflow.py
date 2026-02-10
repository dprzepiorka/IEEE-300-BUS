"""
Monte Carlo Power Flow Analysis dla IEEE-300 BUS
Wersja z poprawkami:
- PV używa pgini/qgini (nie p_set)
- Transformatory ElmTr2 i ElmTr3
- Excel z wartościami nastawianymi i przeciążeniami
"""

import sys
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# Scipy dla rozkładu obciętego
try:
    from scipy.stats import truncnorm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy nie zainstalowane - TruncatedNormal niedostępny")

# PowerFactory API
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP5A\Python\3.12")
try:
    import powerfactory
except ImportError:
    powerfactory = None
    print("⚠️ Moduł powerfactory niedostępny - tryb testowy")

# ==========================================
# KONFIGURACJA
# ==========================================
class Config:
    # Ścieżki
    EXCEL_FILE = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Wysokie napiecie\PYTHON\dane_IEEE300_stany.xlsx"
    OUT_DIR = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Wysokie napiecie\PYTHON\Wyniki"
    
    # PowerFactory
    PROJECT_NAME = "IEEE300AKT"
    USER = "KE"
    
    # Monte Carlo
    MAX_ITERATIONS = 1000
    
    # Warunki akceptacji (None = wyłączone)
    MIN_GEN_POWER = None  # Wyłączone na razie
    MAX_GEN_POWER = None
    
    # Próg przeciążenia linii
    OVERLOAD_THRESHOLD = 100.0  # %

CONFIG = Config()

# ==========================================
# LOGGER
# ==========================================
class Logger:
    def __init__(self, filename, pf_app=None):
        self.log_file = open(filename, 'w', encoding='utf-8')
        self.pf_app = pf_app
        self.start_time = time.time()
        
        header = "="*80 + "\n"
        header += f"MONTE CARLO POWER FLOW - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += "="*80 + "\n\n"
        self.write(header)
    
    def write(self, message):
        timestamp = f"[{time.time() - self.start_time:8.2f}s] "
        full_msg = timestamp + str(message)
        self.log_file.write(full_msg + "\n")
        self.log_file.flush()
        print(message)
        
        if self.pf_app:
            try:
                self.pf_app.PrintPlain(str(message))
            except:
                pass
    
    def close(self):
        elapsed = time.time() - self.start_time
        self.write(f"\n{'='*80}")
        self.write(f"Ukończone w: {elapsed/3600:.2f}h ({elapsed:.1f}s)")
        self.write(f"{'='*80}")
        self.log_file.close()

_logger = None

def log(message):
    if _logger:
        _logger.write(message)
    else:
        print(message)

# ==========================================
# FUNKCJE POMOCNICZE POWERFACTORY
# ==========================================
def find_element_multi_method(app, name, pf_class):
    """Wielometodowe szukanie elementu w PowerFactory"""
    
    # METODA 1: GetCalcRelevantObjects z dokładną nazwą
    try:
        objs = app.GetCalcRelevantObjects(f"{name}.{pf_class}")
        if objs and len(objs) > 0:
            return objs[0]
    except:
        pass
    
    # METODA 2: Wildcard search
    try:
        all_objs = app.GetCalcRelevantObjects(f"*.{pf_class}")
        if all_objs:
            for obj in all_objs:
                if hasattr(obj, 'loc_name') and obj.loc_name == name:
                    return obj
    except:
        pass
    
    # METODA 3: Przez Study Case
    try:
        study_case = app.GetActiveStudyCase()
        if study_case:
            contents = study_case.GetContents(f"*.{pf_class}", 1)
            if contents:
                for obj in contents:
                    if hasattr(obj, 'loc_name') and obj.loc_name == name:
                        return obj
    except:
        pass
    
    # METODA 4: Przeszukiwanie całego projektu
    try:
        prj = app.GetActiveProject()
        if prj:
            all_folders = prj.GetContents('*.IntFolder', 1)
            for folder in all_folders:
                contents = folder.GetContents(f'*.{pf_class}', 1)
                if contents:
                    for obj in contents:
                        if hasattr(obj, 'loc_name') and obj.loc_name == name:
                            return obj
    except:
        pass
    
    return None

def set_element_parameter(elm, param, value):
    """Bezpieczne ustawienie parametru elementu"""
    try:
        elm.SetAttribute(param, float(value))
        return True
    except Exception as e:
        return False

# ==========================================
# KLASA DO ZARZĄDZANIA DANYMI
# ==========================================
class PowerSystemData:
    """Przechowuje dane bazowe i zarządza losowaniem"""
    
    def __init__(self, excel_file):
        self.excel_file = excel_file
        
        # DataFrames z danymi bazowymi
        self.loads_base = None
        self.generators_base = None
        self.pv_base = None
        self.statgen_base = None
        
        # Definicje losowania z arkusza Zakresy
        self.randomization_rules = None
        
        # Aktualne wartości (po losowaniu)
        self.loads_current = None
        self.generators_current = None
        self.pv_current = None
        self.statgen_current = None
        
        self.load_base_data()
        self.load_randomization_rules()
    
    def load_base_data(self):
        """Wczytaj dane bazowe z Excela"""
        log("\n" + "="*80)
        log("📂 WCZYTYWANIE DANYCH BAZOWYCH Z EXCELA")
        log("="*80)
        
        try:
            self.loads_base = pd.read_excel(self.excel_file, sheet_name="Loads")
            log(f"✓ Loads: {len(self.loads_base)} elementów")
            
            self.generators_base = pd.read_excel(self.excel_file, sheet_name="Generators")
            log(f"✓ Generators: {len(self.generators_base)} elementów")
            
            try:
                self.pv_base = pd.read_excel(self.excel_file, sheet_name="PV")
                log(f"✓ PV: {len(self.pv_base)} elementów")
            except:
                self.pv_base = pd.DataFrame()
                log("  (brak arkusza PV)")
            
            try:
                self.statgen_base = pd.read_excel(self.excel_file, sheet_name="StatGen")
                log(f"✓ StatGen: {len(self.statgen_base)} elementów")
            except:
                self.statgen_base = pd.DataFrame()
                log("  (brak arkusza StatGen)")
            
            # Inicjalizuj current jako kopie base
            self.loads_current = self.loads_base.copy()
            self.generators_current = self.generators_base.copy()
            self.pv_current = self.pv_base.copy()
            self.statgen_current = self.statgen_base.copy()
            
        except Exception as e:
            log(f"❌ Błąd wczytywania danych: {e}")
            raise
    
    def load_randomization_rules(self):
        """Wczytaj reguły losowania z arkusza Zakresy"""
        log("\n" + "="*80)
        log("🎲 WCZYTYWANIE REGUŁ LOSOWANIA (arkusz Zakresy)")
        log("="*80)
        
        try:
            self.randomization_rules = pd.read_excel(
                self.excel_file, 
                sheet_name="Zakresy"
            )
            log(f"✓ Zakresy: {len(self.randomization_rules)} reguł losowania")
            
            # Dodaj brakujące kolumny opcjonalne
            optional_cols = ['StdDev', 'Min', 'Max']
            for col in optional_cols:
                if col not in self.randomization_rules.columns:
                    self.randomization_rules[col] = np.nan
            
            # Wyświetl przykładowe reguły
            log("\nPrzykładowe reguły:")
            for idx, row in self.randomization_rules.head(5).iterrows():
                min_str = f"{row['Min']:8.2f}" if pd.notna(row['Min']) else "     N/A"
                max_str = f"{row['Max']:8.2f}" if pd.notna(row['Max']) else "     N/A"
                std_str = f"{row['StdDev']:6.3f}" if pd.notna(row['StdDev']) else "   N/A"
                
                log(f"  {row['Element_Name']:20s} | {row['Element_Type']:12s} | "
                    f"{row['Parameter']:8s} | Base={row['BaseValue']:8.2f} | "
                    f"Std={std_str} | {row['Distribution']:15s} | [{min_str}, {max_str}]")
            
        except Exception as e:
            log(f"❌ Błąd wczytywania Zakresy: {e}")
            self.randomization_rules = pd.DataFrame()
    
    def randomize_values(self, iteration_nr):
        """Losuj wartości według rozkładów z arkusza Zakresy"""
        log(f"\n🎲 Losowanie wartości (iteracja {iteration_nr})...")
        
        if self.randomization_rules is None or len(self.randomization_rules) == 0:
            log("  ⚠️ Brak reguł losowania - używam wartości bazowych")
            return
        
        randomized_count = 0
        stats = {
            'uniform': 0,
            'normal': 0,
            'truncated_normal': 0,
            'rand': 0,
            'errors': 0
        }
        
        for idx, rule in self.randomization_rules.iterrows():
            elem_name = str(rule['Element_Name']).strip()
            elem_type = str(rule['Element_Type']).strip()
            param = str(rule['Parameter']).strip()
            base_value = float(rule['BaseValue'])
            distribution = str(rule['Distribution']).strip().lower()
            
            std_dev = float(rule['StdDev']) if pd.notna(rule['StdDev']) else 0.0
            min_val = float(rule['Min']) if pd.notna(rule['Min']) else None
            max_val = float(rule['Max']) if pd.notna(rule['Max']) else None
            
            new_value = None
            
            try:
                if distribution == 'uniform':
                    if min_val is not None and max_val is not None:
                        new_value = np.random.uniform(min_val, max_val)
                        stats['uniform'] += 1
                    else:
                        stats['errors'] += 1
                        continue
                
                elif distribution == 'normal':
                    if elem_type == 'ElmLod':
                        multiplier = np.random.rand()
                        new_value = base_value * multiplier
                    elif elem_type in ['ElmSym', 'ElmPvsys']:
                        multiplier = np.random.normal(1.0, std_dev)
                        new_value = base_value * multiplier
                    elif elem_type == 'ElmGenstat':
                        new_value = np.random.normal(base_value, std_dev)
                    else:
                        stats['errors'] += 1
                        continue
                    
                    if min_val is not None:
                        new_value = max(new_value, min_val)
                    if max_val is not None:
                        new_value = min(new_value, max_val)
                    
                    stats['normal'] += 1
                
                elif distribution in ['truncated_normal', 'truncatednormal', 'truncnorm']:
                    if not SCIPY_AVAILABLE:
                        stats['errors'] += 1
                        continue
                    
                    if min_val is not None and max_val is not None and std_dev > 0:
                        a = (min_val - base_value) / std_dev
                        b = (max_val - base_value) / std_dev
                        new_value = truncnorm.rvs(a, b, loc=base_value, scale=std_dev)
                        stats['truncated_normal'] += 1
                    else:
                        stats['errors'] += 1
                        continue
                
                elif distribution == 'rand':
                    new_value = base_value * np.random.rand()
                    stats['rand'] += 1
                
                else:
                    stats['errors'] += 1
                    continue
                
                if new_value is not None:
                    self._update_current_value(elem_name, elem_type, param, new_value)
                    randomized_count += 1
            
            except Exception as e:
                stats['errors'] += 1
        
        log(f"  ✓ Wylosowano {randomized_count} wartości:")
        log(f"    - Uniform: {stats['uniform']}")
        log(f"    - Normal: {stats['normal']}")
        log(f"    - TruncatedNormal: {stats['truncated_normal']}")
        log(f"    - Rand: {stats['rand']}")
        if stats['errors'] > 0:
            log(f"    - Błędy: {stats['errors']}")
    
    def _update_current_value(self, elem_name, elem_type, param, value):
        """Aktualizuj wartość w odpowiednim DataFrame"""
        
        if elem_type == 'ElmLod':
            df = self.loads_current
        elif elem_type == 'ElmSym':
            df = self.generators_current
        elif elem_type == 'ElmPvsys':
            df = self.pv_current
        elif elem_type == 'ElmGenstat':
            df = self.statgen_current
        else:
            return
        
        mask = df['name'] == elem_name
        if not mask.any():
            mask = df['name'].str.lower() == elem_name.lower()
        
        if mask.any():
            param_map = {
                'plini': 'P',
                'qlini': 'Q',
                'pgini': 'P',
                'qgini': 'Q',
            }
            
            col = param_map.get(param, param)
            
            if col in df.columns:
                df.loc[mask, col] = value

# ==========================================
# KLASA DO KOMUNIKACJI Z POWERFACTORY
# ==========================================
class PowerFactoryInterface:
    """Interfejs do PowerFactory"""
    
    def __init__(self, app, project_name):
        self.app = app
        self.project_name = project_name
        self.ldf = None
        
        self.activate_project()
        self.get_load_flow_object()
    
    def activate_project(self):
        """Aktywuj projekt"""
        log(f"\n🔌 Aktywacja projektu: {self.project_name}")
        try:
            self.app.ActivateProject(self.project_name)
            log("  ✓ Projekt aktywowany")
        except Exception as e:
            log(f"  ❌ Błąd aktywacji projektu: {e}")
            raise
    
    def get_load_flow_object(self):
        """Pobierz obiekt rozpływu mocy"""
        try:
            self.ldf = self.app.GetFromStudyCase("ComLdf")
            if self.ldf:
                log("  ✓ Obiekt rozpływu mocy pobrany")
            else:
                log("  ⚠️ Nie znaleziono obiektu rozpływu mocy")
        except Exception as e:
            log(f"  ❌ Błąd pobierania obiektu LDF: {e}")
    
    def set_system_state(self, ps_data):
        """Ustaw stan systemu w PowerFactory"""
        log("\n📝 Zapisywanie danych do PowerFactory...")
        
        stats = {
            'loads_set': 0,
            'generators_set': 0,
            'pv_set': 0,
            'statgen_set': 0,
            'loads_not_found': [],
            'generators_not_found': [],
            'pv_not_found': [],
            'statgen_not_found': [],
        }
        
        # === OBCIĄŻENIA ===
        if len(ps_data.loads_current) > 0:
            log("  📦 Ustawianie obciążeń (ElmLod)...")
            for idx, row in ps_data.loads_current.iterrows():
                name = str(row['name']).strip()
                P = float(row['P'])
                Q = float(row['Q']) if 'Q' in row and pd.notna(row['Q']) else 0.0
                
                elm = find_element_multi_method(self.app, name, 'ElmLod')
                if elm:
                    ok_p = set_element_parameter(elm, 'plini', P)
                    ok_q = set_element_parameter(elm, 'qlini', Q)
                    if ok_p and ok_q:
                        stats['loads_set'] += 1
                else:
                    stats['loads_not_found'].append(name)
            
            log(f"    ✓ Ustawiono: {stats['loads_set']}/{len(ps_data.loads_current)}")
            if len(stats['loads_not_found']) > 0:
                log(f"    ⚠️ Nie znaleziono: {len(stats['loads_not_found'])} elementów")
        
        # === GENERATORY ===
        if len(ps_data.generators_current) > 0:
            log("  ⚡ Ustawianie generatorów (ElmSym)...")
            for idx, row in ps_data.generators_current.iterrows():
                name = str(row['name']).strip()
                P = float(row['P'])
                
                elm = find_element_multi_method(self.app, name, 'ElmSym')
                if elm:
                    if set_element_parameter(elm, 'pgini', P):
                        stats['generators_set'] += 1
                else:
                    stats['generators_not_found'].append(name)
            
            log(f"    ✓ Ustawiono: {stats['generators_set']}/{len(ps_data.generators_current)}")
            if len(stats['generators_not_found']) > 0:
                log(f"    ⚠️ Nie znaleziono: {len(stats['generators_not_found'])} elementów")
        
        # === PV (POPRAWKA: pgini i qgini) ===
        if len(ps_data.pv_current) > 0:
            log("  ☀️ Ustawianie PV (ElmPvsys)...")
            for idx, row in ps_data.pv_current.iterrows():
                name = str(row['name']).strip()
                P = float(row['P'])
                Q = float(row['Q']) if 'Q' in row and pd.notna(row['Q']) else 0.0
                
                elm = find_element_multi_method(self.app, name, 'ElmPvsys')
                if elm:
                    ok_p = set_element_parameter(elm, 'pgini', P)  # POPRAWKA!
                    ok_q = set_element_parameter(elm, 'qgini', Q)  # POPRAWKA!
                    if ok_p and ok_q:
                        stats['pv_set'] += 1
                else:
                    stats['pv_not_found'].append(name)
            
            log(f"    ✓ Ustawiono: {stats['pv_set']}/{len(ps_data.pv_current)}")
            if len(stats['pv_not_found']) > 0:
                log(f"    ⚠️ Nie znaleziono: {len(stats['pv_not_found'])} elementów")
        
        # === STATGEN (magazyny) ===
        if len(ps_data.statgen_current) > 0:
            log("  🔋 Ustawianie magazynów (ElmGenstat)...")
            for idx, row in ps_data.statgen_current.iterrows():
                name = str(row['name']).strip()
                P = float(row['P'])
                Q = float(row['Q']) if 'Q' in row and pd.notna(row['Q']) else 0.0
                
                elm = find_element_multi_method(self.app, name, 'ElmGenstat')
                if elm:
                    ok_p = set_element_parameter(elm, 'pgini', P)
                    ok_q = set_element_parameter(elm, 'qgini', Q)
                    if ok_p and ok_q:
                        stats['statgen_set'] += 1
                else:
                    stats['statgen_not_found'].append(name)
            
            log(f"    ✓ Ustawiono: {stats['statgen_set']}/{len(ps_data.statgen_current)}")
            if len(stats['statgen_not_found']) > 0:
                log(f"    ⚠️ Nie znaleziono: {len(stats['statgen_not_found'])} elementów")
        
        return stats
    
    def run_load_flow(self):
        """Uruchom rozpływ mocy"""
        if not self.ldf:
            log("  ❌ Brak obiektu rozpływu mocy")
            return False, -1
        
        try:
            time_start = time.time()
            code = self.ldf.Execute()
            time_end = time.time()
            
            if code == 0:
                log(f"  ✓ Rozpływ OK ({time_end - time_start:.3f}s)")
                return True, code
            else:
                log(f"  ⚠️ Rozpływ z błędem: kod {code}")
                return False, code
        
        except Exception as e:
            log(f"  ❌ Wyjątek podczas rozpływu: {e}")
            return False, -999
    
    def collect_results(self):
        """Zbierz wyniki rozpływu"""
        log("\n📊 Zbieranie wyników...")
        
        results = {
            'buses': [],
            'lines': [],
            'transformers': [],
            'generators': [],
            'loads': [],
            'pv': [],
            'statgen': [],
        }
        
        try:
            # === WĘZŁY ===
            buses = self.app.GetCalcRelevantObjects('*.ElmTerm')
            if buses:
                for bus in buses:
                    try:
                        u_kv = bus.GetAttribute('m:u')
                        uknom = bus.uknom if hasattr(bus, 'uknom') else 1.0
                        results['buses'].append({
                            'name': bus.loc_name,
                            'U_kV': u_kv,
                            'U_pu': u_kv / uknom if uknom > 0 else 0,
                            'phi_deg': bus.GetAttribute('m:phiu'),
                        })
                    except:
                        pass
            
            log(f"  ✓ Węzły: {len(results['buses'])}")
            
            # === LINIE ===
            lines = self.app.GetCalcRelevantObjects('*.ElmLne')
            if lines:
                for line in lines:
                    try:
                        loading = line.GetAttribute('c:loading')
                        
                        results['lines'].append({
                            'name': line.loc_name,
                            'I_A': line.GetAttribute('m:I:bus1'),
                            'loading_pct': loading,
                            'P_MW': line.GetAttribute('m:P:bus1'),
                            'Q_Mvar': line.GetAttribute('m:Q:bus1'),
                        })
                    except:
                        pass
            
            log(f"  ✓ Linie: {len(results['lines'])}")
            
            # === TRANSFORMATORY (ElmTr2 i ElmTr3) ===
            trafos_2 = self.app.GetCalcRelevantObjects('*.ElmTr2')
            trafos_3 = self.app.GetCalcRelevantObjects('*.ElmTr3')
            
            all_trafos = []
            if trafos_2:
                all_trafos.extend(trafos_2)
            if trafos_3:
                all_trafos.extend(trafos_3)
            
            for tr in all_trafos:
                try:
                    results['transformers'].append({
                        'name': tr.loc_name,
                        'type': tr.GetClassName(),
                        'loading_pct': tr.GetAttribute('c:loading'),
                        'P_MW': tr.GetAttribute('m:P:bushv'),
                    })
                except:
                    pass
            
            log(f"  ✓ Transformatory: {len(results['transformers'])} (ElmTr2 + ElmTr3)")
            
            # === GENERATORY ===
            generators = self.app.GetCalcRelevantObjects('*.ElmSym')
            if generators:
                for gen in generators:
                    try:
                        results['generators'].append({
                            'name': gen.loc_name,
                            'P_MW': gen.GetAttribute('m:P:bus1'),
                            'Q_Mvar': gen.GetAttribute('m:Q:bus1'),
                        })
                    except:
                        pass
            
            log(f"  ✓ Generatory: {len(results['generators'])}")
            
            # === OBCIĄŻENIA ===
            loads = self.app.GetCalcRelevantObjects('*.ElmLod')
            if loads:
                for load in loads:
                    try:
                        results['loads'].append({
                            'name': load.loc_name,
                            'P_MW': load.GetAttribute('m:P:bus1'),
                            'Q_Mvar': load.GetAttribute('m:Q:bus1'),
                        })
                    except:
                        pass
            
            log(f"  ✓ Obciążenia: {len(results['loads'])}")
            
            # === PV ===
            pvs = self.app.GetCalcRelevantObjects('*.ElmPvsys')
            if pvs:
                for pv in pvs:
                    try:
                        results['pv'].append({
                            'name': pv.loc_name,
                            'P_MW': pv.GetAttribute('m:P:bus1'),
                            'Q_Mvar': pv.GetAttribute('m:Q:bus1'),
                        })
                    except:
                        pass
            
            log(f"  ✓ PV: {len(results['pv'])}")
            
            # === STATGEN ===
            statgens = self.app.GetCalcRelevantObjects('*.ElmGenstat')
            if statgens:
                for sg in statgens:
                    try:
                        results['statgen'].append({
                            'name': sg.loc_name,
                            'P_MW': sg.GetAttribute('m:P:bus1'),
                            'Q_Mvar': sg.GetAttribute('m:Q:bus1'),
                        })
                    except:
                        pass
            
            log(f"  ✓ StatGen: {len(results['statgen'])}")
            
        except Exception as e:
            log(f"  ❌ Błąd zbierania wyników: {e}")
        
        return results
    
    def calculate_statistics(self, results):
        """Oblicz statystyki"""
        stats = {
            'total_gen_P': 0,
            'total_gen_Q': 0,
            'total_load_P': 0,
            'total_load_Q': 0,
            'overloaded_lines': [],
            'total_overload': 0,
            'max_line_loading': 0,
        }
        
        # Suma mocy generatorów + PV + StatGen
        for gen in results['generators']:
            stats['total_gen_P'] += gen['P_MW']
            stats['total_gen_Q'] += gen['Q_Mvar']
        
        for pv in results['pv']:
            stats['total_gen_P'] += pv['P_MW']
            stats['total_gen_Q'] += pv['Q_Mvar']
        
        for sg in results['statgen']:
            stats['total_gen_P'] += sg['P_MW']
            stats['total_gen_Q'] += sg['Q_Mvar']
        
        # Suma obciążeń
        for load in results['loads']:
            stats['total_load_P'] += load['P_MW']
            stats['total_load_Q'] += load['Q_Mvar']
        
        # Przeciążenia linii
        for line in results['lines']:
            loading = line['loading_pct']
            
            if loading > stats['max_line_loading']:
                stats['max_line_loading'] = loading
            
            if loading > CONFIG.OVERLOAD_THRESHOLD:
                overload = loading - CONFIG.OVERLOAD_THRESHOLD
                stats['total_overload'] += overload
                stats['overloaded_lines'].append({
                    'name': line['name'],
                    'loading': loading,
                    'overload': overload
                })
        
        return stats

# ==========================================
# KLASA GŁÓWNA - MONTE CARLO
# ==========================================
class MonteCarloSimulation:
    """Główna klasa symulacji Monte Carlo"""
    
    def __init__(self, excel_file, app):
        self.ps_data = PowerSystemData(excel_file)
        self.pf_interface = PowerFactoryInterface(app, CONFIG.PROJECT_NAME)
        
        # Wyniki wszystkich iteracji
        self.all_iterations = []
        
        # Timestamp dla plików wynikowych
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = Path(CONFIG.OUT_DIR) / f"MonteCarlo_{self.timestamp}.xlsx"
    
    def run(self):
        """Główna pętla Monte Carlo"""
        log("\n" + "="*80)
        log("🚀 START SYMULACJI MONTE CARLO")
        log("="*80)
        log(f"Liczba iteracji: {CONFIG.MAX_ITERATIONS}")
        log(f"Plik wyjściowy: {self.output_file}")
        log("="*80)
        
        accepted_count = 0
        rejected_count = 0
        lf_failed_count = 0
        
        for iteration in range(1, CONFIG.MAX_ITERATIONS + 1):
            log(f"\n{'='*80}")
            log(f"ITERACJA {iteration}/{CONFIG.MAX_ITERATIONS}")
            log(f"{'='*80}")
            
            # KROK 1: Losowanie nastaw
            if iteration > 1:
                self.ps_data.randomize_values(iteration)
            else:
                log("\n  (Iteracja 1 - używam wartości bazowych)")
            
            # KROK 2: Zapis do PowerFactory
            set_stats = self.pf_interface.set_system_state(self.ps_data)
            
            # KROK 3: Rozpływ mocy
            log("\n⚙️ Uruchamianie rozpływu mocy...")
            success, code = self.pf_interface.run_load_flow()
            
            if not success:
                log(f"  ⚠️ Rozpływ nieudany (kod {code}) - pomijam iterację")
                rejected_count += 1
                lf_failed_count += 1
                continue
            
            # KROK 4: Zbieranie wyników
            results = self.pf_interface.collect_results()
            
            # KROK 5: Obliczanie statystyk
            stats = self.pf_interface.calculate_statistics(results)
            
            # KROK 6: Sprawdzenie warunków akceptacji
            total_gen_p = stats['total_gen_P']
            
            accept = True
            if CONFIG.MIN_GEN_POWER is not None and total_gen_p < CONFIG.MIN_GEN_POWER:
                log(f"  ⚠️ Moc generacji za niska: {total_gen_p:.2f} MW < {CONFIG.MIN_GEN_POWER} MW")
                accept = False
            
            if CONFIG.MAX_GEN_POWER is not None and total_gen_p > CONFIG.MAX_GEN_POWER:
                log(f"  ⚠️ Moc generacji za wysoka: {total_gen_p:.2f} MW > {CONFIG.MAX_GEN_POWER} MW")
                accept = False
            
            if not accept:
                rejected_count += 1
                continue
            
            # KROK 7: Zapisz wyniki iteracji
            iteration_data = self._build_iteration_data(
                iteration, 
                set_stats, 
                stats, 
                results
            )
            
            self.all_iterations.append(iteration_data)
            accepted_count += 1
            
            log(f"\n✅ ITERACJA ZAAKCEPTOWANA")
            log(f"   Moc generacji: {stats['total_gen_P']:.2f} MW")
            log(f"   Przeciążone linie: {len(stats['overloaded_lines'])}")
            log(f"   Suma przeciążeń: {stats['total_overload']:.2f}%")
            log(f"   Postęp: {accepted_count} zaakceptowanych, {rejected_count} odrzuconych")
            
            # KROK 8: Zapis pośredni co 100 iteracji
            if accepted_count > 0 and accepted_count % 100 == 0:
                self.save_results()
        
        # KROK 9: Końcowy zapis wyników
        log("\n" + "="*80)
        log("📁 ZAPISYWANIE KOŃCOWYCH WYNIKÓW")
        log("="*80)
        self.save_results()
        
        log(f"\n✅ Symulacja zakończona!")
        log(f"   Zaakceptowane iteracje: {accepted_count}")
        log(f"   Odrzucone iteracje: {rejected_count}")
        log(f"     - Błędy rozpływu: {lf_failed_count}")
        log(f"     - Moc poza zakresem: {rejected_count - lf_failed_count}")
        if (accepted_count + rejected_count) > 0:
            log(f"   Współczynnik akceptacji: {100*accepted_count/(accepted_count+rejected_count):.1f}%")
    
    def _build_iteration_data(self, iteration, set_stats, stats, results):
        """Zbuduj dane dla jednej iteracji"""
        data = {
            'Iteration': iteration,
            'Total_Gen_P_MW': stats['total_gen_P'],
            'Total_Gen_Q_Mvar': stats['total_gen_Q'],
            'Total_Load_P_MW': stats['total_load_P'],
            'Total_Load_Q_Mvar': stats['total_load_Q'],
            'Overloaded_Lines_Count': len(stats['overloaded_lines']),
            'Total_Overload_pct': stats['total_overload'],
            'Max_Line_Loading_pct': stats['max_line_loading'],
        }
        
        # Dodaj nastawiane wartości (NOWA FUNKCJA!)
        # Obciążenia
        for idx, row in self.ps_data.loads_current.iterrows():
            data[f"Load_{row['name']}_P_set"] = row['P']
        
        # Generatory
        for idx, row in self.ps_data.generators_current.iterrows():
            data[f"Gen_{row['name']}_P_set"] = row['P']
        
        # PV
        for idx, row in self.ps_data.pv_current.iterrows():
            data[f"PV_{row['name']}_P_set"] = row['P']
        
        # StatGen
        for idx, row in self.ps_data.statgen_current.iterrows():
            data[f"StatGen_{row['name']}_P_set"] = row['P']
        
        # Dodaj przeciążone linie (NOWA FUNKCJA!)
        for ol in stats['overloaded_lines']:
            col_name = f"OverloadedLine_{ol['name']}_loading_pct"
            data[col_name] = ol['loading']
        
        return data
    
    def save_results(self):
        """Zapisz wyniki do Excela z wieloma arkuszami"""
        if not self.all_iterations:
            log("  ⚠️ Brak danych do zapisu")
            return
        
        try:
            df_results = pd.DataFrame(self.all_iterations)
            
            with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
                # ARKUSZ 1: Podsumowanie
                summary = {
                    'Parametr': [
                        'Liczba iteracji',
                        'Średnia moc generacji [MW]',
                        'Średnia suma przeciążeń [%]',
                        'Max obciążenie linii [%]',
                        'Średnia liczba przeciążonych linii',
                    ],
                    'Wartość': [
                        len(self.all_iterations),
                        df_results['Total_Gen_P_MW'].mean(),
                        df_results['Total_Overload_pct'].mean(),
                        df_results['Max_Line_Loading_pct'].max(),
                        df_results['Overloaded_Lines_Count'].mean(),
                    ]
                }
                pd.DataFrame(summary).to_excel(writer, sheet_name='Summary', index=False)
                
                # ARKUSZ 2: Wszystkie wyniki (pełne dane)
                df_results.to_excel(writer, sheet_name='All_Results', index=False)
                
                # ARKUSZ 3: Tylko podstawowe statystyki
                basic_cols = [col for col in df_results.columns if col.startswith(('Iteration', 'Total_', 'Overloaded_', 'Max_'))]
                df_results[basic_cols].to_excel(writer, sheet_name='Basic_Stats', index=False)
                
                # ARKUSZ 4: Nastawione wartości
                set_cols = ['Iteration'] + [col for col in df_results.columns if '_set' in col]
                if len(set_cols) > 1:
                    df_results[set_cols].to_excel(writer, sheet_name='Set_Values', index=False)
                
                # ARKUSZ 5: Przeciążenia linii
                overload_cols = ['Iteration'] + [col for col in df_results.columns if 'OverloadedLine_' in col]
                if len(overload_cols) > 1:
                    df_results[overload_cols].to_excel(writer, sheet_name='Overloaded_Lines', index=False)
            
            log(f"  ✓ Zapisano {len(self.all_iterations)} iteracji do:")
            log(f"    {self.output_file}")
            log(f"  Arkusze: Summary, All_Results, Basic_Stats, Set_Values, Overloaded_Lines")
        
        except Exception as e:
            log(f"  ❌ Błąd zapisu: {e}")
            import traceback
            log(traceback.format_exc())

# ==========================================
# MAIN
# ==========================================
def main():
    """Główna funkcja"""
    global _logger
    
    print("\n" + "="*80)
    print("🚀 MONTE CARLO POWER FLOW - IEEE 300 BUS")
    print("="*80)
    
    if powerfactory is None:
        print("❌ Moduł powerfactory niedostępny")
        return
    
    try:
        print("\n[1/3] Łączenie z PowerFactory...")
        app = powerfactory.GetApplicationExt(CONFIG.USER)
        if not app:
            print("❌ Nie można połączyć z PowerFactory")
            return
        print("  ✓ Połączono")
        
        log_file = Path(CONFIG.OUT_DIR) / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        _logger = Logger(log_file, app)
        
        print("\n[2/3] Inicjalizacja symulacji...")
        sim = MonteCarloSimulation(CONFIG.EXCEL_FILE, app)
        
        print("\n[3/3] Uruchamianie symulacji Monte Carlo...")
        sim.run()
        
        print("\n" + "="*80)
        print("✅ PROGRAM ZAKOŃCZONY POMYŚLNIE")
        print("="*80)
    
    except Exception as e:
        print(f"\n❌ BŁĄD KRYTYCZNY:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if _logger:
            _logger.close()

if __name__ == "__main__":
    main()