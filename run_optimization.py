"""
Uruchomienie optymalizacji N-1 dla PowerFactory
URUCHAMIAĆ Z POWERFACTORY jako Python Script! 
"""

import sys
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Usuń stare moduły z cache
if 'objective_function' in sys.modules:
    del sys.modules['objective_function']
if 'PSO' in sys.modules:
    del sys.modules['PSO']

# Teraz importuj na nowo
try:
    from PSO import PSO
    from objective_function import PowerFactoryObjective
    print("✅ Moduły zaimportowane")
except ImportError as e:
    print(f"❌ Błąd importu:{e}")
    raise

# ==========================================
# KONFIGURACJA (edytuj tutaj!)
# ==========================================
EXCEL_FILE = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Wysokie napiecie\PYTHON\dane_IEEE300.xlsx"
PROJECT_NAME = "IEEE300AKT"
OUT_DIR = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Wysokie napiecie\PYTHON\Wyniki"
USER = "KE"

# Algorytm do użycia
ALGORITHM = 'PSO'

# Dodaj ścieżkę do modułów lokalnych
SCRIPT_DIR = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Wysokie napiecie\PYTHON"
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ==========================================
# FUNKCJE POMOCNICZE (skopiowane z Python.py)
# ==========================================

class Logger:
    def __init__(self, filename, pf_app=None):
        self.log_file = open(filename, 'w', encoding='utf-8')
        self.pf_app = pf_app
        header = "="*80 + "\n"
        header += f"LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += "="*80 + "\n\n"
        self.log_file.write(header)
        self.log_file.flush()
        if self.pf_app:
            try:
                self.pf_app.PrintPlain(header)
            except:
                pass
    
    def write(self, message):
        self.log_file.write(str(message))
        self.log_file.flush()
        if self.pf_app:
            try:
                self.pf_app.PrintPlain(str(message).rstrip('\n'))
            except:
                pass
    
    def close(self):
        self.log_file.close()

_logger = None

def log(message):
    global _logger
    if _logger:
        _logger.write(str(message) + "\n")
    else:
        print(message)

def find_element_multi_method(app, name, pf_class):
    """Wielometodowe szukanie elementu w PowerFactory"""
    
    # METODA 1:Dokładna nazwa
    try:
        objs = app.GetCalcRelevantObjects(f"{name}.{pf_class}")
        if objs and len(objs) > 0:
            return objs[0]
    except:
        pass
    
    # METODA 2:Wildcard search
    try:
        all_objs = app.GetCalcRelevantObjects(f"*.{pf_class}")
        if all_objs:
            for obj in all_objs:
                if getattr(obj, "loc_name", None) == name:
                    return obj
    except:
        pass
    
    # METODA 3:Przeszukiwanie przez Study Case
    try:
        study_case = app.GetActiveStudyCase()
        if study_case:
            contents = study_case.GetContents(f"*.{pf_class}", 1)
            if contents:
                for obj in contents:
                    if getattr(obj, "loc_name", None) == name:
                        return obj
    except:
        pass
    
    # METODA 4:Przeszukiwanie całego projektu
    try:
        prj = app.GetActiveProject()
        if prj:
            all_folders = prj.GetContents('*.IntFolder', 1)
            for folder in all_folders:
                contents = folder.GetContents(f'*.{pf_class}', 1)
                if contents:
                    for obj in contents:
                        if getattr(obj, "loc_name", None) == name:
                            return obj
    except:
        pass
    
    return None

def load_export_config(excel_file):
    """Wczytaj konfigurację eksportu z arkusza ExportConfig"""
    try:
        df_config = pd.read_excel(excel_file, sheet_name="ExportConfig")
        
        config = {}
        for element_type in df_config['Element_Type'].unique():
            config[element_type] = df_config[df_config['Element_Type'] == element_type].to_dict('records')
        
        log("\n✓ Konfiguracja eksportu wczytana")
        return config
    except Exception as e:
        log(f"⚠️ Nie znaleziono arkusza ExportConfig:{e}")
        # Domyślna konfiguracja
        return {
            'ElmTerm':[
                {'Column_Name':'Bus', 'PF_Attribute':'loc_name', 'Format':'str'},
                {'Column_Name':'U [p.u.]', 'PF_Attribute':'m:u', 'Format':'.4f'},
            ],
            'ElmLne':[
                {'Column_Name':'Line', 'PF_Attribute':'loc_name', 'Format':'str'},
                {'Column_Name':'Loading [%]', 'PF_Attribute':'c:loading', 'Format':'.2f'},
            ],
        }

def collect_results_parametrized(app, export_config):
    """Zbierz wyniki według konfiguracji"""
    
    results = {}
    
    for element_type, columns_config in export_config.items():
        element_results = []
        
        try:
            elements = app.GetCalcRelevantObjects(f"*.{element_type}")
            if not elements:
                continue
            
            for elem in elements:
                row_data = {}
                
                for col_config in columns_config:
                    col_name = col_config['Column_Name']
                    pf_attr = col_config['PF_Attribute']
                    
                    try:
                        if ':' in pf_attr:
                            value = elem.GetAttribute(pf_attr)
                        else:
                            value = getattr(elem, pf_attr, None)
                        
                        row_data[col_name] = value if value is not None else None
                            
                    except:
                        row_data[col_name] = None
                
                element_results.append(row_data)
            
            results[element_type] = element_results
            
        except Exception as e:
            log(f"⚠️ Błąd zbierania {element_type}:{e}")
            results[element_type] = []
    
    return results

def save_results_to_excel(results, output_file):
    """Zapisz wyniki do Excela"""
    
    try:
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            
            for element_type, data in results.items():
                if not data:
                    continue
                
                df = pd.DataFrame(data)
                sheet_name = element_type.replace("Elm", "")[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        log(f"✅ Wyniki zapisane:{output_file}")
        return True
        
    except Exception as e:
        log(f"❌ Błąd zapisu:{e}")
        return False

# ==========================================
# IMPORT MODUŁÓW OPTYMALIZACJI
# ==========================================

try:
    from PSO import PSO
    from objective_function import PowerFactoryObjective
except ImportError as e:
    log(f"❌ Błąd importu modułów optymalizacji:{e}")
    log("Sprawdź czy pliki PSO.py i objective_function.py są w katalogu:" + SCRIPT_DIR)
    raise

# ==========================================
# KLASA OPTYMALIZATORA (uproszczona wersja)
# ==========================================

# Parametry PSO
PSO_PARAMS = {
    'n_particles':100,
    'max_iter':500,
    'w':0.7,
    'c1':1.5,
    'c2':1.5,
    'autosave_every':50,
}

class N1Optimizer:
    """Optymalizator dla stanu N-1"""
    
    def __init__(self, app, ldf, excel_file, out_dir, scenario_name="N1"):
        self.app = app
        self.ldf = ldf
        self.excel_file = excel_file
        self.out_dir = out_dir
        self.scenario_name = scenario_name
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_prefix = os.path.join(out_dir, f"{scenario_name}_{self.timestamp}")
        
        # Wczytaj konfigurację
        self.opt_variables = self._load_optimization_config()
        self.lb, self.ub = self._extract_bounds()
        self.dim = len(self.lb)
        
        self.results_before = None
        self.results_after = None
    
    def _load_optimization_config(self):
        """Wczytaj arkusz Optymalizacja"""
        try:
            df = pd.read_excel(self.excel_file, sheet_name="Optymalizacja")
            log(f"✓ Wczytano konfigurację:{len(df)} elementów")
            return df.to_dict('records')
        except Exception as e:
            log(f"❌ Błąd wczytywania Optymalizacja:{e}")
            return []
    
    def _extract_bounds(self):
        """Wyciągnij granice zmiennych"""
        lb = []
        ub = []
        
        for var in self.opt_variables:
            if 'Pmin' in var and 'Pmax' in var:
                lb.append(float(var['Pmin']))
                ub.append(float(var['Pmax']))
            
            if 'Qmin' in var and 'Qmax' in var:
                lb.append(float(var['Qmin']))
                ub.append(float(var['Qmax']))
        
        return np.array(lb), np.array(ub)
    
    def save_current_state(self, phase="before"):
        """Zapisz stan sieci"""
        export_config = load_export_config(self.excel_file)
        results = collect_results_parametrized(self.app, export_config)
        
        if phase == "before":
            self.results_before = results
        else:
            self.results_after = results
        
        return results
    
    def run_pso(self):
        """Uruchom optymalizację PSO"""
        
        log("\n" + "="*80)
        log("OPTYMALIZACJA PSO")
        log("="*80)
        log(f"Scenariusz:{self.scenario_name}")
        log(f"Zmiennych:{self.dim}")
        log(f"Cząstek:{PSO_PARAMS['n_particles']}")
        log(f"Iteracji:{PSO_PARAMS['max_iter']}")
        log("="*80)
        
        # Stan przed
        log("\n📸 Zapisywanie stanu PRZED...")
        self.save_current_state("before")
        
        # Funkcja celu
        log_file = f"{self.out_prefix}_iterations.csv"
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("Iteration,Timestamp,F_Objective,Penalty,F_Total,")
            var_names = []
            for var in self.opt_variables:
                if 'Attribute_P' in var:
                    var_names.append(f"{var['Element_Name']}_P")
                if 'Attribute_Q' in var:
                    var_names.append(f"{var['Element_Name']}_Q")
            f.write(','.join(var_names) + '\n')
        
        objective = PowerFactoryObjective(
            self.app, 
            self.ldf, 
            self.opt_variables,
            log_file=log_file
        )
        
        # PSO
        checkpoint_path = f"{self.out_prefix}_checkpoint.npz"
        
        pso = PSO(
            func=objective,
            n_particles=PSO_PARAMS['n_particles'],
            dim=self.dim,
            lb=self.lb,
            ub=self.ub,
            max_iter=PSO_PARAMS['max_iter'],
            w=PSO_PARAMS['w'],
            c1=PSO_PARAMS['c1'],
            c2=PSO_PARAMS['c2'],
            autosave_every_iters=PSO_PARAMS['autosave_every'],
            autosave_path=checkpoint_path,
            early_stop_threshold=0.0,      # ✅ NOWE - zatrzymaj gdy f <= 0
            early_stop_patience=1000         # ✅ NOWE - lub brak poprawy przez 50 iter
        )
        
        time_start = time.time()
        result = pso.optimize()
        time_end = time.time()
        
        # ✅ Sprawdź czy zatrzymano wcześniej
        if result.get('early_stopped'):
            log(f"\n⏸️  Zatrzymano wcześniej:{result.get('reason')}")
            log(f"   Iteracja:{result.get('stopped_at_iter')}/{PSO_PARAMS['max_iter']}")
        
        log(f"\n✅ Zakończono:{time_end - time_start:.2f}s")
        log(f"F_best:{result['gbest_val']:.6f}")
        
        # Ustaw najlepsze rozwiązanie
        objective.set_variables(result['gbest'])
        self.ldf.Execute()
        
        # Stan po
        log("\n📸 Zapisywanie stanu PO...")
        self.save_current_state("after")
        
        # Zapisz wyniki
        self._save_results(result, objective, time_end - time_start)
        
        return result
    
    def _save_results(self, pso_result, objective, elapsed_time):
        """Zapisz wyniki do Excela"""
        
        output_file = f"{self.out_prefix}.xlsx"
        log(f"\n💾 Zapisywanie:{output_file}")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Podsumowanie
            summary = {
                'Scenariusz':[self.scenario_name],
                'Timestamp':[self.timestamp],
                'Algorytm':['PSO'],
                'Zmiennych':[self.dim],
                'Cząstek':[PSO_PARAMS['n_particles']],
                'Iteracji':[PSO_PARAMS['max_iter']],
                'Czas [s]':[elapsed_time],
                'Ewaluacji':[objective.eval_count],
                'F_best':[pso_result['gbest_val']],
            }
            pd.DataFrame(summary).to_excel(writer, sheet_name='Podsumowanie', index=False)
            
            # Najlepsze rozwiązanie
            best_solution = []
            idx = 0
            for var in self.opt_variables:
                row = {
                    'Element':var['Element_Name'],
                    'Type':var['Element_Type'],
                }
                if 'Attribute_P' in var:
                    row['P'] = pso_result['gbest'][idx]
                    idx += 1
                if 'Attribute_Q' in var:
                    row['Q'] = pso_result['gbest'][idx]
                    idx += 1
                best_solution.append(row)
            
            pd.DataFrame(best_solution).to_excel(writer, sheet_name='BestSolution', index=False)
            
            # Historia
            history = pd.DataFrame({
                'Iteration':range(len(pso_result['best_per_iter'])),
                'Best_Value':pso_result['best_per_iter']
            })
            history.to_excel(writer, sheet_name='History', index=False)
            
            # Wyniki PRZED
            if self.results_before:
                for element_type, data in self.results_before.items():
                    if data:
                        df = pd.DataFrame(data)
                        sheet_name = f"BEFORE_{element_type.replace('Elm', '')}"[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Wyniki PO
            if self.results_after:
                for element_type, data in self.results_after.items():
                    if data:
                        df = pd.DataFrame(data)
                        sheet_name = f"AFTER_{element_type.replace('Elm', '')}"[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        log(f"✅ Zapisano:{output_file}")

# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================

def check_n1_state(app):
    """Sprawdź wyłączone elementy (outserv=1 LUB loading=0)"""
    disabled = []
    
    try:
        # Sprawdź linie
        lines = app.GetCalcRelevantObjects("*.ElmLne")
        for line in lines:
            is_disabled = False
            reason = ""
            
            # Metoda 1:outserv
            if hasattr(line, 'outserv') and line.outserv == 1:
                is_disabled = True
                reason = "outserv=1"
            
            # Metoda 2:loading = 0 (może być wyłączona)
            try:
                loading = line.GetAttribute("c:loading")
                if loading is not None and loading == 0.0:
                    # Sprawdź czy to nie jest przypadkiem linia bez obciążenia
                    # (np.nowo dodana, ale aktywna)
                    if not is_disabled:
                        is_disabled = True
                        reason = "loading=0"
            except:
                pass
            
            # Metoda 3:status operacyjny
            try:
                if hasattr(line, 'on_off') and line.on_off == 0:
                    is_disabled = True
                    reason = "on_off=0"
            except:
                pass
            
            if is_disabled:
                disabled.append(f"Linia:{line.loc_name} ({reason})")
        
        # Sprawdź transformatory
        trafos = app.GetCalcRelevantObjects("*.ElmTr2")
        for trafo in trafos:
            is_disabled = False
            reason = ""
            
            if hasattr(trafo, 'outserv') and trafo.outserv == 1:
                is_disabled = True
                reason = "outserv=1"
            
            try:
                loading = trafo.GetAttribute("c:loading")
                if loading is not None and loading == 0.0:
                    if not is_disabled:
                        is_disabled = True
                        reason = "loading=0"
            except:
                pass
            
            try:
                if hasattr(trafo, 'on_off') and trafo.on_off == 0:
                    is_disabled = True
                    reason = "on_off=0"
            except:
                pass
            
            if is_disabled:
                disabled.append(f"Trafo:{trafo.loc_name} ({reason})")
    
    except Exception as e:
        log(f"⚠️ Błąd sprawdzania N-1:{e}")
    
    return disabled

def check_line_overloads(results):
    """Sprawdź przeciążenia"""
    overloads = []
    
    if 'ElmLne' in results and results['ElmLne']:
        df = pd.DataFrame(results['ElmLne'])
        
        load_col = None
        for col in df.columns:
            if 'loading' in col.lower():
                load_col = col
                break
        
        if load_col:
            overloaded = df[df[load_col] > 100]
            for idx, row in overloaded.iterrows():
                name = row.get('Line') or row.get('loc_name') or f"#{idx}"
                loading = row[load_col]
                log(f"   ⚠️ {name}:{loading:.2f}%")
                overloads.append({'name':name, 'loading':loading})
    
    return overloads

# ==========================================
# FUNKCJA GŁÓWNA
# ==========================================

def main():
    """Główna funkcja"""
    global _logger
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(OUT_DIR, f"optimization_log_{timestamp}.txt")
    
    try:
        # Pobierz PowerFactory (jest już dostępny!)
        import powerfactory
        app = powerfactory.GetApplicationExt()
        
        if app is None:
            print("❌ Nie można połączyć z PowerFactory")
            return
        
        # Aktywuj projekt
        prj = app.GetActiveProject()
        if prj is None or prj.loc_name != PROJECT_NAME:
            app.ActivateProject(PROJECT_NAME)
        
        ldf = app.GetFromStudyCase("ComLdf")
        
        # Logger
        _logger = Logger(log_file, app)
        
        log("="*80)
        log("OPTYMALIZACJA N-1 - POWERFACTORY")
        log("="*80)
        log(f"Projekt:{PROJECT_NAME}")
        log(f"Excel:{EXCEL_FILE}")
        log("="*80)
        
        # Nazwa scenariusza
        scenario_name = app.GetFromStudyCase("SetSelect")
        if scenario_name:
            scenario_name = scenario_name.loc_name
        else:
            scenario_name = f"N1_{timestamp}"
        
        log(f"\nScenariusz:{scenario_name}")
        
        # Sprawdź N-1
        log("\n" + "="*80)
        log("WERYFIKACJA STANU N-1")
        log("="*80)
        
        disabled = check_n1_state(app)
        if disabled:
            log(f"✓ Wyłączone elementy ({len(disabled)}):")
            for elem in disabled:
                log(f"   - {elem}")
        else:
            log("⚠️ Nie wykryto wyłączonych elementów")
        
        # Rozpływ bazowy
        log("\n" + "="*80)
        log("ROZPŁYW BAZOWY")
        log("="*80)
        
        code = ldf.Execute()
        if code == 0:
            log("✓ Rozpływ OK")
        else:
            log(f"⚠️ Rozpływ błąd:{code}")
        
        export_config = load_export_config(EXCEL_FILE)
        results_base = collect_results_parametrized(app, export_config)
        
        # Sprawdź przeciążenia
        overloads = check_line_overloads(results_base)
        
        if overloads:
            log(f"\n⚠️ Znaleziono {len(overloads)} przeciążonych linii")
        else:
            log("\n✅ Brak przeciążeń")
        
        # OPTYMALIZACJA
        log("\n" + "="*80)
        log("URUCHAMIANIE OPTYMALIZACJI")
        log("="*80)
        
        optimizer = N1Optimizer(app, ldf, EXCEL_FILE, OUT_DIR, scenario_name)
        
        log(f"Zmiennych decyzyjnych:{optimizer.dim}")
        
        result = optimizer.run_pso()
        
        log("\n" + "="*80)
        log("✅ ZAKOŃCZONO")
        log("="*80)
        log(f"Najlepsza wartość:{result['gbest_val']:.6f}")
        log(f"Wyniki w:{OUT_DIR}")
        
    except Exception as e:
        log(f"\n❌ BŁĄD:{e}")
        import traceback
        log(traceback.format_exc())
    
    finally:
        if _logger:
            _logger.close()
            _logger = None

# ==========================================
# URUCHOMIENIE
# ==========================================

if __name__ == "__main__":
    main()