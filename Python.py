"""
Skrypt rozpływu mocy z wielometodowym szukaniem elementów w PowerFactory.
Wersja zabezpieczona przed blokowaniem plików.
"""

import sys
import os
import time
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend bez GUI - nie blokuje plików
import matplotlib.pyplot as plt

sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP5A\Python\3.12")
try:
    import powerfactory
except: 
    powerfactory = None

# KONFIGURACJA
EXCEL_FILE = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Wysokie napiecie\PYTHON\dane_IEEE300.xlsx"
PROJECT_NAME = "IEEE300AKT"
OUT_DIR = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Wysokie napiecie\PYTHON\Wyniki"
USER = "KE"

# Generuj unikalne nazwy plików z timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_FILE = os.path.join(OUT_DIR, f"IEEE300_{timestamp}.xlsx")
LOG_FILE = os.path.join(OUT_DIR, f"debug_log_{timestamp}.txt")

# -------------------------
# LOGGER CLASS
# -------------------------
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

# -------------------------
# ULEPSZONE FUNKCJE SZUKANIA
# -------------------------
def find_element_multi_method(app, name, pf_class):
    """
    Wielometodowe szukanie elementu w PowerFactory.
    Próbuje 3 różnych metod:
    1.GetCalcRelevantObjects z dokładną nazwą
    2.GetCalcRelevantObjects z wildcards
    3.Przeszukiwanie całego projektu
    """
    
    # METODA 1: Dokładna nazwa
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
                if getattr(obj, "loc_name", None) == name:
                    return obj
    except: 
        pass
    
    # METODA 3: Przeszukiwanie przez Study Case
    try:
        study_case = app.GetActiveStudyCase()
        if study_case:
            contents = study_case.GetContents(f"*.{pf_class}", 1)  # 1 = rekurencyjnie
            if contents:
                for obj in contents:
                    if getattr(obj, "loc_name", None) == name:
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
                        if getattr(obj, "loc_name", None) == name:
                            return obj
    except:
        pass
    
    return None

def _set_element_attr_safe(elm, attr, val):
    """Bezpieczne ustawienie atrybutu elementu"""
    try:
        elm.SetAttribute(attr, float(val))
        return True
    except:  
        return False

# -------------------------
# WCZYTYWANIE I USTAWIANIE DANYCH
# -------------------------
def load_and_set_elements_from_excel(app, file_path):
    """Wczytaj i ustaw parametry elementów z Excel - z wielometodowym szukaniem"""
    
    log("\n" + "="*80)
    log("WCZYTYWANIE DANYCH Z EXCEL")
    log("="*80)
    
    if not os.path.exists(file_path):
        log(f"❌ Plik nie istnieje: {file_path}")
        return
    
    log(f"Plik:  {file_path}")
    
    # Wczytaj dane z Excela
    try:
        loads_df = pd.read_excel(file_path, sheet_name="Loads")
        gens_df = pd.read_excel(file_path, sheet_name="Generators")
        pv_df = pd.read_excel(file_path, sheet_name="PV")
        es_df = pd.read_excel(file_path, sheet_name="StatGen")
        log("✓ Arkusze wczytane z Excela")
    except Exception as e:
        log(f"❌ Błąd wczytywania: {e}")
        return
    
    # === OBCIĄŻENIA (ElmLod) ===
    log("\n" + "-"*80)
    log("📦 USTAWIANIE OBCIĄŻEŃ (ElmLod)")
    log("-"*80)
    loads_set = 0
    loads_not_found = []
    
    for idx, row in loads_df.iterrows():
        name = str(row["name"]).strip()
        P = float(row["P"])
        Q = float(row["Q"])
        
        elm = find_element_multi_method(app, name, "ElmLod")
        if elm:
            success_p = _set_element_attr_safe(elm, "plini", P)
            success_q = _set_element_attr_safe(elm, "qlini", Q)
            if success_p and success_q: 
                loads_set += 1
                if loads_set <= 3:  # Pokaż pierwsze 3
                    log(f"  ✓ [{loads_set}] {name}:  P={P:.2f}, Q={Q:.2f}")
            else:
                log(f"  ⚠️ {name}: znaleziony, ale błąd ustawienia")
        else:
            loads_not_found.append(name)
            if len(loads_not_found) <= 3:  # Pokaż pierwsze 3 błędy
                log(f"  ❌ {name}: nie znaleziono")
    
    if loads_set > 3:
        log(f"  ...i {loads_set - 3} więcej")
    
    log(f"\n✓ Obciążenia: {loads_set}/{len(loads_df)} ({100*loads_set/max(1,len(loads_df)):.1f}%)")
    if loads_not_found:
        log(f"❌ Nie znaleziono:  {len(loads_not_found)}")
    
    # === GENERATORY SYNCHRONICZNE (ElmSym) ===
    log("\n" + "-"*80)
    log("⚡ USTAWIANIE GENERATORÓW (ElmSym)")
    log("-"*80)
    gens_set = 0
    gens_not_found = []
    
    for idx, row in gens_df.iterrows():
        name = str(row["name"]).strip()
        P = float(row["P"])
        Q = float(row["Q"])
        
        elm = find_element_multi_method(app, name, "ElmSym")
        if elm:
            success_p = _set_element_attr_safe(elm, "pgini", P)
            success_q = _set_element_attr_safe(elm, "qgini", Q)
            if success_p and success_q: 
                gens_set += 1
                if gens_set <= 3:
                    log(f"  ✓ [{gens_set}] {name}: P={P:.2f}, Q={Q:.2f}")
            else:
                log(f"  ⚠️ {name}: znaleziony, ale błąd ustawienia")
        else:
            gens_not_found.append(name)
            if len(gens_not_found) <= 3:
                log(f"  ❌ {name}: nie znaleziono")
    
    if gens_set > 3:
        log(f"  ...i {gens_set - 3} więcej")
    
    log(f"\n✓ Generatory: {gens_set}/{len(gens_df)} ({100*gens_set/max(1,len(gens_df)):.1f}%)")
    if gens_not_found:
        log(f"❌ Nie znaleziono: {len(gens_not_found)}")
    
    # === SYSTEMY PV (ElmPvsys) ===
    log("\n" + "-"*80)
    log("☀️ USTAWIANIE SYSTEMÓW PV (ElmPvsys)")
    log("-"*80)
    pv_set = 0
    pv_not_found = []
    
    for idx, row in pv_df.iterrows():
        name = str(row["name"]).strip()
        P = float(row["P"])
        Q = float(row["Q"])
        
        elm = find_element_multi_method(app, name, "ElmPvsys")
        if elm:
            success_p = _set_element_attr_safe(elm, "pgini", P)
            success_q = _set_element_attr_safe(elm, "qgini", Q)
            if success_p and success_q: 
                pv_set += 1
                if pv_set <= 3:
                    log(f"  ✓ [{pv_set}] {name}: P={P:.2f}, Q={Q:.2f}")
            else:
                log(f"  ⚠️ {name}: znaleziony, ale błąd ustawienia")
        else:
            pv_not_found.append(name)
            if len(pv_not_found) <= 3:
                log(f"  ❌ {name}: nie znaleziono")
    
    if pv_set > 3:
        log(f"  ...i {pv_set - 3} więcej")
    
    log(f"\n✓ PV: {pv_set}/{len(pv_df)} ({100*pv_set/max(1,len(pv_df)):.1f}%)")
    if pv_not_found:
        log(f"❌ Nie znaleziono: {len(pv_not_found)}")
    
    # === GENERATORY STATYCZNE (ElmGenstat) ===
    log("\n" + "-"*80)
    log("🔋 USTAWIANIE GENERATORÓW STATYCZNYCH (ElmGenstat)")
    log("-"*80)
    es_set = 0
    es_not_found = []
    
    for idx, row in es_df.iterrows():
        name = str(row["name"]).strip()
        P = float(row["P"])
        Q = float(row["Q"])
        
        elm = find_element_multi_method(app, name, "ElmGenstat")
        if elm:
            success_p = _set_element_attr_safe(elm, "pgini", P)
            success_q = _set_element_attr_safe(elm, "qgini", Q)
            if success_p and success_q:
                es_set += 1
                if es_set <= 3:
                    log(f"  ✓ [{es_set}] {name}: P={P:.2f}, Q={Q:.2f}")
            else:
                log(f"  ⚠️ {name}: znaleziony, ale błąd ustawienia")
        else:
            es_not_found.append(name)
            if len(es_not_found) <= 3:
                log(f"  ❌ {name}: nie znaleziono")
    
    if es_set > 3:
        log(f"  ...i {es_set - 3} więcej")
    
    log(f"\n✓ StatGen: {es_set}/{len(es_df)} ({100*es_set/max(1,len(es_df)):.1f}%)")
    if es_not_found:
        log(f"❌ Nie znaleziono: {len(es_not_found)}")
    
    # === PODSUMOWANIE ===
    total_set = loads_set + gens_set + pv_set + es_set
    total_all = len(loads_df) + len(gens_df) + len(pv_df) + len(es_df)
    
    log("\n" + "="*80)
    log("PODSUMOWANIE WCZYTYWANIA")
    log("="*80)
    log(f"✅ USTAWIONO: {total_set}/{total_all} ({100*total_set/max(1,total_all):.1f}%)")
    log(f"   - Obciążenia: {loads_set}/{len(loads_df)}")
    log(f"   - Generatory: {gens_set}/{len(gens_df)}")
    log(f"   - PV: {pv_set}/{len(pv_df)}")
    log(f"   - StatGen: {es_set}/{len(es_df)}")
    log("="*80)


# =========================================================================
# NOWA SEKCJA:  PARAMETRYZOWANY EKSPORT WYNIKÓW
# =========================================================================

def load_export_config(excel_file):
    """Wczytaj konfigurację eksportu z arkusza ExportConfig"""
    try:
        df_config = pd.read_excel(excel_file, sheet_name="ExportConfig")
        
        # Grupuj według Element_Type
        config = {}
        for element_type in df_config['Element_Type'].unique():
            config[element_type] = df_config[df_config['Element_Type'] == element_type].to_dict('records')
        
        log("\n" + "="*80)
        log("KONFIGURACJA EKSPORTU")
        log("="*80)
        for elem_type, columns in config.items():
            log(f"  ✓ {elem_type}:  {len(columns)} kolumn")
        
        return config
    except Exception as e:
        log(f"⚠️ Nie znaleziono arkusza ExportConfig: {e}")
        log("Używam domyślnej konfiguracji...")
        return get_default_export_config()

def get_default_export_config():
    """Domyślna konfiguracja (fallback gdy brak ExportConfig)"""
    log("\n📋 Używam wbudowanej konfiguracji eksportu")
    return {
        'ElmTerm': [
            {'Column_Name': 'Bus', 'PF_Attribute':  'loc_name', 'Format': 'str'},
            {'Column_Name':  'U [p.u.]', 'PF_Attribute': 'm: u', 'Format': '.4f'},
        ],
        'ElmLne': [
            {'Column_Name': 'Line', 'PF_Attribute': 'loc_name', 'Format': 'str'},
            {'Column_Name': 'Loading [%]', 'PF_Attribute': 'c:loading', 'Format': '.2f'},
        ],
        'ElmTr2': [
            {'Column_Name': 'Trafo', 'PF_Attribute': 'loc_name', 'Format': 'str'},
            {'Column_Name': 'Loading [%]', 'PF_Attribute': 'c: loading', 'Format': '.2f'},
            {'Column_Name': 'Tap', 'PF_Attribute':  'e:nntap', 'Format': '.0f'},
        ]
    }

def collect_results_parametrized(app, export_config):
    """Zbierz wyniki według konfiguracji z ExportConfig"""
    
    log("\n" + "="*80)
    log("ZBIERANIE WYNIKÓW")
    log("="*80)
    
    results = {}
    
    for element_type, columns_config in export_config.items():
        log(f"\n📊 {element_type}")
        element_results = []
        
        try:
            # Pobierz wszystkie elementy danego typu
            elements = app.GetCalcRelevantObjects(f"*.{element_type}")
            if not elements:
                log(f"  ⚠️ Brak elementów")
                continue
            
            log(f"  Znaleziono:  {len(elements)} elementów")
            successful = 0
            errors = 0
            
            for elem in elements:
                row_data = {}
                has_error = False
                
                for col_config in columns_config:
                    col_name = col_config['Column_Name']
                    pf_attr = col_config['PF_Attribute']
                    
                    try:
                        # Pobierz wartość atrybutu
                        if ': ' in pf_attr: 
                            # Atrybuty wynikowe (m:, c:, e:)
                            value = elem.GetAttribute(pf_attr)
                        else:
                            # Atrybuty podstawowe (loc_name, itp.)
                            value = getattr(elem, pf_attr, None)
                        
                        # Zapisz wartość
                        if value is None:
                            row_data[col_name] = None
                        else:
                            row_data[col_name] = value
                            
                    except Exception as e:
                        row_data[col_name] = None
                        has_error = True
                        if errors < 3:  # Pokaż tylko pierwsze 3 błędy
                            log(f"  ⚠️ {elem.loc_name}: brak '{pf_attr}'")
                        errors += 1
                
                element_results.append(row_data)
                if not has_error:
                    successful += 1
            
            if errors > 3:
                log(f"  ⚠️ ...i {errors - 3} więcej błędów")
            
            log(f"  ✓ Zebrano: {successful}/{len(elements)} ({100*successful/max(1,len(elements)):.1f}%)")
            results[element_type] = element_results
            
        except Exception as e:
            log(f"  ❌ Błąd: {e}")
            results[element_type] = []
    
    return results

def save_results_to_excel(results, output_file):
    """Zapisz wyniki do Excela"""
    
    log("\n" + "="*80)
    log("ZAPISYWANIE WYNIKÓW")
    log("="*80)
    log(f"Plik: {output_file}")
    
    try:
        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            
            for element_type, data in results.items():
                if not data:
                    log(f"  ⚠️ {element_type}: brak danych (pomijam)")
                    continue
                
                df = pd.DataFrame(data)
                
                # Nazwa arkusza (usuń prefix "Elm")
                sheet_name = element_type.replace("Elm", "")
                if len(sheet_name) > 31:  # Excel limit
                    sheet_name = sheet_name[:31]
                
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Auto-szerokość kolumn
                try:
                    worksheet = writer.sheets[sheet_name]
                    for idx, col in enumerate(df.columns):
                        max_length = max(
                            df[col].astype(str).map(len).max(),
                            len(col)
                        ) + 2
                        col_letter = chr(65 + idx) if idx < 26 else f"A{chr(65 + idx - 26)}"
                        worksheet.column_dimensions[col_letter].width = min(max_length, 50)
                except:
                    pass
                
                log(f"  ✓ {sheet_name}: {len(df)} wierszy, {len(df.columns)} kolumn")
        
        log(f"✅ Wyniki zapisane")
        return True
        
    except PermissionError:
        log("❌ BŁĄD: Plik jest otwarty w Excelu!  Zamknij go i spróbuj ponownie.")
        return False
    except Exception as e:
        log(f"❌ Błąd zapisu: {e}")
        return False

def extract_node_num(bus_name):
    """Wyciągnij numer węzła z nazwy"""
    try:
        if isinstance(bus_name, str) and bus_name.startswith("Bus"):
            return int(bus_name.replace("Bus", '').strip())
        return int(bus_name.strip())
    except:  
        return 9999

def generate_voltage_plot(results, out_dir, timestamp):
    """Generuj wykres napięć z wyników"""
    
    log("\n📈 Generowanie wykresu napięć...")
    
    try:
        # Znajdź dane węzłów
        buses_data = None
        for element_type, data in results.items():
            if 'Term' in element_type and data:
                buses_data = data
                break
        
        if not buses_data:
            log("  ⚠️ Brak danych węzłów - pomijam wykres")
            return
        
        # Znajdź kolumnę z napięciem i nazwą
        voltage_col = None
        name_col = None
        
        for col in buses_data[0].keys():
            if 'p.u.' in col.lower():
                voltage_col = col
            if any(x in col.lower() for x in ['bus', 'węzeł', 'nazwa', 'node']):
                name_col = col
        
        if not voltage_col or not name_col: 
            log(f"  ⚠️ Brak kolumn napięcia/nazwy - pomijam wykres")
            return
        
        # Przygotuj dane
        u_pairs = []
        for b in buses_data:
            try:
                voltage = b.get(voltage_col)
                name = b.get(name_col)
                if voltage is not None and voltage > 0:
                    node_num = extract_node_num(name)
                    u_pairs.append((node_num, float(voltage)))
            except: 
                continue
        
        if not u_pairs:
            log("  ⚠️ Brak danych do wykresu")
            return
        
        u_pairs.sort()
        node_numbers, voltages = zip(*u_pairs)
        
        # Rysuj wykres
        plt.figure(figsize=(12, 6))
        plt.plot(node_numbers, voltages, marker='o', linestyle='-', linewidth=2, markersize=4)
        plt.axhline(y=1.0, color='green', linestyle='--', label='Nominalne (1.0 p.u.)')
        plt.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='Min (0.9 p.u.)')
        plt.axhline(y=1.1, color='red', linestyle='--', alpha=0.5, label='Max (1.1 p.u.)')
        plt.xlabel("Numer węzła")
        plt.ylabel("Napięcie [p.u.]")
        plt.title("Profil napięć w sieci")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(out_dir, f"Voltage_Profile_{timestamp}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        
        log(f"  ✓ Wykres zapisany: {plot_path}")
        
    except Exception as e: 
        log(f"  ⚠️ Błąd wykresu: {e}")

# -------------------------
# MAIN
# -------------------------
def main():
    global _logger
    
    try:
        if powerfactory is None:
            print("❌ Moduł powerfactory niedostępny")
            return
        
        # Połącz z PowerFactory
        app = powerfactory.GetApplicationExt(USER)
        app.ActivateProject(PROJECT_NAME)
        ldf = app.GetFromStudyCase("ComLdf")
        
        # Inicjalizacja loggera
        _logger = Logger(LOG_FILE, app)
        
        log("="*80)
        log("SKRYPT ROZPŁYWU MOCY - WERSJA MULTI-SEARCH")
        log("="*80)
        log(f"Projekt: {PROJECT_NAME}")
        log(f"Użytkownik: {USER}")
        log(f"Plik Excel: {EXCEL_FILE}")
        log(f"Plik wynikowy: {OUT_FILE}")
        log(f"Plik logu: {LOG_FILE}")
        log("="*80)
        
        # Wczytaj i ustaw dane
        load_and_set_elements_from_excel(app, EXCEL_FILE)
        
        export_config = load_export_config(EXCEL_FILE)
        
        # Wykonaj rozpływ mocy
        log("\n" + "="*80)
        log("WYKONYWANIE ROZPŁYWU MOCY")
        log("="*80)
        time_start = time.time()
        
        try:
            code = ldf.Execute()
            if code == 0:
                log("✓ Rozpływ mocy zakończony sukcesem")
            else:
                log(f"⚠️ Rozpływ mocy zakończony z kodem błędu: {code}")
        except Exception as e:
            log(f"❌ Błąd podczas wykonywania rozpływu:  {e}")
        
        time_end = time.time()
        log(f"Czas obliczeń: {time_end - time_start:.3f} s")
        log("="*80)
        
        # Zbierz wyniki
        results = collect_results_parametrized(app, export_config)
        
        # Zapisz do Excela
        save_results_to_excel(results, OUT_FILE)
        
        # Wykres napięć
        generate_voltage_plot(results, OUT_DIR, timestamp)
        
        # PODSUMOWANIE
        log("\n" + "="*80)
        log("PODSUMOWANIE")
        log("="*80)
        
        # Statystyki z wyników
        for element_type, data in results.items():
            if data:
                df = pd.DataFrame(data)
                log(f"\n{element_type}:")
                
                # Dla węzłów - statystyki napięć
                if 'Term' in element_type:
                    u_col = next((col for col in df.columns if 'p.u.' in col.lower()), None)
                    if u_col: 
                        voltages = df[u_col].dropna()
                        if len(voltages) > 0:
                            log(f"  Min napięcie: {voltages.min():.4f} p.u.")
                            log(f"  Max napięcie: {voltages.max():.4f} p.u.")
                            log(f"  Śr.napięcie: {voltages.mean():.4f} p.u.")
                
                # Dla linii - statystyki obciążeń
                if 'Lne' in element_type or 'Tr2' in element_type:
                    load_col = next((col for col in df.columns if 'obciążenie' in col.lower() or 'loading' in col.lower()), None)
                    if load_col:
                        loadings = df[load_col].dropna()
                        if len(loadings) > 0:
                            log(f"  Max obciążenie: {loadings.max():.2f}%")
                            log(f"  Śr.obciążenie: {loadings.mean():.2f}%")
                            overloaded = len(loadings[loadings > 100])
                            if overloaded > 0:
                                log(f"  ⚠️ Przeciążonych: {overloaded}")
        
        log("\n" + "="*80)
        log("✅ ANALIZA ZAKOŃCZONA")
        log(f"\n📄 Pliki zapisane:")
        log(f"   - Wyniki:  {OUT_FILE}")
        log(f"   - Log: {LOG_FILE}")
        log("="*80)
        
    except Exception as e:
        log(f"\n❌ KRYTYCZNY BŁĄD: {e}")
        import traceback
        log("\nTraceback:")
        log(traceback.format_exc())
    
    finally:
        if _logger:
            _logger.close()
            _logger = None

if __name__ == "__main__":
    main()






    
# =========================================================================
# SEKCJA OPTYMALIZACJI
# =========================================================================

def run_optimization(app, ldf, excel_file, out_dir, scenario_name="N1", algorithm='PSO'):
    """
    Uruchom optymalizację dla obecnego stanu sieci (N-1)
    
    Parametry:
    - app: PowerFactory application
    - ldf: Load Flow object  
    - excel_file: ścieżka do dane_IEEE300.xlsx
    - out_dir: katalog wyników
    - scenario_name: nazwa scenariusza (np."N1_Line257")
    - algorithm: 'PSO', 'CEO', 'GA' (na razie tylko PSO)
    """
    from optimizer import N1Optimizer
    
    optimizer = N1Optimizer(app, ldf, excel_file, out_dir, scenario_name)
    result = optimizer.run_pso(algorithm)
    
    return result

# Dodaj do main() opcję uruchomienia optymalizacji
def main_with_optimization():
    """Main z opcją optymalizacji"""
    global _logger
    
    try:
        if powerfactory is None:
            print("❌ Moduł powerfactory niedostępny")
            return
        
        # Połącz z PowerFactory
        app = powerfactory.GetApplicationExt(USER)
        app.ActivateProject(PROJECT_NAME)
        ldf = app.GetFromStudyCase("ComLdf")
        
        # Inicjalizacja loggera
        _logger = Logger(LOG_FILE, app)
        
        log("="*80)
        log("SKRYPT ROZPŁYWU MOCY + OPTYMALIZACJA N-1")
        log("="*80)
        log(f"Projekt: {PROJECT_NAME}")
        log(f"Plik Excel: {EXCEL_FILE}")
        log("="*80)
        
        # KROK 1: Wczytaj i ustaw dane
        load_and_set_elements_from_excel(app, EXCEL_FILE)
        export_config = load_export_config(EXCEL_FILE)
        
        # KROK 2: Rozpływ PRZED optymalizacją
        log("\n" + "="*80)
        log("ROZPŁYW MOCY - STAN BAZOWY")
        log("="*80)
        
        time_start = time.time()
        code = ldf.Execute()
        time_end = time.time()
        
        if code == 0:
            log(f"✓ Rozpływ OK ({time_end - time_start:.3f}s)")
        else:
            log(f"⚠️ Rozpływ z błędem: kod {code}")
        
        # Zbierz i zapisz wyniki bazowe
        results_base = collect_results_parametrized(app, export_config)
        base_file = os.path.join(OUT_DIR, f"BASE_{timestamp}.xlsx")
        save_results_to_excel(results_base, base_file)
        
        # KROK 3: OPTYMALIZACJA
        log("\n" + "="*80)
        log("❓ Uruchomić optymalizację?  (Stan N-1 ustawiony ręcznie)")
        log("="*80)
        
        # Tutaj możesz dodać input() lub parametr z linii poleceń
        run_opt = input("Uruchomić optymalizację? [t/n]: ").lower() == 't'
        
        if run_opt:
            scenario_name = input("Nazwa scenariusza (np.N1_Line257): ") or "N1_Manual"
            
            result = run_optimization(
                app, 
                ldf, 
                EXCEL_FILE, 
                OUT_DIR, 
                scenario_name=scenario_name,
                algorithm='PSO'
            )
            
            log("\n✅ OPTYMALIZACJA ZAKOŃCZONA")
        else:
            log("\n⏭️ Optymalizacja pominięta")
        
        log("\n" + "="*80)
        log("✅ ANALIZA ZAKOŃCZONA")
        log("="*80)
        
    except Exception as e:
        log(f"\n❌ KRYTYCZNY BŁĄD: {e}")
        import traceback
        log(traceback.format_exc())
    
    finally:
        if _logger:
            _logger.close()
            _logger = None

# Możliwość wyboru trybu
if __name__ == "__main__": 
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--optimize':
        main_with_optimization()
    else:
        main()  # Standardowy tryb