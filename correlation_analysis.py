"""
Analiza korelacji Pearsona dla wyników Monte Carlo
Z pełną macierzą korelacji i klasyfikacją siły
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import warnings
import sys
warnings.filterwarnings('ignore')

# ==========================================
# KONFIGURACJA
# ==========================================
class Config:
    # Katalog z wynikami
    RESULTS_DIR = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Wysokie napiecie\PYTHON\Wyniki"
    
    # Plik wynikowy (najnowszy lub konkretny)
    INPUT_FILE = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Wysokie napiecie\PYTHON\Podsumowanie\Wył 70.71\Korelacja\MonteCarlo.xlsx"  # None = weź najnowszy plik MonteCarlo_*.xlsx
    
    # Próg korelacji (tylko powyżej tego progu będzie wyświetlane w raportach)
    CORRELATION_THRESHOLD = 0.1  # |r| > 0.1
    
    # Poziom istotności statystycznej
    P_VALUE_THRESHOLD = 0.05  # p < 0.05
    
    # Klasyfikacja siły korelacji
    CORRELATION_RANGES = [
        (0.9, 1.0, "Pełna"),
        (0.7, 0.9, "Bardzo_Silna"),
        (0.5, 0.7, "Silna"),
        (0.3, 0.5, "Umiarkowana"),
        (0.1, 0.3, "Słaba"),
        (0.0, 0.1, "Nikła"),
    ]

CONFIG = Config()

# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================
def find_latest_results_file(results_dir):
    """Znajdź najnowszy plik wyników Monte Carlo"""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Katalog nie istnieje: {results_dir}")
        return None
    
    monte_carlo_files = [f for f in results_path.glob("MonteCarlo_*.xlsx") if f.is_file()]
    
    if not monte_carlo_files:
        print(f"❌ Nie znaleziono plików MonteCarlo_*.xlsx w katalogu:")
        print(f"   {results_dir}")
        return None
    
    latest_file = max(monte_carlo_files, key=lambda p: p.stat().st_mtime)
    return latest_file

def load_results(file_path):
    """Wczytaj wyniki z Excela"""
    print("\n" + "="*80)
    print("📂 WCZYTYWANIE WYNIKÓW MONTE CARLO")
    print("="*80)
    print(f"Plik: {file_path}")
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Plik nie istnieje: {file_path}")
    
    if Path(file_path).is_dir():
        raise IsADirectoryError(f"To jest folder, nie plik: {file_path}")
    
    try:
        df = pd.read_excel(file_path, sheet_name='All_Results')
        print(f"✓ Wczytano: {len(df)} iteracji")
        print(f"✓ Kolumny: {len(df.columns)}")
        return df
    
    except ValueError as e:
        print(f"\n❌ Błąd: Arkusz 'All_Results' nie istnieje w pliku")
        xl_file = pd.ExcelFile(file_path)
        print(f"\nDostępne arkusze:")
        for sheet_name in xl_file.sheet_names:
            print(f"  - {sheet_name}")
        raise

def identify_columns(df):
    """Zidentyfikuj kolumny z nastawianymi wartościami i przeciążeniami"""
    print("\n" + "="*80)
    print("🔍 IDENTYFIKACJA KOLUMN")
    print("="*80)
    
    set_value_cols = [col for col in df.columns if '_set' in col]
    print(f"\n✓ Nastawiane wartości: {len(set_value_cols)} kolumn")
    
    if set_value_cols:
        print("  Przykłady:")
        for col in set_value_cols[:5]:
            print(f"    - {col}")
        if len(set_value_cols) > 5:
            print(f"    ... i {len(set_value_cols) - 5} więcej")
    
    overload_cols = [col for col in df.columns if 'OverloadedLine_' in col and '_loading_pct' in col]
    print(f"\n✓ Przeciążone linie: {len(overload_cols)} kolumn")
    
    if overload_cols:
        print("  Przykłady:")
        for col in overload_cols[:5]:
            line_name = col.replace('OverloadedLine_', '').replace('_loading_pct', '')
            print(f"    - {line_name}")
        if len(overload_cols) > 5:
            print(f"    ... i {len(overload_cols) - 5} więcej")
    
    return set_value_cols, overload_cols

def calculate_correlations_full(df, set_value_cols, overload_cols):
    """
    Oblicz WSZYSTKIE korelacje (bez filtrowania)
    Zwraca również pełną macierz i p-values
    """
    print("\n" + "="*80)
    print("📊 OBLICZANIE PEŁNEJ MACIERZY KORELACJI")
    print("="*80)
    
    correlations = {}  # Dla raportów (tylko istotne >= 0.3)
    weak_correlations = {}  # ✅ NOWE - słabe korelacje (0.1-0.3)
    full_matrix_r = {}  # Pełna macierz współczynników r
    full_matrix_p = {}  # Pełna macierz p-values
    full_matrix_n = {}  # Pełna macierz liczby próbek
    
    total_calculations = len(overload_cols) * len(set_value_cols)
    current = 0
    
    print(f"Obliczam {total_calculations:,} korelacji...")
    print(f"({len(overload_cols)} linii × {len(set_value_cols)} elementów)")
    
    for overload_col in overload_cols:
        line_name = overload_col.replace('OverloadedLine_', '').replace('_loading_pct', '')
        line_correlations = []  # Tylko istotne >= 0.3 dla raportów
        line_weak_correlations = []  # ✅ NOWE - słabe korelacje
        
        full_matrix_r[line_name] = {}
        full_matrix_p[line_name] = {}
        full_matrix_n[line_name] = {}
        
        y = df[overload_col].values
        
        # Sprawdź czy są dane
        has_variation = not (np.all(np.isnan(y)) or len(np.unique(y[~np.isnan(y)])) < 2)
        
        for set_col in set_value_cols:
            current += 1
            
            element_name = set_col.replace('_P_set', '').replace('_Q_set', '')
            element_name = element_name.replace('Load_', '').replace('Gen_', '').replace('PV_', '').replace('StatGen_', '')
            
            x = df[set_col].values
            mask = ~(np.isnan(x) | np.isnan(y))
            
            # Domyślne wartości (brak danych)
            r_value = np.nan
            p_value = np.nan
            n_samples = 0
            
            if has_variation and mask.sum() >= 3:
                x_clean = x[mask]
                y_clean = y[mask]
                
                if np.std(x_clean) > 0 and np.std(y_clean) > 0:
                    try:
                        r_value, p_value = pearsonr(x_clean, y_clean)
                        n_samples = mask.sum()
                        
                        abs_r = abs(r_value)
                        
                        # Zapisz istotne (>= 0.3) do głównych raportów
                        if abs_r >= CONFIG.CORRELATION_THRESHOLD and p_value <= CONFIG.P_VALUE_THRESHOLD:
                            line_correlations.append({
                                'Element': element_name,
                                'Element_Full': set_col,
                                'Correlation_R': r_value,
                                'P_Value': p_value,
                                'N_Samples': n_samples
                            })
                        
                        # ✅ NOWE - Zapisz słabe (0.1-0.3) jeśli są istotne
                        elif 0.1 <= abs_r < 0.3 and p_value <= CONFIG.P_VALUE_THRESHOLD:
                            line_weak_correlations.append({
                                'Element': element_name,
                                'Element_Full': set_col,
                                'Correlation_R': r_value,
                                'P_Value': p_value,
                                'N_Samples': n_samples
                            })
                    
                    except Exception as e:
                        pass
            
            # Zapisz do pełnej macierzy (nawet NaN)
            full_matrix_r[line_name][element_name] = r_value
            full_matrix_p[line_name][element_name] = p_value
            full_matrix_n[line_name][element_name] = n_samples
            
            # Postęp
            if current % max(1, total_calculations // 10) == 0:
                progress = 100 * current / total_calculations
                print(f"  Postęp: {progress:.0f}%", end='\r')
        
        # Sortuj istotne korelacje
        if line_correlations:
            line_correlations.sort(key=lambda x: abs(x['Correlation_R']), reverse=True)
            correlations[line_name] = line_correlations
        
        # ✅ NOWE - Sortuj słabe korelacje
        if line_weak_correlations:
            line_weak_correlations.sort(key=lambda x: abs(x['Correlation_R']), reverse=True)
            weak_correlations[line_name] = line_weak_correlations
    
    print(f"  Postęp: 100%   ")
    print(f"✓ Obliczono {total_calculations:,} korelacji")
    print(f"✓ Istotne korelacje (|r| >= 0.3): {len(correlations)} linii")
    print(f"✓ Słabe korelacje (0.1 <= |r| < 0.3): {len(weak_correlations)} linii")
    
    return correlations, weak_correlations, full_matrix_r, full_matrix_p, full_matrix_n


def classify_correlation_strength(abs_r):
    """Klasyfikuj siłę korelacji"""
    if np.isnan(abs_r):
        return "N/A"
    for min_val, max_val, label in CONFIG.CORRELATION_RANGES:
        if min_val <= abs_r < max_val:
            return label
    if abs_r == 0.0:
        return "Brak"
    return "N/A"

def create_correlation_report(correlations):
    """Stwórz raport tekstowy z korelacjami"""
    print("\n" + "="*80)
    print("📋 RAPORT KORELACJI (tylko istotne)")
    print("="*80)
    
    if not correlations:
        print("⚠️ Brak istotnych korelacji spełniających kryteria")
        return
    
    for line_name, line_corrs in list(correlations.items())[:5]:
        print(f"\n{'='*80}")
        print(f"LINIA: {line_name}")
        print(f"{'='*80}")
        print(f"Znaleziono {len(line_corrs)} istotnych korelacji:")
        print()
        
        print(f"{'Element':<40} {'Korelacja':>12} {'p-value':>12} {'N':>6}")
        print("-"*80)
        
        for corr in line_corrs[:20]:
            element = corr['Element'][:38]
            r = corr['Correlation_R']
            p = corr['P_Value']
            n = corr['N_Samples']
            direction = "↑↑" if r > 0 else "↓↓"
            print(f"{element:<40} {r:>+11.4f} {direction} {p:>11.4e} {n:>6}")
        
        if len(line_corrs) > 20:
            print(f"... i {len(line_corrs) - 20} więcej")
    
    if len(correlations) > 5:
        print(f"\n... i {len(correlations) - 5} więcej linii (szczegóły w pliku Excel)")

def save_correlations_to_excel(correlations, weak_correlations, full_matrix_r, full_matrix_p, full_matrix_n, input_file):
    """Zapisz korelacje do Excela"""
    print("\n" + "="*80)
    print("💾 ZAPISYWANIE DO EXCELA")
    print("="*80)
    
    input_path = Path(input_file)
    output_file = input_path.parent / f"Correlations_{input_path.stem}.xlsx"
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        
        # ========================================
        # ARKUSZ 1: PEŁNA MACIERZ - WSPÓŁCZYNNIKI R
        # ========================================
        print("  📊 Tworzenie pełnej macierzy współczynników r...")
        df_full_r = pd.DataFrame(full_matrix_r).T
        df_full_r.index.name = 'Line \\ Element'
        df_full_r.to_excel(writer, sheet_name='Full_Matrix_R')
        print(f"  ✓ Arkusz 'Full_Matrix_R': {df_full_r.shape[0]} linii × {df_full_r.shape[1]} elementów")
        
        # ========================================
        # ARKUSZ 2: PEŁNA MACIERZ - P-VALUES
        # ========================================
        print("  📊 Tworzenie pełnej macierzy p-values...")
        df_full_p = pd.DataFrame(full_matrix_p).T
        df_full_p.index.name = 'Line \\ Element'
        df_full_p.to_excel(writer, sheet_name='Full_Matrix_P')
        print(f"  ✓ Arkusz 'Full_Matrix_P': {df_full_p.shape[0]} linii × {df_full_p.shape[1]} elementów")
        
        # ========================================
        # ARKUSZ 3: PEŁNA MACIERZ - LICZBA PRÓBEK
        # ========================================
        print("  📊 Tworzenie pełnej macierzy liczby próbek...")
        df_full_n = pd.DataFrame(full_matrix_n).T
        df_full_n.index.name = 'Line \\ Element'
        df_full_n.to_excel(writer, sheet_name='Full_Matrix_N')
        print(f"  ✓ Arkusz 'Full_Matrix_N': {df_full_n.shape[0]} linii × {df_full_n.shape[1]} elementów")
        
        # ========================================
        # ✅ NOWY ARKUSZ: SŁABE KORELACJE (0.1-0.3)
        # ========================================
        if weak_correlations:
            print("\n  📊 Tworzenie arkusza słabych korelacji...")
            weak_corrs_data = []
            for line_name, line_corrs in weak_correlations.items():
                for corr in line_corrs:
                    abs_r = abs(corr['Correlation_R'])
                    weak_corrs_data.append({
                        'Line_Name': line_name,
                        'Element': corr['Element'],
                        'Element_Full_Name': corr['Element_Full'],
                        'Correlation_R': corr['Correlation_R'],
                        'Abs_Correlation_R': abs_r,
                        'Strength': classify_correlation_strength(abs_r),
                        'P_Value': corr['P_Value'],
                        'N_Samples': corr['N_Samples'],
                        'Direction': 'Positive' if corr['Correlation_R'] > 0 else 'Negative',
                    })
            
            if weak_corrs_data:
                df_weak = pd.DataFrame(weak_corrs_data)
                df_weak = df_weak.sort_values(['Line_Name', 'Abs_Correlation_R'], ascending=[True, False])
                df_weak.to_excel(writer, sheet_name='Weak_Correlations', index=False)
                print(f"  ✓ Arkusz 'Weak_Correlations': {len(df_weak)} korelacji (0.1 <= |r| < 0.3)")
                
                # Dodatkowy podział na dodatnie i ujemne
                df_weak_positive = df_weak[df_weak['Correlation_R'] > 0]
                df_weak_negative = df_weak[df_weak['Correlation_R'] < 0]
                
                if not df_weak_positive.empty:
                    df_weak_positive.to_excel(writer, sheet_name='Weak_Positive', index=False)
                    print(f"  ✓ Arkusz 'Weak_Positive': {len(df_weak_positive)} korelacji (0.1 <= r < 0.3)")
                
                if not df_weak_negative.empty:
                    df_weak_negative.to_excel(writer, sheet_name='Weak_Negative', index=False)
                    print(f"  ✓ Arkusz 'Weak_Negative': {len(df_weak_negative)} korelacji (-0.3 < r <= -0.1)")
        
        # ========================================
        # ARKUSZ 4: PODSUMOWANIE (tylko istotne >= 0.3)
        # ========================================
        if correlations:
            summary_data = []
            for line_name, line_corrs in correlations.items():
                if line_corrs:
                    max_corr = max(line_corrs, key=lambda x: abs(x['Correlation_R']))
                    
                    # ✅ Dodaj info o słabych korelacjach
                    num_weak = len(weak_correlations.get(line_name, []))
                    
                    summary_data.append({
                        'Line_Name': line_name,
                        'Num_Significant_Correlations': len(line_corrs),
                        'Num_Weak_Correlations': num_weak,  # ✅ NOWE
                        'Max_Abs_Correlation': abs(max_corr['Correlation_R']),
                        'Max_Corr_Element': max_corr['Element'],
                        'Max_Corr_R': max_corr['Correlation_R'],
                        'Max_Corr_P': max_corr['P_Value'],
                    })
            
            df_summary = pd.DataFrame(summary_data)
            df_summary = df_summary.sort_values('Max_Abs_Correlation', ascending=False)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            print(f"\n  ✓ Arkusz 'Summary': {len(df_summary)} linii")
        
        # ========================================
        # ARKUSZ 5: Wszystkie korelacje (długi format, tylko >= 0.3)
        # ========================================
        if correlations:
            all_corrs_data = []
            for line_name, line_corrs in correlations.items():
                for corr in line_corrs:
                    abs_r = abs(corr['Correlation_R'])
                    all_corrs_data.append({
                        'Line_Name': line_name,
                        'Element': corr['Element'],
                        'Element_Full_Name': corr['Element_Full'],
                        'Correlation_R': corr['Correlation_R'],
                        'Abs_Correlation_R': abs_r,
                        'Strength': classify_correlation_strength(abs_r),
                        'P_Value': corr['P_Value'],
                        'N_Samples': corr['N_Samples'],
                        'Direction': 'Positive' if corr['Correlation_R'] > 0 else 'Negative',
                    })
            
            df_all = pd.DataFrame(all_corrs_data)
            df_all = df_all.sort_values(['Line_Name', 'Abs_Correlation_R'], ascending=[True, False])
            df_all.to_excel(writer, sheet_name='All_Correlations', index=False)
            print(f"  ✓ Arkusz 'All_Correlations': {len(df_all)} korelacji (|r| >= 0.3)")
        
        # [pozostała część bez zmian: arkusze dla top 10 linii, Top_Matrix, klasyfikacja wg siły]
        # ... (linie 339-408 bez zmian)
        
        # ========================================
        # ARKUSZ 6+: Osobne arkusze dla top 10 linii
        # ========================================
        if correlations:
            df_summary_sorted = df_summary.sort_values('Max_Abs_Correlation', ascending=False)
            top_lines = df_summary_sorted.head(10)['Line_Name'].tolist()
            
            for line_name in top_lines:
                if line_name in correlations:
                    line_corrs = correlations[line_name]
                    df_line = pd.DataFrame(line_corrs)
                    df_line['Abs_R'] = df_line['Correlation_R'].abs()
                    df_line['Strength'] = df_line['Abs_R'].apply(classify_correlation_strength)
                    df_line = df_line.sort_values('Abs_R', ascending=False)
                    
                    sheet_name = f"Line_{line_name}"[:31]
                    df_line.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  ✓ Arkusz '{sheet_name}': {len(df_line)} elementów")
        
        # ========================================
        # ARKUSZ: Macierz korelacji (top 20 x top 50)
        # ========================================
        if correlations:
            top_20_lines = df_summary_sorted.head(20)['Line_Name'].tolist()
            element_counts = {}
            for line_corrs in correlations.values():
                for corr in line_corrs:
                    elem = corr['Element']
                    element_counts[elem] = element_counts.get(elem, 0) + 1
            
            top_elements = sorted(element_counts.keys(), key=lambda e: element_counts[e], reverse=True)[:50]
            
            matrix_data = []
            for line_name in top_20_lines:
                row = {'Line': line_name}
                if line_name in correlations:
                    for corr in correlations[line_name]:
                        if corr['Element'] in top_elements:
                            row[corr['Element']] = corr['Correlation_R']
                matrix_data.append(row)
            
            if matrix_data:
                df_matrix = pd.DataFrame(matrix_data)
                df_matrix = df_matrix.set_index('Line')
                df_matrix.to_excel(writer, sheet_name='Top_Matrix')
                print(f"  ✓ Arkusz 'Top_Matrix': {df_matrix.shape[0]}x{df_matrix.shape[1]}")
        
        # ========================================
        # ARKUSZE: KLASYFIKACJA WG SIŁY
        # ========================================
        if correlations:
            print("\n  📊 Tworzenie arkuszy wg siły korelacji...")
            
            for min_val, max_val, label in CONFIG.CORRELATION_RANGES:
                filtered = [c for c in all_corrs_data if min_val <= c['Abs_Correlation_R'] < max_val]
                
                if filtered:
                    df_strength = pd.DataFrame(filtered)
                    df_strength = df_strength.sort_values('Abs_Correlation_R', ascending=False)
                    
                    sheet_name = f"{label} ({min_val}-{max_val})"[:31]
                    df_strength.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  ✓ Arkusz '{sheet_name}': {len(df_strength)} korelacji")
            
            # Brak korelacji (0.0)
            zero_corrs = [c for c in all_corrs_data if c['Abs_Correlation_R'] == 0.0]
            if zero_corrs:
                df_zero = pd.DataFrame(zero_corrs)
                df_zero.to_excel(writer, sheet_name='Brak (0.0)', index=False)
                print(f"  ✓ Arkusz 'Brak (0.0)': {len(df_zero)} korelacji")
    
    print(f"\n✅ Zapisano do: {output_file}")
    return output_file

def create_interpretation_guide(correlations, output_file):
    """Stwórz plik tekstowy z interpretacją wyników"""
    output_file_str = str(output_file)
    guide_file = output_file_str.replace('.xlsx', '_interpretation.txt')
    
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("INTERPRETACJA WYNIKÓW ANALIZY KORELACJI PEARSONA\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. ARKUSZE W EXCELU\n")
        f.write("-"*80 + "\n")
        f.write("  Full_Matrix_R:   Pełna macierz współczynników r (11 linii × 527 elementów)\n")
        f.write("  Full_Matrix_P:   Pełna macierz p-values (istotność statystyczna)\n")
        f.write("  Full_Matrix_N:   Pełna macierz liczby próbek\n")
        f.write("  Summary:         Podsumowanie istotnych korelacji\n")
        f.write("  All_Correlations: Wszystkie istotne korelacje (długi format)\n")
        f.write("  Line_XXX:        Szczegóły dla poszczególnych linii (top 10)\n")
        f.write("  Top_Matrix:      Macierz top 20 linii × top 50 elementów\n")
        f.write("  Pełna/Bardzo_Silna/...: Klasyfikacja wg siły\n\n")
        
        f.write("2. WSPÓŁCZYNNIK KORELACJI PEARSONA (r)\n")
        f.write("-"*80 + "\n")
        f.write("  r ∈ [-1, 1]\n")
        f.write("  - r > 0: korelacja dodatnia (wzrost elementu → wzrost przeciążenia)\n")
        f.write("  - r < 0: korelacja ujemna (wzrost elementu → spadek przeciążenia)\n")
        f.write("  - r = NaN: brak danych lub brak zmienności\n\n")
        
        f.write("  KLASYFIKACJA SIŁY KORELACJI:\n")
        f.write("  " + "-"*76 + "\n")
        for min_val, max_val, label in CONFIG.CORRELATION_RANGES:
            f.write(f"  - {label:20s}: {min_val:.1f} ≤ |r| < {max_val:.1f}\n")
        f.write("  - Brak:                0.0 = |r|\n\n")
        
        f.write("3. WARTOŚĆ p (p-value)\n")
        f.write("-"*80 + "\n")
        f.write(f"  Próg istotności: p < {CONFIG.P_VALUE_THRESHOLD}\n")
        f.write("  - p < 0.05: korelacja statystycznie istotna\n")
        f.write("  - p < 0.01: korelacja bardzo istotna\n")
        f.write("  - p < 0.001: korelacja wysoce istotna\n\n")
        
        if correlations:
            f.write("4. NAJWAŻNIEJSZE WYNIKI\n")
            f.write("="*80 + "\n\n")
            
            top_5_lines = sorted(correlations.items(), 
                                key=lambda x: max([abs(c['Correlation_R']) for c in x[1]]) if x[1] else 0, 
                                reverse=True)[:5]
            
            for i, (line_name, line_corrs) in enumerate(top_5_lines, 1):
                if not line_corrs:
                    continue
                
                f.write(f"{i}. LINIA: {line_name}\n")
                f.write("-"*80 + "\n")
                
                top_3 = sorted(line_corrs, key=lambda x: abs(x['Correlation_R']), reverse=True)[:3]
                
                for j, corr in enumerate(top_3, 1):
                    r = corr['Correlation_R']
                    p = corr['P_Value']
                    elem = corr['Element']
                    abs_r = abs(r)
                    strength = classify_correlation_strength(abs_r)
                    
                    direction = "ZWIĘKSZENIE" if r > 0 else "ZMNIEJSZENIE"
                    
                    f.write(f"  {j}. {elem}\n")
                    f.write(f"     Korelacja: r = {r:+.4f} ({strength.replace('_', ' ')})\n")
                    f.write(f"     {direction} mocy elementu → {'WZROST' if r > 0 else 'SPADEK'} przeciążenia\n")
                    f.write(f"     Istotność: p = {p:.4e}\n\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("UWAGI:\n")
        f.write("- Pełna macierz zawiera WSZYSTKIE pary (linia × element), nawet nieistotne\n")
        f.write("- NaN oznacza brak danych lub brak zmienności\n")
        f.write("- Korelacja NIE oznacza związku przyczynowo-skutkowego!\n")
        f.write("- Analizowane są tylko zależności liniowe\n")
        f.write(f"- W raportach wyświetlane tylko: |r| > {CONFIG.CORRELATION_THRESHOLD} i p < {CONFIG.P_VALUE_THRESHOLD}\n")
        f.write("="*80 + "\n")
    
    print(f"✅ Zapisano interpretację do: {guide_file}")
    return guide_file

# ==========================================
# MAIN
# ==========================================
def main():
    """Główna funkcja"""
    print("\n" + "="*80)
    print("🔬 ANALIZA KORELACJI PEARSONA - MONTE CARLO RESULTS")
    print("="*80)
    print(f"Próg korelacji dla raportów: |r| > {CONFIG.CORRELATION_THRESHOLD}")
    print(f"Próg istotności: p < {CONFIG.P_VALUE_THRESHOLD}")
    print("\nKlasyfikacja siły:")
    for min_val, max_val, label in CONFIG.CORRELATION_RANGES:
        print(f"  - {label:20s}: [{min_val:.1f}, {max_val:.1f})")
    print("="*80)
    
    try:
        if CONFIG.INPUT_FILE is None:
            input_file = find_latest_results_file(CONFIG.RESULTS_DIR)
        else:
            input_file = Path(CONFIG.INPUT_FILE)
        
        if input_file is None:
            return
        
        df = load_results(input_file)
        set_value_cols, overload_cols = identify_columns(df)
        
        if not set_value_cols:
            print("\n❌ Nie znaleziono kolumn z nastawianymi wartościami (_set)")
            return
        
        if not overload_cols:
            print("\n❌ Nie znaleziono kolumn z przeciążonymi liniami (OverloadedLine_)")
            return
        
        # OBLICZ PEŁNĄ MACIERZ
        correlations, weak_correlations, full_matrix_r, full_matrix_p, full_matrix_n = calculate_correlations_full(
            df, set_value_cols, overload_cols
        )
        
        create_correlation_report(correlations)
        
        output_file = save_correlations_to_excel(
            correlations, weak_correlations, full_matrix_r, full_matrix_p, full_matrix_n, input_file
        )
        
        if output_file:
            create_interpretation_guide(correlations, output_file)
        
        print("\n" + "="*80)
        print("✅ ANALIZA ZAKOŃCZONA POMYŚLNIE")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ BŁĄD:")
        print(f"  {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()