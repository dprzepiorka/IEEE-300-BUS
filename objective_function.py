"""
Funkcja celu dla optymalizacji PowerFactory
WERSJA Z WYMUSZONYM DEBUGOWANIEM - NAPRAWIONA
"""

import numpy as np
import os
import time

# ==========================================
# KONFIGURACJA
# ==========================================
#OBSERVED_LINES = [257, 306, 304, 226, 271, 289, 286]
observed_lineS = ['5.73',	'8.14',	'79.73',	'35.77',	'6.2',	'73.71',	'71.72']
VOLTAGE_MIN = 0.9
VOLTAGE_MAX = 1.1

DEBUG_MODE = True  
DEBUG_FIRST_N = 10  
VERBOSE_CONSOLE = True

PENALTY_VOLTAGE = 0.0
PENALTY_OVERLOAD = 0.0
PENALTY_LF_FAIL = 1e10

print("="*80)
print("⚠️ OBJECTIVE_FUNCTION.PY - TRYB DEBUG")
print(f"   Kary wyłączone:V={PENALTY_VOLTAGE}, O={PENALTY_OVERLOAD}, LF={PENALTY_LF_FAIL}")
print("="*80)

class PowerFactoryObjective:
    
    def __init__(self, app, ldf, opt_variables, observed_lines=observed_lineS, log_file=None):
        self.app = app
        self.ldf = ldf
        self.opt_variables = opt_variables
        self.observed_lines = observed_lines or OBSERVED_LINES
        self.log_file = log_file
        
        self.eval_count = 0
        self.best_value = np.inf
        self.best_x = None
        self._element_cache = {}
        
        # Plik DEBUG (BEZ SPACJI!)
        if log_file:
            debug_dir = os.path.dirname(log_file)
        else:
            debug_dir = r"N:\ksiz\STUDIA_DOKTORANCKIE\PRZEPIORKA\Modele sieci\Wysokie napiecie\PYTHON\Wyniki"
        
        self.debug_file = os.path.join(debug_dir, "DEBUG_OBJECTIVE.txt")  # ← POPRAWIONE
        
        try:
            with open(self.debug_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("DEBUG FUNKCJI CELU\n")
                f.write(f"Start:{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
            print(f"✅ DEBUG file:{self.debug_file}")
        except Exception as e:
            print(f"❌ Błąd DEBUG file:{e}")
            self.debug_file = None
        
        self._cache_elements()
    
    def _cache_elements(self):
        print(f"\n🔍 Cachowanie {len(self.opt_variables)} zmiennych...")
        cached = 0
        
        for var in self.opt_variables:
            key = f"{var['Element_Name']}_{var['Element_Type']}"
            if key not in self._element_cache:
                elem = self._find_element(var['Element_Name'], var['Element_Type'])
                if elem:
                    self._element_cache[key] = elem
                    cached += 1
                else:
                    print(f"  ⚠️ Brak:{var['Element_Name']}")
        
        print(f"✅ Cached:{cached}/{len(self.opt_variables)}")
    
    def _find_element(self, name, pf_class):
        try:
            objs = self.app.GetCalcRelevantObjects(f"{name}.{pf_class}")  # ← BEZ SPACJI
            if objs and len(objs) > 0:
                return objs[0]
        except:
            pass
        
        try:
            all_objs = self.app.GetCalcRelevantObjects(f"*.{pf_class}")  # ← BEZ SPACJI
            if all_objs:
                for obj in all_objs:
                    if getattr(obj, "loc_name", None) == name:
                        return obj
        except:
            pass
        
        return None
    
    def set_variables(self, x):
        idx = 0
        for var in self.opt_variables:
            key = f"{var['Element_Name']}_{var['Element_Type']}"
            elem = self._element_cache.get(key)
            
            if elem:
                if 'Attribute_P' in var and var['Attribute_P']:
                    try:
                        elem.SetAttribute(var['Attribute_P'], float(x[idx]))
                    except:
                        pass
                    idx += 1
                
                if 'Attribute_Q' in var and var['Attribute_Q']:
                    try:
                        elem.SetAttribute(var['Attribute_Q'], float(x[idx]))
                    except:
                        pass
                    idx += 1
    
    def run_load_flow(self):
        try:
            code = self.ldf.Execute()
            return code == 0
        except:
            return False
    
    def calculate_overloads(self):
        overload = 0.0
        details = []
        
        try:
            lines = self.app.GetCalcRelevantObjects("*.ElmLne")  # ← BEZ SPACJI
            
            if isinstance(self.observed_lines[0], int):
                for idx in self.observed_lines:
                    if idx < len(lines):
                        try:
                            loading = lines[idx].GetAttribute("c:loading")  # ← BEZ SPACJI! 
                            if loading and loading > 100:
                                excess = loading - 100
                                overload += excess
                                details.append(f"#{idx} ({lines[idx].loc_name}):{loading:.2f}% → +{excess:.2f}")
                        except:
                            pass
            else:
                for line in lines:
                    if line.loc_name in self.observed_lines:
                        try:
                            loading = line.GetAttribute("c:loading")  # ← BEZ SPACJI!
                            if loading and loading > 100:
                                excess = loading - 100
                                overload += excess
                                details.append(f"{line.loc_name}:{loading:.2f}% → +{excess:.2f}")
                        except:
                            pass
        except Exception as e:
            details.append(f"ERROR:{e}")
        
        return overload, details
    
    def calculate_voltage_violations(self):
        violation = 0.0
        details = []
        
        try:
            buses = self.app.GetCalcRelevantObjects("*.ElmTerm")
            
            for bus in buses:
                try:
                    u_pu = bus.GetAttribute("m:u")  # ← BEZ SPACJI!
                    if u_pu:
                        if u_pu > VOLTAGE_MAX:
                            excess = u_pu - VOLTAGE_MAX
                            violation += excess
                            details.append(f"{bus.loc_name}:{u_pu:.4f} p.u.(>{VOLTAGE_MAX})")
                        elif u_pu < VOLTAGE_MIN:
                            excess = VOLTAGE_MIN - u_pu
                            violation += excess
                            details.append(f"{bus.loc_name}:{u_pu:.4f} p.u.(<{VOLTAGE_MIN})")
                except:
                    pass
        except:
            pass
        
        return violation, details[:5]
    
    def calculate_line_overloads_all(self):
        overload = 0.0
        details = []
        
        try:
            lines = self.app.GetCalcRelevantObjects("*.ElmLne")
            
            for line in lines:
                try:
                    loading = line.GetAttribute("c:loading")  # ← BEZ SPACJI!
                    if loading and loading > 100:
                        excess = loading - 100
                        overload += excess
                        details.append(f"{line.loc_name}:{loading:.2f}%")
                except:
                    pass
        except:
            pass
        
        return overload, details[:10]
    
    def calculate_trafo_overloads(self):
        overload = 0.0
        details = []
        
        try:
            trafos = self.app.GetCalcRelevantObjects("*.ElmTr2")
            
            for trafo in trafos:
                try:
                    loading = trafo.GetAttribute("c:loading")  # ← BEZ SPACJI!
                    if loading and loading > 100:
                        excess = loading - 100
                        overload += excess
                        details.append(f"{trafo.loc_name}:{loading:.2f}%")
                except:
                    pass
        except:
            pass
        
        return overload, details
    
    def _write_debug(self, message):
        if self.debug_file:
            try:
                with open(self.debug_file, 'a', encoding='utf-8') as f:
                    f.write(message + "\n")
            except:
                pass
    
    def __call__(self, x):
        self.eval_count += 1
        show_details = (self.eval_count <= DEBUG_FIRST_N)
        
        if show_details or VERBOSE_CONSOLE:
            msg = f"\n{'='*60}\nEVAL #{self.eval_count}\n{'='*60}"
            print(msg)
            self._write_debug(msg)
        
        self.set_variables(x)
        lf_success = self.run_load_flow()
        
        if show_details:
            lf_msg = f"LF:{'✓' if lf_success else '✗'}"
            print(f"  {lf_msg}")
            self._write_debug(lf_msg)
        
        if not lf_success:
            fail_msg = f"→ {PENALTY_LF_FAIL} (LF failed)"
            print(f"  {fail_msg}")
            self._write_debug(fail_msg)
            return PENALTY_LF_FAIL
        
        overload_obs, obs_det = self.calculate_overloads()
        volt_viol, volt_det = self.calculate_voltage_violations()
        overload_all, all_det = self.calculate_line_overloads_all()
        overload_trf, trf_det = self.calculate_trafo_overloads()
        
        if show_details:
            self._write_debug(f"\nSKŁADNIKI:")
            self._write_debug(f"  Obserwowane:{overload_obs:.6f}")
            if obs_det:
                for d in obs_det:
                    self._write_debug(f"    {d}")
                    print(f"    {d}")
            
            self._write_debug(f"  Napięcia:{volt_viol:.6f}")
            self._write_debug(f"  Linie (all):{overload_all:.6f}")
            self._write_debug(f"  Trafos:{overload_trf:.6f}")
        
        f_objective = overload_obs
        penalty = volt_viol * PENALTY_VOLTAGE + overload_all * PENALTY_OVERLOAD + overload_trf * PENALTY_OVERLOAD
        f_total = f_objective + penalty
        
        result_msg = f"\nWYNIK:obj={f_objective:.3f}, penalty={penalty:.3f}, total={f_total:.3f}"
        
        if show_details or VERBOSE_CONSOLE:
            print(result_msg)
        
        self._write_debug(result_msg)
        
        if f_total < self.best_value:
            self.best_value = f_total
            self.best_x = x.copy()
            best_msg = f"\n✅ NEW BEST:{f_total:.3f}"
            print(best_msg)
            self._write_debug(best_msg)
        
        if self.log_file and (self.eval_count % 10 == 0):
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(f"{self.eval_count},{time.time()},{f_objective:.6f},{penalty:.6f},{f_total:.6f},")
                    f.write(','.join([f"{xi:.6f}" for xi in x]) + '\n')
            except:
                pass
        
        return f_total