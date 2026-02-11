import numpy as np
import time
import traceback
import os

class PO:
    """
    Puma Optimizer (PO) with early stopping capability.
    Based on:
    Abdollahzadeh et al. (2023), Puma optimizer (PO): a novel metaheuristic optimization algorithm
    DOI: 10.1007/s10586-023-04221-5
    
    Usage:
        po = PO(func, n_solutions, dim, lb, ub, max_iter,
                autosave_every_iters=5, autosave_path="po_checkpoint.npz", eval_delay=0.0,
                early_stop_threshold=0.0, early_stop_patience=50)
        res = po.optimize()
    Returns dict:{"gbest":..., "gbest_val":..., "best_per_iter":[...], "early_stopped":..., "reason":..., "stopped_at_iter":...}
    """
    def __init__(self, func, n_solutions, dim, lb, ub, max_iter,
                 autosave_every_iters=0, autosave_path="po_checkpoint.npz", eval_delay=0.0,
                 early_stop_threshold=None, early_stop_patience=100):
        self.func = func
        self.n_solutions = int(n_solutions)
        self.dim = int(dim)
        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)
        self.max_iter = int(max_iter)

        self.autosave_every_iters = int(autosave_every_iters)
        self.autosave_path = autosave_path
        self.eval_delay = float(eval_delay)
        
        # ✅ EARLY STOPPING parameters
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.no_improvement_count = 0
        self.last_best = np.inf

        # PO-specific parameters
        self.PF = np.array([0.5, 0.5, 0.3])  # Performance factors
        self.Mega_Explor = 0.99
        self.Mega_Exploit = 0.99
        
        # Initialize solutions
        self.solutions = []
        for i in range(self.n_solutions):
            x = np.random.uniform(self.lb, self.ub, self.dim)
            cost = self._eval_solution(x)
            self.solutions.append({'X': x.copy(), 'Cost': cost})
        
        # Find initial best
        self.best = min(self.solutions, key=lambda s: s['Cost']).copy()
        self.initial_best = self.best.copy()
        self.best_per_iter = []
        
        # Tracking variables
        self.UnSelected = np.ones(2)  # [Exploration, Exploitation]
        self.F3_Explore = 0.0
        self.F3_Exploit = 0.0
        self.Seq_Time_Explore = np.ones(3)
        self.Seq_Time_Exploit = np.ones(3)
        self.Seq_Cost_Explore = np.ones(3)
        self.Seq_Cost_Exploit = np.ones(3)
        self.Score_Explore = 0.0
        self.Score_Exploit = 0.0
        self.PF_F3 = []
        
        self.iter = 0
        self.Flag_Change = 1

    def _eval_solution(self, x):
        """Evaluate a solution"""
        try:
            val = float(self.func(x))
        except Exception as e:
            try:
                with open("failed_evals_po.csv", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()},{','.join(map(str, x))},exception,{str(e)}\n")
            except Exception:
                pass
            val = np.inf
        if self.eval_delay:
            time.sleep(self.eval_delay)
        return val

    def _exploration(self):
        """Exploration phase (Eq 25-30)"""
        # Sort solutions by cost
        self.solutions.sort(key=lambda s: s['Cost'])
        
        pCR = 0.20
        PCR = 1 - pCR
        p = PCR / self.n_solutions
        
        new_solutions = []
        for i in range(self.n_solutions):
            x = self.solutions[i]['X'].copy()
            
            # Select 6 random distinct solutions
            indices = list(range(self.n_solutions))
            indices.remove(i)
            np.random.shuffle(indices)
            a, b, c, d, e, f = indices[:6]
            
            G = 2 * np.random.rand() - 1  # Eq 26
            
            if np.random.rand() < 0.5:
                # Random exploration
                y = np.random.rand(self.dim) * (self.ub - self.lb) + self.lb  # Eq 25
            else:
                # Differential evolution-based exploration
                y = (self.solutions[a]['X'] + 
                     G * (self.solutions[a]['X'] - self.solutions[b]['X']) +
                     G * (((self.solutions[a]['X'] - self.solutions[b]['X']) - 
                           (self.solutions[c]['X'] - self.solutions[d]['X'])) +
                          ((self.solutions[c]['X'] - self.solutions[d]['X']) - 
                           (self.solutions[e]['X'] - self.solutions[f]['X']))))  # Eq 25
            
            # Boundary check
            y = np.clip(y, self.lb, self.ub)
            
            # Crossover
            z = x.copy()
            j0 = np.random.randint(0, self.dim)
            for j in range(self.dim):
                if j == j0 or np.random.rand() <= pCR:
                    z[j] = y[j]
            
            new_cost = self._eval_solution(z)
            
            if new_cost < self.solutions[i]['Cost']:
                new_solutions.append({'X': z.copy(), 'Cost': new_cost})
            else:
                new_solutions.append(self.solutions[i].copy())
                pCR = pCR + p  # Eq 30
        
        self.solutions = new_solutions
        return min(self.solutions, key=lambda s: s['Cost'])

    def _exploitation(self):
        """Exploitation phase (Eq 32-38)"""
        Q = 0.67
        Beta = 2.0
        
        new_solutions = []
        for i in range(self.n_solutions):
            beta1 = 2 * np.random.rand()
            beta2 = np.random.randn(self.dim)
            w = np.random.randn(self.dim)  # Eq 37
            v = np.random.randn(self.dim)  # Eq 38
            
            F1 = np.random.randn(self.dim) * np.exp(2 - self.iter * (2 / self.max_iter))  # Eq 35
            F2 = w * v**2 * np.cos(2 * np.random.rand() * w)  # Eq 36
            
            mbest = np.mean([s['X'] for s in self.solutions], axis=0) / self.n_solutions
            R_1 = 2 * np.random.rand() - 1  # Eq 34
            
            S1 = 2 * np.random.rand() - 1 + np.random.randn(self.dim)
            S2 = F1 * R_1 * self.solutions[i]['X'] + F2 * (1 - R_1) * self.best['X']
            
            # Avoid division by zero
            S1 = np.where(np.abs(S1) < 1e-10, 1e-10, S1)
            VEC = S2 / S1
            
            if np.random.rand() <= 0.5:
                Xatack = VEC
                if np.random.rand() > Q:
                    r_idx = np.random.randint(0, self.n_solutions)
                    new_x = self.best['X'] + beta1 * np.exp(beta2) * (self.solutions[r_idx]['X'] - self.solutions[i]['X'])  # Eq 32
                else:
                    new_x = beta1 * Xatack - self.best['X']  # Eq 32
            else:
                r1 = np.random.randint(0, self.n_solutions)  # Eq 33
                sign = (-1) ** np.random.randint(0, 2)
                new_x = (mbest * self.solutions[r1]['X'] - sign * self.solutions[i]['X']) / (1 + Beta * np.random.rand())  # Eq 32
            
            # Boundary check
            new_x = np.clip(new_x, self.lb, self.ub)
            new_cost = self._eval_solution(new_x)
            
            if new_cost < self.solutions[i]['Cost']:
                new_solutions.append({'X': new_x.copy(), 'Cost': new_cost})
            else:
                new_solutions.append(self.solutions[i].copy())
        
        self.solutions = new_solutions
        return min(self.solutions, key=lambda s: s['Cost'])

    def save_checkpoint(self, path=None):
        """Save checkpoint"""
        if path is None:
            path = self.autosave_path
        try:
            solutions_X = np.array([s['X'] for s in self.solutions])
            solutions_Cost = np.array([s['Cost'] for s in self.solutions])
            
            np.savez(path,
                     solutions_X=solutions_X,
                     solutions_Cost=solutions_Cost,
                     best_X=self.best['X'],
                     best_Cost=self.best['Cost'],
                     best_per_iter=np.array(self.best_per_iter),
                     iter=self.iter,
                     lb=self.lb,
                     ub=self.ub)
            
            # Human-readable history
            with open(os.path.splitext(path)[0] + "_history.txt", "w", encoding="utf-8") as f:
                for el in self.best_per_iter:
                    f.write(f"{el}\n")
            print(f"  💾 Checkpoint saved:{os.path.basename(path)}")
        except Exception as e:
            try:
                with open("po_save_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} save error:{e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass

    def optimize(self):
        """Main optimization loop"""
        try:
            # ✅ Print configuration
            print("\n" + "="*80)
            print("🚀 PUMA OPTIMIZER (PO) - EARLY STOPPING ENABLED")
            print("="*80)
            print(f"📊 Configuration:")
            print(f"   Solutions:{self.n_solutions}")
            print(f"   Max iterations:{self.max_iter}")
            print(f"   Dimensions:{self.dim}")
            if self.early_stop_threshold is not None:
                print(f"   🎯 Target threshold:f <= {self.early_stop_threshold}")
            if self.early_stop_patience:
                print(f"   ⏸️  Patience:{self.early_stop_patience} iterations without improvement")
            print("="*80 + "\n")
            
            # ===== UNEXPERIENCED PHASE (First 3 iterations) =====
            print("🔄 Phase 1: Unexperienced Phase (Iterations 1-3)...")
            Costs_Explor = []
            Costs_Exploit = []
            
            for it in range(1, 4):
                self.iter = it
                
                # Run exploration
                TBest_Explor = self._exploration()
                Costs_Explor.append(TBest_Explor['Cost'])
                
                # Run exploitation
                TBest_Exploit = self._exploitation()
                Costs_Exploit.append(TBest_Exploit['Cost'])
                
                # Update best
                current_best = min(self.solutions, key=lambda s: s['Cost'])
                if current_best['Cost'] < self.best['Cost']:
                    self.best = current_best.copy()
                
                self.best_per_iter.append(self.best['Cost'])
                print(f"Iter {it:4d}/{self.max_iter} | Best:{self.best['Cost']:12.6f}")
            
            # ===== Initialize hyperparameters (Eq 1-12) =====
            self.Seq_Cost_Explore[0] = abs(self.initial_best['Cost'] - Costs_Explor[0])  # Eq 5
            self.Seq_Cost_Exploit[0] = abs(self.initial_best['Cost'] - Costs_Exploit[0])  # Eq 8
            self.Seq_Cost_Explore[1] = abs(Costs_Explor[1] - Costs_Explor[0])  # Eq 6
            self.Seq_Cost_Exploit[1] = abs(Costs_Exploit[1] - Costs_Exploit[0])  # Eq 9
            self.Seq_Cost_Explore[2] = abs(Costs_Explor[2] - Costs_Explor[1])  # Eq 7
            self.Seq_Cost_Exploit[2] = abs(Costs_Exploit[2] - Costs_Exploit[1])  # Eq 10
            
            # Build PF_F3 list
            for i in range(3):
                if self.Seq_Cost_Explore[i] != 0:
                    self.PF_F3.append(self.Seq_Cost_Explore[i])
                if self.Seq_Cost_Exploit[i] != 0:
                    self.PF_F3.append(self.Seq_Cost_Exploit[i])
            
            # Calculate F1 and F2
            F1_Explor = self.PF[0] * (self.Seq_Cost_Explore[0] / self.Seq_Time_Explore[0])  # Eq 1
            F1_Exploit = self.PF[0] * (self.Seq_Cost_Exploit[0] / self.Seq_Time_Exploit[0])  # Eq 2
            F2_Explor = self.PF[1] * (np.sum(self.Seq_Cost_Explore) / np.sum(self.Seq_Time_Explore))  # Eq 3
            F2_Exploit = self.PF[1] * (np.sum(self.Seq_Cost_Exploit) / np.sum(self.Seq_Time_Exploit))  # Eq 4
            
            # Calculate scores
            self.Score_Explore = self.PF[0] * F1_Explor + self.PF[1] * F2_Explor  # Eq 11
            self.Score_Exploit = self.PF[0] * F1_Exploit + self.PF[1] * F2_Exploit  # Eq 12
            
            self.last_best = self.best['Cost']
            
            # ===== EXPERIENCED PHASE (Remaining iterations) =====
            print(f"\n🎯 Phase 2: Experienced Phase (Iterations 4-{self.max_iter})...\n")
            
            for it in range(4, self.max_iter + 1):
                self.iter = it
                
                # Decide: Exploration or Exploitation
                if self.Score_Explore > self.Score_Exploit:
                    # ===== EXPLORATION =====
                    SelectFlag = 1
                    TBest = self._exploration()
                    
                    Count_select = self.UnSelected.copy()
                    self.UnSelected[1] += 1
                    self.UnSelected[0] = 1
                    self.F3_Explore = self.PF[2]
                    self.F3_Exploit += self.PF[2]
                    
                    self.Seq_Cost_Explore[2] = self.Seq_Cost_Explore[1]
                    self.Seq_Cost_Explore[1] = self.Seq_Cost_Explore[0]
                    self.Seq_Cost_Explore[0] = abs(self.best['Cost'] - TBest['Cost'])
                    
                    if self.Seq_Cost_Explore[0] != 0:
                        self.PF_F3.append(self.Seq_Cost_Explore[0])
                else:
                    # ===== EXPLOITATION =====
                    SelectFlag = 2
                    TBest = self._exploitation()
                    
                    Count_select = self.UnSelected.copy()
                    self.UnSelected[0] += 1
                    self.UnSelected[1] = 1
                    self.F3_Explore += self.PF[2]
                    self.F3_Exploit = self.PF[2]
                    
                    self.Seq_Cost_Exploit[2] = self.Seq_Cost_Exploit[1]
                    self.Seq_Cost_Exploit[1] = self.Seq_Cost_Exploit[0]
                    self.Seq_Cost_Exploit[0] = abs(self.best['Cost'] - TBest['Cost'])
                    
                    if self.Seq_Cost_Exploit[0] != 0:
                        self.PF_F3.append(self.Seq_Cost_Exploit[0])
                
                # Update best
                if TBest['Cost'] < self.best['Cost']:
                    self.best = TBest.copy()
                
                # Update time sequences when switching phase
                if self.Flag_Change != SelectFlag:
                    self.Flag_Change = SelectFlag
                    self.Seq_Time_Explore[2] = self.Seq_Time_Explore[1]
                    self.Seq_Time_Explore[1] = self.Seq_Time_Explore[0]
                    self.Seq_Time_Explore[0] = Count_select[0]
                    self.Seq_Time_Exploit[2] = self.Seq_Time_Exploit[1]
                    self.Seq_Time_Exploit[1] = self.Seq_Time_Exploit[0]
                    self.Seq_Time_Exploit[0] = Count_select[1]
                
                # ===== Recalculate scores (Eq 13-20) =====
                F1_Explor = self.PF[0] * (self.Seq_Cost_Explore[0] / self.Seq_Time_Explore[0])  # Eq 14
                F1_Exploit = self.PF[0] * (self.Seq_Cost_Exploit[0] / self.Seq_Time_Exploit[0])  # Eq 13
                F2_Explor = self.PF[1] * (np.sum(self.Seq_Cost_Explore) / np.sum(self.Seq_Time_Explore))  # Eq 16
                F2_Exploit = self.PF[1] * (np.sum(self.Seq_Cost_Exploit) / np.sum(self.Seq_Time_Exploit))  # Eq 15
                
                # Adaptive weights (Eq 17-18)
                if self.Score_Explore < self.Score_Exploit:
                    self.Mega_Explor = max(self.Mega_Explor - 0.01, 0.01)
                    self.Mega_Exploit = 0.99
                elif self.Score_Explore > self.Score_Exploit:
                    self.Mega_Explor = 0.99
                    self.Mega_Exploit = max(self.Mega_Exploit - 0.01, 0.01)
                
                lmn_Explore = 1 - self.Mega_Explor  # Eq 24
                lmn_Exploit = 1 - self.Mega_Exploit  # Eq 22
                
                min_PF_F3 = min(self.PF_F3) if len(self.PF_F3) > 0 else 1.0
                
                # Update scores (Eq 19-20)
                self.Score_Exploit = (self.Mega_Exploit * F1_Exploit + 
                                     self.Mega_Exploit * F2_Exploit + 
                                     lmn_Exploit * min_PF_F3 * self.F3_Exploit)  # Eq 19
                self.Score_Explore = (self.Mega_Explor * F1_Explor + 
                                     self.Mega_Explor * F2_Explor + 
                                     lmn_Explore * min_PF_F3 * self.F3_Explore)  # Eq 20
                
                self.best_per_iter.append(self.best['Cost'])
                
                # ✅ Progress reporting
                if it % 10 == 0 or it == 4:
                    improvement = self.last_best - self.best['Cost'] if it > 3 else 0
                    phase = "Explore" if SelectFlag == 1 else "Exploit"
                    patience_info = f" | No improvement:{self.no_improvement_count}/{self.early_stop_patience}" if self.early_stop_patience else ""
                    print(f"Iter {it:4d}/{self.max_iter} | Best:{self.best['Cost']:12.6f} | Δ:{improvement:+.6f} | Phase:{phase}{patience_info}")
                
                # ✅ EARLY STOPPING CHECK 1: Target threshold
                if self.early_stop_threshold is not None and self.best['Cost'] <= self.early_stop_threshold:
                    print("\n" + "="*80)
                    print("🎯 TARGET THRESHOLD REACHED!")
                    print("="*80)
                    print(f"   Objective value:{self.best['Cost']:.6f} <= {self.early_stop_threshold}")
                    print(f"   Stopped at iteration:{it}/{self.max_iter} ({100*it/self.max_iter:.1f}%)")
                    print("="*80)
                    
                    if self.autosave_every_iters:
                        early_stop_path = self.autosave_path.replace(".npz", "_EARLY_STOP_THRESHOLD.npz")
                        self.save_checkpoint(early_stop_path)
                    
                    return {
                        "gbest": self.best['X'],
                        "gbest_val": self.best['Cost'],
                        "best_per_iter": self.best_per_iter,
                        "early_stopped": True,
                        "reason": "threshold_reached",
                        "stopped_at_iter": it
                    }
                
                # ✅ EARLY STOPPING CHECK 2: No improvement
                if abs(self.best['Cost'] - self.last_best) < 1e-8:
                    self.no_improvement_count += 1
                else:
                    self.no_improvement_count = 0
                
                self.last_best = self.best['Cost']
                
                if self.early_stop_patience and self.no_improvement_count >= self.early_stop_patience:
                    print("\n" + "="*80)
                    print("⏸️  NO IMPROVEMENT - EARLY STOPPING")
                    print("="*80)
                    print(f"   No improvement for:{self.early_stop_patience} iterations")
                    print(f"   Best value:{self.best['Cost']:.6f}")
                    print(f"   Stopped at iteration:{it}/{self.max_iter} ({100*it/self.max_iter:.1f}%)")
                    print("="*80)
                    
                    if self.autosave_every_iters:
                        early_stop_path = self.autosave_path.replace(".npz", "_EARLY_STOP_PATIENCE.npz")
                        self.save_checkpoint(early_stop_path)
                    
                    return {
                        "gbest": self.best['X'],
                        "gbest_val": self.best['Cost'],
                        "best_per_iter": self.best_per_iter,
                        "early_stopped": True,
                        "reason": "no_improvement",
                        "stopped_at_iter": it
                    }
                
                # Regular autosave
                if self.autosave_every_iters and (it % self.autosave_every_iters == 0):
                    self.save_checkpoint()
            
            # ✅ Completed all iterations
            print("\n" + "="*80)
            print("✅ OPTIMIZATION COMPLETED - ALL ITERATIONS")
            print("="*80)
            print(f"   Final best value:{self.best['Cost']:.6f}")
            print(f"   Total iterations:{self.max_iter}")
            print("="*80)
            
            if self.autosave_every_iters:
                self.save_checkpoint()
            
            return {
                "gbest": self.best['X'],
                "gbest_val": self.best['Cost'],
                "best_per_iter": self.best_per_iter,
                "early_stopped": False,
                "reason": "completed",
                "stopped_at_iter": self.max_iter
            }
            
        except Exception as e:
            try:
                with open("po_exception.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} exception:{e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass
            try:
                self.save_checkpoint(self.autosave_path.replace(".npz", "_onexception.npz"))
            except Exception:
                pass
            raise


# Convenience function for backwards compatibility
def run_po_with_obj(func, vars_def_or_dim, lb, ub, n_solutions, n_iter,
                     autosave_every=0, autosave_path="po_checkpoint.npz", eval_delay=0.0,
                     early_stop_threshold=None, early_stop_patience=100):
    """
    Helper to call PO directly with a function:
      func(x) -> objective
    vars_def_or_dim: can be integer dim or list-like of variable defs
    """
    if isinstance(vars_def_or_dim, int):
        dim = vars_def_or_dim
    else:
        dim = len(vars_def_or_dim)
    
    po = PO(func, n_solutions, dim, lb, ub, n_iter, 
            autosave_every, autosave_path, eval_delay,
            early_stop_threshold, early_stop_patience)
    return po.optimize()