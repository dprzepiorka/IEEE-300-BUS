import numpy as np
import time
import traceback
import os

class GBO:
    """
    Gradient-Based Optimizer (GBO) with early stopping capability.
    Based on:
    Ahmadianfar et al. (2020), "Gradient-based optimizer: A new metaheuristic optimization algorithm"
    DOI: 10.1016/j.ins.2020.06.037
    
    Usage:
        gbo = GBO(func, n_solutions, dim, lb, ub, max_iter,
                  autosave_every_iters=5, autosave_path="gbo_checkpoint.npz", eval_delay=0.0,
                  early_stop_threshold=0.0, early_stop_patience=50)
        res = gbo.optimize()
    Returns dict:{"gbest":..., "gbest_val":..., "best_per_iter":[...], "early_stopped":..., "reason":..., "stopped_at_iter":...}
    """
    def __init__(self, func, n_solutions, dim, lb, ub, max_iter, 
                 pr=0.5,  # Probability of using GSR (Gradient Search Rule)
                 autosave_every_iters=0, autosave_path="gbo_checkpoint.npz", eval_delay=0.0,
                 early_stop_threshold=None, early_stop_patience=100):
        self.func = func
        self.n_solutions = int(n_solutions)
        self.dim = int(dim)
        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)
        self.max_iter = int(max_iter)
        self.pr = float(pr)  # Probability parameter for GSR vs LEO
        
        self.autosave_every_iters = int(autosave_every_iters)
        self.autosave_path = autosave_path
        self.eval_delay = float(eval_delay)
        
        # ✅ EARLY STOPPING parameters
        self.early_stop_threshold = early_stop_threshold
        self.early_stop_patience = early_stop_patience
        self.no_improvement_count = 0
        self.last_best = np.inf
        
        # Initialize population
        self.X = np.random.uniform(self.lb, self.ub, (self.n_solutions, self.dim))
        self.fitness = np.full(self.n_solutions, np.inf)
        
        # Evaluate initial population
        for i in range(self.n_solutions):
            self.fitness[i] = self._eval_solution(self.X[i])
        
        # Find best solution
        best_idx = np.argmin(self.fitness)
        self.best_X = self.X[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]
        
        self.best_per_iter = []
        self.iter = 0
        
        # GBO-specific parameters
        self.beta_min = 0.2
        self.beta_max = 1.2

    def _eval_solution(self, x):
        """Evaluate a solution"""
        try:
            val = float(self.func(x))
        except Exception as e:
            try:
                with open("failed_evals_gbo.csv", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()},{','.join(map(str, x))},exception,{str(e)}\n")
            except Exception:
                pass
            val = np.inf
        if self.eval_delay:
            time.sleep(self.eval_delay)
        return val

    def _gsr(self, x, x_best, x_worst, iteration):
        """
        Gradient Search Rule (GSR) - Main exploration mechanism
        Equation: x_new = x - r * (2*Δ*x - x_best) + α*(x_best - x) + β*(x - x_worst)
        """
        # Calculate normalized iteration
        m = iteration / self.max_iter
        
        # Direction vector (Δ)
        delta = 2 * np.random.rand() * (1 - m)
        
        # Random coefficient
        r = np.random.rand()
        
        # Alpha: decreases from 2 to 0
        alpha = 2 * (1 - m)
        
        # Beta: random in [beta_min, beta_max]
        beta = self.beta_min + (self.beta_max - self.beta_min) * np.random.rand()
        
        # GSR update
        x_new = (x - r * (2 * delta * x - x_best) + 
                 alpha * (x_best - x) + 
                 beta * (x - x_worst))
        
        return x_new

    def _leo(self, x, x_best, iteration):
        """
        Local Escaping Operator (LEO) - Exploitation and local optima avoidance
        Uses multiple search directions
        """
        # Calculate normalized iteration
        m = iteration / self.max_iter
        
        # Step size decreases with iterations
        step_size = (1 - m) * np.random.randn(self.dim)
        
        # Random walk around best solution
        if np.random.rand() < 0.5:
            # Direction 1: Levy flight-inspired
            x_new = x_best + step_size * (x - x_best)
        else:
            # Direction 2: Random walk
            x_new = x + step_size * np.random.randn(self.dim)
        
        # Add mutation with decreasing probability
        if np.random.rand() < 0.5 * (1 - m):
            # Mutation: random dimension gets random value
            mut_dim = np.random.randint(0, self.dim)
            x_new[mut_dim] = self.lb[mut_dim] + np.random.rand() * (self.ub[mut_dim] - self.lb[mut_dim])
        
        return x_new

    def _boundary_check(self, x):
        """Ensure solution is within bounds"""
        return np.clip(x, self.lb, self.ub)

    def save_checkpoint(self, path=None):
        """Save checkpoint"""
        if path is None:
            path = self.autosave_path
        try:
            np.savez(path,
                     X=self.X,
                     fitness=self.fitness,
                     best_X=self.best_X,
                     best_fitness=self.best_fitness,
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
                with open("gbo_save_error.log", "a", encoding="utf-8") as f:
                    f.write(f"{time.time()} save error:{e}\n")
                    f.write(traceback.format_exc())
            except Exception:
                pass

    def optimize(self):
        """Main optimization loop"""
        try:
            # ✅ Print configuration
            print("\n" + "="*80)
            print("🚀 GRADIENT-BASED OPTIMIZER (GBO) - EARLY STOPPING ENABLED")
            print("="*80)
            print(f"📊 Configuration:")
            print(f"   Solutions:{self.n_solutions}")
            print(f"   Max iterations:{self.max_iter}")
            print(f"   Dimensions:{self.dim}")
            print(f"   Probability (pr):{self.pr}")
            if self.early_stop_threshold is not None:
                print(f"   🎯 Target threshold:f <= {self.early_stop_threshold}")
            if self.early_stop_patience:
                print(f"   ⏸️  Patience:{self.early_stop_patience} iterations without improvement")
            print("="*80 + "\n")
            
            print("🔄 Initializing population...")
            self.best_per_iter.append(self.best_fitness)
            self.last_best = self.best_fitness
            print(f"✅ Initialization complete | Initial best:{self.best_fitness:.6f}\n")
            
            # Main optimization loop
            for iteration in range(1, self.max_iter + 1):
                self.iter = iteration
                
                # Find worst solution for GSR
                worst_idx = np.argmax(self.fitness)
                x_worst = self.X[worst_idx].copy()
                
                # Update each solution
                for i in range(self.n_solutions):
                    # Decide: GSR or LEO
                    if np.random.rand() < self.pr:
                        # ===== Gradient Search Rule (GSR) =====
                        x_new = self._gsr(self.X[i], self.best_X, x_worst, iteration)
                    else:
                        # ===== Local Escaping Operator (LEO) =====
                        x_new = self._leo(self.X[i], self.best_X, iteration)
                    
                    # Boundary check
                    x_new = self._boundary_check(x_new)
                    
                    # Evaluate new solution
                    fitness_new = self._eval_solution(x_new)
                    
                    # Greedy selection
                    if fitness_new < self.fitness[i]:
                        self.X[i] = x_new.copy()
                        self.fitness[i] = fitness_new
                        
                        # Update global best
                        if fitness_new < self.best_fitness:
                            self.best_fitness = fitness_new
                            self.best_X = x_new.copy()
                
                self.best_per_iter.append(self.best_fitness)
                
                # ✅ Progress reporting
                if iteration % 10 == 0 or iteration == 1:
                    improvement = self.last_best - self.best_fitness if iteration > 1 else 0
                    patience_info = f" | No improvement:{self.no_improvement_count}/{self.early_stop_patience}" if self.early_stop_patience else ""
                    print(f"Iter {iteration:4d}/{self.max_iter} | Best:{self.best_fitness:12.6f} | Δ:{improvement:+.6f}{patience_info}")
                
                # ✅ EARLY STOPPING CHECK 1: Target threshold
                if self.early_stop_threshold is not None and self.best_fitness <= self.early_stop_threshold:
                    print("\n" + "="*80)
                    print("🎯 TARGET THRESHOLD REACHED!")
                    print("="*80)
                    print(f"   Objective value:{self.best_fitness:.6f} <= {self.early_stop_threshold}")
                    print(f"   Stopped at iteration:{iteration}/{self.max_iter} ({100*iteration/self.max_iter:.1f}%)")
                    print("="*80)
                    
                    if self.autosave_every_iters:
                        early_stop_path = self.autosave_path.replace(".npz", "_EARLY_STOP_THRESHOLD.npz")
                        self.save_checkpoint(early_stop_path)
                    
                    return {
                        "gbest": self.best_X,
                        "gbest_val": self.best_fitness,
                        "best_per_iter": self.best_per_iter,
                        "early_stopped": True,
                        "reason": "threshold_reached",
                        "stopped_at_iter": iteration
                    }
                
                # ✅ EARLY STOPPING CHECK 2: No improvement
                if abs(self.best_fitness - self.last_best) < 1e-8:
                    self.no_improvement_count += 1
                else:
                    self.no_improvement_count = 0
                
                self.last_best = self.best_fitness
                
                if self.early_stop_patience and self.no_improvement_count >= self.early_stop_patience:
                    print("\n" + "="*80)
                    print("⏸️  NO IMPROVEMENT - EARLY STOPPING")
                    print("="*80)
                    print(f"   No improvement for:{self.early_stop_patience} iterations")
                    print(f"   Best value:{self.best_fitness:.6f}")
                    print(f"   Stopped at iteration:{iteration}/{self.max_iter} ({100*iteration/self.max_iter:.1f}%)")
                    print("="*80)
                    
                    if self.autosave_every_iters:
                        early_stop_path = self.autosave_path.replace(".npz", "_EARLY_STOP_PATIENCE.npz")
                        self.save_checkpoint(early_stop_path)
                    
                    return {
                        "gbest": self.best_X,
                        "gbest_val": self.best_fitness,
                        "best_per_iter": self.best_per_iter,
                        "early_stopped": True,
                        "reason": "no_improvement",
                        "stopped_at_iter": iteration
                    }
                
                # Regular autosave
                if self.autosave_every_iters and (iteration % self.autosave_every_iters == 0):
                    self.save_checkpoint()
            
            # ✅ Completed all iterations
            print("\n" + "="*80)
            print("✅ OPTIMIZATION COMPLETED - ALL ITERATIONS")
            print("="*80)
            print(f"   Final best value:{self.best_fitness:.6f}")
            print(f"   Total iterations:{self.max_iter}")
            print("="*80)
            
            if self.autosave_every_iters:
                self.save_checkpoint()
            
            return {
                "gbest": self.best_X,
                "gbest_val": self.best_fitness,
                "best_per_iter": self.best_per_iter,
                "early_stopped": False,
                "reason": "completed",
                "stopped_at_iter": self.max_iter
            }
            
        except Exception as e:
            try:
                with open("gbo_exception.log", "a", encoding="utf-8") as f:
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
def run_gbo_with_obj(func, vars_def_or_dim, lb, ub, n_solutions, n_iter, pr=0.5,
                      autosave_every=0, autosave_path="gbo_checkpoint.npz", eval_delay=0.0,
                      early_stop_threshold=None, early_stop_patience=100):
    """
    Helper to call GBO directly with a function:
      func(x) -> objective
    vars_def_or_dim: can be integer dim or list-like of variable defs
    """
    if isinstance(vars_def_or_dim, int):
        dim = vars_def_or_dim
    else:
        dim = len(vars_def_or_dim)
    
    gbo = GBO(func, n_solutions, dim, lb, ub, n_iter, pr,
              autosave_every, autosave_path, eval_delay,
              early_stop_threshold, early_stop_patience)
    return gbo.optimize()