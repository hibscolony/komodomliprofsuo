import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer
from functools import lru_cache
import warnings
from typing import Dict, Tuple, Union, List, Callable

class KomodoMlipirOptimizer:
    def __init__(self, 
                 model_class: Callable,
                 param_bounds: Dict[str, Tuple[Union[int, float], Union[int, float]]],
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 metric: str = 'accuracy',
                 random_state: int = 42):
        """
        Enhanced Komodo Mlipir Algorithm for hyperparameter optimization.
        
        Args:
            model_class: The model class to optimize (e.g., XGBClassifier)
            param_bounds: Dictionary of parameter bounds {param_name: (min, max)}
            X_train, y_train: Training data
            X_val, y_val: Validation data (fixed)
            metric: Scoring metric (default: 'accuracy')
            random_state: Random seed for reproducibility
        """
        self.model_class = model_class
        self.param_bounds = param_bounds
        self.metric = metric
        self.scorer = get_scorer(metric)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.random_state = random_state
        self.best_params_history = []
        self.best_score_history = []
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
        warnings.filterwarnings("ignore", category=UserWarning)

    def decode_params(self, vec: np.ndarray) -> Dict:
        """Convert real-valued vector to model parameters with proper types."""
        params = {}
        for i, (name, bounds) in enumerate(self.param_bounds.items()):
            val = np.clip(vec[i], bounds[0], bounds[1])
            params[name] = int(round(val)) if isinstance(bounds[0], int) else float(val)
        return params

    @lru_cache(maxsize=None)
    def _evaluate_cached(self, params_tuple: Tuple) -> float:
        """Cached evaluation function using hashable parameters."""
        params = self.decode_params(np.array(params_tuple))
        try:
            model = self.model_class(**params)
            model.fit(self.X_train, self.y_train)
            score = self.scorer(model, self.X_val, self.y_val)
            return score if np.isfinite(score) else (-np.inf if self.scorer._sign == 1 else np.inf)
        except Exception as e:
            if hasattr(self, 'verbose') and self.verbose:
                print(f"Evaluation error with params {params}: {str(e)}")
            return -np.inf if self.scorer._sign == 1 else np.inf

    def evaluate(self, params: np.ndarray) -> float:
        """Evaluate parameters with caching."""
        return self._evaluate_cached(tuple(params))

    def _initialize_population(self, pop_size: int) -> np.ndarray:
        """Initialize population within parameter bounds."""
        dim = len(self.param_bounds)
        pop = np.zeros((pop_size, dim))
        for i, bounds in enumerate(self.param_bounds.values()):
            if isinstance(bounds[0], int):
                pop[:, i] = np.random.randint(bounds[0], bounds[1] + 1, pop_size)
            else:
                pop[:, i] = np.random.uniform(bounds[0], bounds[1], pop_size)
        return pop

    def _move_big_males(self, big: np.ndarray, fitness_big: np.ndarray) -> np.ndarray:
        """Update big males position based on fitness and interactions."""
        q = len(big)
        new_big = big.copy()
        for i in range(q):
            move = np.zeros_like(big[i])
            for j in range(q):
                if i == j:
                    continue
                r1 = np.random.rand()
                r2 = np.random.rand()
                if (self.scorer._sign == 1 and fitness_big[j] > fitness_big[i]) or \
                   (self.scorer._sign == -1 and fitness_big[j] < fitness_big[i]) or \
                   r2 < 0.5:
                    move += r1 * (big[j] - big[i])
                else:
                    move += r1 * (big[i] - big[j])
            new_big[i] = big[i] + move
        return new_big

    def _mate_or_parthenogenesis(self, female: np.ndarray, winner_big: np.ndarray) -> np.ndarray:
        """Update female through mating or parthenogenesis."""
        if np.random.rand() < 0.5:  # Mating
            r = np.random.rand(female.shape[0])
            offspring1 = r * female + (1 - r) * winner_big
            offspring2 = r * winner_big + (1 - r) * female
            return offspring1 if np.random.rand() < 0.5 else offspring2
        else:  # Parthenogenesis
            r = np.random.randn(female.shape[0])
            alpha = 0.1
            bounds = np.array(list(self.param_bounds.values()))
            ranges = bounds[:,1] - bounds[:,0]
            return female + (2 * r - 1) * alpha * ranges

    def _mlipir_small_males(self, small: np.ndarray, big: np.ndarray, d: float) -> np.ndarray:
        """Update small males through mlipir movement."""
        q = len(big)
        new_small = small.copy()
        for i in range(len(small)):
            move = np.zeros_like(small[i])
            for j in range(q):
                r1 = np.random.rand()
                r2 = np.random.rand()
                dim_choice = (r2 < d)
                if dim_choice:
                    move += r1 * (big[j] - small[i])
            new_small[i] = small[i] + move
        return new_small

    def _clip_to_bounds(self, population: np.ndarray) -> np.ndarray:
        """Ensure all parameters stay within defined bounds."""
        for i, bounds in enumerate(self.param_bounds.values()):
            population[:, i] = np.clip(population[:, i], bounds[0], bounds[1])
        return population

    def optimize(self, 
                pop_size: int = 200, 
                generations: int = 30, 
                p: float = 0.5, 
                d: float = 0.5, 
                adapt_population: bool = True,
                min_pop: int = 20,
                max_pop: int = 200,
                change_amount: int = 5,
                verbose: bool = True) -> Tuple[Dict, float]:
        """
        Run the Komodo Mlipir optimization algorithm.
        
        Args:
            pop_size: Initial population size
            generations: Number of generations
            p: Proportion of big males
            d: Small males movement probability
            adapt_population: Whether to adapt population size
            min_pop: Minimum population size
            max_pop: Maximum population size
            change_amount: Population size change amount
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best_parameters, best_score)
        """
        self.verbose = verbose
        dim = len(self.param_bounds)
        n = pop_size
        
        # Initialize population
        pop = self._initialize_population(n)
        fitness = np.array([self.evaluate(ind) for ind in pop])

        global_best_score = -np.inf if self.scorer._sign == 1 else np.inf
        global_best_params = None

        for gen in range(generations):
            # Sort population by fitness
            idx = np.argsort(fitness)[::-1] if self.scorer._sign == 1 else np.argsort(fitness)
            pop, fitness = pop[idx], fitness[idx]

            # Classify individuals
            q = max(2, int((p - 1) * n))  # Number of big males
            s = n - q - 1  # Number of small males

            big = pop[:q]
            fitness_big = fitness[:q]
            female = pop[q]
            fitness_female = fitness[q]
            small = pop[q+1:]

            # 1. Move Big Males (exploitation)
            big = self._move_big_males(big, fitness_big)

            # 2. Update Female (mating or parthenogenesis)
            winner_big = big[np.argmax(fitness_big) if self.scorer._sign == 1 else np.argmin(fitness_big)]
            female = self._mate_or_parthenogenesis(female, winner_big)

            # 3. Move Small Males (exploration)
            small = self._mlipir_small_males(small, big, d)

            # 4. Clip all parameters to bounds
            big = self._clip_to_bounds(big)
            female = self._clip_to_bounds(female[np.newaxis, :])[0]
            small = self._clip_to_bounds(small)

            # 5. Combine new population
            pop = np.vstack((big, [female], small))
            fitness = np.array([self.evaluate(ind) for ind in pop])

            # Track best solution
            best_idx = np.argmax(fitness) if self.scorer._sign == 1 else np.argmin(fitness)
            current_best_score = fitness[best_idx]
            current_best_params = self.decode_params(pop[best_idx])

            # Update global best
            if (self.scorer._sign == 1 and current_best_score > global_best_score) or \
               (self.scorer._sign == -1 and current_best_score < global_best_score):
                global_best_score = current_best_score
                global_best_params = current_best_params

            self.best_params_history.append(global_best_params.copy())
            self.best_score_history.append(global_best_score)

            # 6. Adaptive population size
            if adapt_population and gen >= 2:
                delta_f1 = abs(self.best_score_history[-1] - self.best_score_history[-2])
                delta_f2 = abs(self.best_score_history[-2] - self.best_score_history[-3])
                if delta_f1 == 0 and delta_f2 == 0:  # Stagnation - increase diversity
                    n = min(max_pop, n + change_amount)
                elif delta_f1 > 0 and delta_f2 > 0:  # Improving - focus search
                    n = max(min_pop, n - change_amount)

            # Progress reporting
            if verbose:
                print(f"Gen {gen+1:03d}/{generations} | "
                      f"Best {self.metric}: {global_best_score:.4f} | "
                      f"Pop size: {n} | "
                      f"Best Global Params: {global_best_params} ")

        return global_best_params, global_best_score
