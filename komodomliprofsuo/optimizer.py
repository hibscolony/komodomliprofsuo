import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer
import warnings
from functools import lru_cache

class KomodoMlipirOptimizer:
    def __init__(self, model_class, param_bounds, X_train, y_train, X_test, y_test, metric='accuracy'):
        """
        Args:
            model_class: Class model (e.g., XGBClassifier).
            param_bounds: Dict of parameter bounds (e.g., {'max_depth': (3, 10)}).
            X_train, y_train: Training data.
            X_val, y_val: Validation data (fixed).
            metric: Scoring metric (default: 'accuracy').
        """
        self.model_class = model_class
        self.param_bounds = param_bounds
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.metric = metric
        self.scorer = get_scorer(metric)
        self.best_params_history = []
        self.best_score_history = []
        
        # Untuk avoid warning saat training gagal
        warnings.filterwarnings("ignore", category=UserWarning)

    def decode_params(self, vec):
        """Convert real-valued vector to model parameters."""
        params = {}
        for i, (name, bounds) in enumerate(self.param_bounds.items()):
            val = np.clip(vec[i], bounds[0], bounds[1])
            params[name] = int(round(val)) if isinstance(bounds[0], int) else float(val)
        return params

    @lru_cache(maxsize=None)
    def evaluate(self, params_tuple):
        """
        Evaluate model with caching. 
        params_tuple: Tuple version of params (hashable untuk lru_cache).
        """
        params = self.decode_params(np.array(params_tuple))
        try:
            model = self.model_class(**params)
            model.fit(self.X_train, self.y_train)
            score = self.scorer(model, self.X_test, self.y_test)
            return score if np.isfinite(score) else (-np.inf if self.scorer._sign == 1 else np.inf)
        except Exception as e:
            print(f"Error with params {params}: {str(e)}")
            return -np.inf if self.scorer._sign == 1 else np.inf

    def optimize(self, pop_size=20, generations=30, mutation_rate=0.1, verbose=True, random_state=42):
        """Run optimization with fixed random seed."""
        np.random.seed(random_state)
        dim = len(self.param_bounds)
        
        # Initialize population
        pop = np.zeros((pop_size, dim))
        for i, bounds in enumerate(self.param_bounds.values()):
            if isinstance(bounds[0], int):
                pop[:, i] = np.random.randint(bounds[0], bounds[1] + 1, pop_size)
            else:
                pop[:, i] = np.random.uniform(bounds[0], bounds[1], pop_size)
        
        # Evaluate initial population (convert params to tuples for caching)
        fitness = np.array([self.evaluate(tuple(ind)) for ind in pop])
        
        for gen in range(generations):
            # Sort by fitness
            idx = np.argsort(fitness)[::-1] if self.scorer._sign == 1 else np.argsort(fitness)
            pop, fitness = pop[idx], fitness[idx]
            
            # Track best
            best_params = self.decode_params(pop[0])
            best_score = fitness[0]
            self.best_params_history.append(best_params)
            self.best_score_history.append(best_score)
            
            if verbose:
                print(f"Gen {gen+1:03d} | Best {self.metric}: {best_score:.4f} | Params: {best_params}")
                
            # Selection: Top 50%
            survivors = pop[:pop_size // 2]
            
            # Mutation
            children = []
            for ind in survivors:
                child = ind.copy()
                for j in range(dim):
                    if np.random.rand() < mutation_rate:
                        bounds = list(self.param_bounds.values())[j]
                        if isinstance(bounds[0], int):
                            child[j] = np.random.randint(bounds[0], bounds[1] + 1)
                        else:
                            child[j] = np.random.uniform(bounds[0], bounds[1])
                children.append(child)
            
            # New generation
            pop = np.vstack([survivors, np.array(children)[:pop_size // 2]])
            fitness = np.array([self.evaluate(tuple(ind)) for ind in pop])
        
        # Return overall best
        best_idx = np.argmax(self.best_score_history) if self.scorer._sign == 1 else np.argmin(self.best_score_history)
        return self.best_params_history[best_idx], self.best_score_history[best_idx]
