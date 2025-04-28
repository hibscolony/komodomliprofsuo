import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer

class KomodoMlipirOptimizer:
    def __init__(self, model_class, param_bounds, X, y, metric='accuracy', test_size=0.2, random_state=42):
        self.model_class = model_class
        self.param_bounds = param_bounds
        self.metric = metric
        self.scorer = get_scorer(metric)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        self.best_params_history = []
        self.best_score_history = []

    def decode_params(self, vec):
        params = {}
        for i, (name, bounds) in enumerate(self.param_bounds.items()):
            val = np.clip(vec[i], bounds[0], bounds[1])
            params[name] = int(round(val)) if isinstance(bounds[0], int) else float(val)
        return params

    def evaluate(self, params):
        try:
            model = self.model_class(**params)
            model.fit(self.X_train, self.y_train)
            score = self.scorer(model, self.X_val, self.y_val)
    
            if not np.isfinite(score):  # Kalau NaN atau inf
                return -np.inf if self.scorer._sign == 1 else np.inf
            
            return score
        except Exception as e:
            print(f"Evaluation error with params {params}: {e}")
            return -np.inf if self.scorer._sign == 1 else np.inf
    

    def optimize(self, pop_size=20, generations=30, mutation_rate=0.1, verbose=True):
        dim = len(self.param_bounds)
        
        # Initialize population
        pop = np.zeros((pop_size, dim))
        for i, bounds in enumerate(self.param_bounds.values()):
            if isinstance(bounds[0], int):
                pop[:, i] = np.random.randint(bounds[0], bounds[1] + 1, pop_size)
            else:
                pop[:, i] = np.random.uniform(bounds[0], bounds[1], pop_size)
    
        # Evaluate initial fitness
        fitness = np.array([self.evaluate(self.decode_params(ind)) for ind in pop])
    
        for gen in range(generations):
            # Sort population by fitness
            idx = np.argsort(fitness)[::-1] if self.scorer._sign == 1 else np.argsort(fitness)
            pop, fitness = pop[idx], fitness[idx]
    
            # Record best
            best_params = self.decode_params(pop[0])
            best_score = fitness[0]
            self.best_params_history.append(best_params)
            self.best_score_history.append(best_score)
    
            if verbose:
                scores_list = np.round(fitness, 4).tolist()
                print(f"Generation {gen+1} - Scores: {scores_list} - Best Score: {best_score:.4f}")
    
            # Selection: Keep top half
            survivors = pop[:pop_size // 2]
    
            # Mutation: Create new individuals by mutating survivors
            children = []
            for individual in survivors:
                child = individual.copy()
                for j in range(dim):
                    if np.random.rand() < mutation_rate:
                        bounds = list(self.param_bounds.values())[j]
                        if isinstance(bounds[0], int):
                            child[j] = np.random.randint(bounds[0], bounds[1] + 1)
                        else:
                            child[j] = np.random.uniform(bounds[0], bounds[1])
                children.append(child)
    
            children = np.array(children)
    
            # Combine survivors and new children to form next generation
            if len(children) < pop_size // 2:
                extra = survivors[:(pop_size // 2 - len(children))]
                pop = np.vstack([survivors, children, extra])
            else:
                pop = np.vstack([survivors, children[:pop_size // 2])
    
            # Evaluate fitness again
            fitness = np.array([self.evaluate(self.decode_params(ind)) for ind in pop])
    
        # Final best
        best_gen = np.argmax(self.best_score_history) if self.scorer._sign == 1 else np.argmin(self.best_score_history)
        return self.best_params_history[best_gen], self.best_score_history[best_gen]
