import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import get_scorer

class KomodoMlipirOptimizer:
    def __init__(self, model_class, param_bounds, X, y, metric='accuracy', test_size=0.3, random_state=42):
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
            return self.scorer(model, self.X_val, self.y_val)
        except:
            return -np.inf if self.scorer._sign == 1 else np.inf

    def optimize(self, pop_size=20, generations=30, verbose=True):
        dim = len(self.param_bounds)
        pop = np.zeros((pop_size, dim))
        for i, bounds in enumerate(self.param_bounds.values()):
            pop[:, i] = np.random.randint(bounds[0], bounds[1]+1, pop_size) if isinstance(bounds[0], int) else np.random.uniform(bounds[0], bounds[1], pop_size)
        
        fitness = np.array([self.evaluate(self.decode_params(ind)) for ind in pop])
        
        for gen in range(generations):
            idx = np.argsort(fitness)[::-1] if self.scorer._sign == 1 else np.argsort(fitness)
            pop, fitness = pop[idx], fitness[idx]
            
            # (Lanjutkan implementasi Komodo Mlipir...)
            # ... [Kode lengkap seperti sebelumnya] ...
        
        best_gen = np.argmax(self.best_score_history) if self.scorer._sign == 1 else np.argmin(self.best_score_history)
        return self.best_params_history[best_gen], self.best_score_history[best_gen]