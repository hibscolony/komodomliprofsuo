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
            if not np.isfinite(score):
                return -np.inf if self.scorer._sign == 1 else np.inf
            return score
        except Exception as e:
            print(f"Evaluation error with params {params}: {e}")
            return -np.inf if self.scorer._sign == 1 else np.inf

    def move_big_males(self, big, fitness_big):
        q = len(big)
        new_big = big.copy()
        for i in range(q):
            move = np.zeros_like(big[i])
            for j in range(q):
                if i == j:
                    continue
                r1 = np.random.rand()
                r2 = np.random.rand()
                if fitness_big[j] > fitness_big[i] or r2 < 0.5:
                    move += r1 * (big[j] - big[i])
                else:
                    move += r1 * (big[i] - big[j])
            new_big[i] = big[i] + move
        return new_big

    def mate_or_parthenogenesis(self, female, winner_big):
        if np.random.rand() < 0.5:
            r = np.random.rand(female.shape[0])
            offspring1 = r * female + (1 - r) * winner_big
            offspring2 = r * winner_big + (1 - r) * female
            return offspring1 if np.random.rand() < 0.5 else offspring2
        else:
            r = np.random.randn(female.shape[0])
            alpha = 0.1
            bounds = np.array(list(self.param_bounds.values()))
            ranges = bounds[:,1] - bounds[:,0]
            return female + (2 * r - 1) * alpha * ranges

    def mlipir_small_males(self, small, big, d):
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

    def optimize(self, pop_size=200, generations=30, p=0.5, d=0.5, adapt_population=True, verbose=True):
        dim = len(self.param_bounds)
        n = pop_size
        min_pop = 20
        max_pop = 200
        change_amount = 5

        pop = np.zeros((n, dim))
        for i, bounds in enumerate(self.param_bounds.values()):
            if isinstance(bounds[0], int):
                pop[:, i] = np.random.randint(bounds[0], bounds[1] + 1, n)
            else:
                pop[:, i] = np.random.uniform(bounds[0], bounds[1], n)

        fitness = np.array([self.evaluate(self.decode_params(ind)) for ind in pop])

        f1, f2, f3 = 0, 0, 0

        for gen in range(generations):
            idx = np.argsort(fitness)[::-1] if self.scorer._sign == 1 else np.argsort(fitness)
            pop, fitness = pop[idx], fitness[idx]

            q = max(2, int((p - 1) * n))
            s = n - q - 1

            big = pop[:q]
            fitness_big = fitness[:q]
            female = pop[q]
            fitness_female = fitness[q]
            small = pop[q+1:]

            # Move Big Males
            big = self.move_big_males(big, fitness_big)

            # Update Female
            winner_big = big[np.argmax(fitness_big) if self.scorer._sign == 1 else np.argmin(fitness_big)]
            female = self.mate_or_parthenogenesis(female, winner_big)

            # Move Small Males
            small = self.mlipir_small_males(small, big, d)

            # Clip to bounds
            for group in (big, small, female[np.newaxis, :]):
                for i, (low, high) in enumerate(self.param_bounds.values()):
                    group[:, i] = np.clip(group[:, i], low, high) if isinstance(group, np.ndarray) else np.clip(group[i], low, high)

            # Combine new population
            pop = np.vstack((big, [female], small))
            fitness = np.array([self.evaluate(self.decode_params(ind)) for ind in pop])

            # Update best history
            best_idx = np.argmax(fitness) if self.scorer._sign == 1 else np.argmin(fitness)
            best_params = self.decode_params(pop[best_idx])
            best_score = fitness[best_idx]
            self.best_params_history.append(best_params)
            self.best_score_history.append(best_score)

            if verbose:
                print(f"Gen {gen+1:03d} | Best {self.metric}: {best_score:.4f} | Params: {best_params}")
                
            # Adapt population size
            if adapt_population and gen >= 2:
                delta_f1 = abs(self.best_score_history[-1] - self.best_score_history[-2])
                delta_f2 = abs(self.best_score_history[-2] - self.best_score_history[-3])
                if delta_f1 == 0 and delta_f2 == 0:
                    n = min(max_pop, n + change_amount)
                elif delta_f1 > 0 and delta_f2 > 0:
                    n = max(min_pop, n - change_amount)

        # Return best overall
        best_gen_idx = np.argmax(self.best_score_history) if self.scorer._sign == 1 else np.argmin(self.best_score_history)
        return self.best_params_history[best_gen_idx], self.best_score_history[best_gen_idx]
