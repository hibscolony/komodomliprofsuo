
# Komodo Mlipir Optimizer (KMA)

**Komodo Mlipir Algorithm (KMA)** adalah algoritma optimasi berbasis perilaku alami komodo,  
yang membagi populasi ke dalam tiga kelompok: **Big Male**, **Female**, dan **Small Male**.  
Setiap kelompok bergerak dan bereproduksi berdasarkan strategi yang meniru adaptasi alami komodo.

Algoritma ini dapat digunakan untuk **optimasi hyperparameter** atau **fungsi objektif** lainnya.

---

## ✨ Fitur

- Evolusi populasi dengan **pembagian peran** (Big Male, Female, Small Male).
- **High Exploitation - Low Exploration (HILE)** untuk Big Males.
- **Mating or Parthenogenesis (MIME)** untuk Female.
- **Low Intensity High Exploration (LIHE)** untuk Small Males.
- **Adaptasi ukuran populasi** (mengecil/membesar) secara otomatis.

---

## 🚀 Cara Pakai

### 1. Inisialisasi Optimizer

```python
# from optimizer import KomodoMlipirOptimizer
from sklearn.datasets import load_iris
from xgboost import XGBClassifier

# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

# Definisikan parameter space
param_bounds = {
    'max_depth': (2, 10),
    'learning_rate': (0.01, 0.3),
    'n_estimators': (50, 200)
}

# Buat optimizer
optimizer = KomodoMlipirOptimizer(
    model_class=XGBClassifier,
    param_bounds=param_bounds,
    X_train = X_train,
    y_train = y_train,
    X_val = X_test,
    y_val = y_test,
    metric='accuracy'
)
```

### 2. Jalankan Optimasi

```python
best_params, best_score = optimizer.optimize(
    pop_size=50,
    generations=30,
    p=0.5,   # Proporsi Big Male
    d=0.5,   # Mlipir rate
    adapt_population=True,
    verbose=True
)
```

---

## 📖 Referensi Resmi

Komodo Mlipir Algorithm diadopsi berdasarkan publikasi ilmiah berikut:

> Suyanto, R., Huda, M., & Suhartono, D. (2022).  
> Komodo mlipir algorithm: A new metaheuristic optimization algorithm.  
> *Applied Soft Computing, 114*, 108055.  
> https://doi.org/10.1016/j.asoc.2021.108055

Pastikan untuk **mengutip paper ini** jika menggunakan algoritma ini dalam penelitian atau proyek akademik.

---

## ⚖️ Lisensi

Proyek ini dilisensikan di bawah [Apache License 2.0](LICENSE).

Copyright (c) 2025  
Koding Muda Nusantara
