
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer



class SimpleAdaBoost:
    def __init__(self, n_estimators=50, base_estimator=None):
        self.n_estimators = int(n_estimators)
        self.base_estimator = base_estimator if base_estimator is not None else DecisionTreeClassifier(max_depth=1)
        # almacenamientos
        self.estimators_: List[DecisionTreeClassifier] = []
        self.alphas_: List[float] = []
        self.errors_: List[float] = []
        self.sample_weights_history_: List[np.ndarray] = []

    def fit(self, X, y):
        # y se espera en {-1, +1}
        n_samples = X.shape[0]
        # distribución inicial uniforme
        w = np.ones(n_samples) / n_samples
        self.sample_weights_history_.append(w.copy())

        for t in range(self.n_estimators):
            # Clonar el estimador base
            stump = DecisionTreeClassifier(max_depth=1, random_state=42+t)
            stump.fit(X, y, sample_weight=w)

            # Predicciones en {-1, +1}
            pred = stump.predict(X)
            # Error ponderado
            err = np.sum(w * (pred != y)) / np.sum(w)

            # Evitar divisiones por cero o alfa infinitos
            err = np.clip(err, 1e-12, 1 - 1e-12)

            alpha = 0.5 * np.log((1 - err) / err)

            # Actualizar pesos
            w = w * np.exp(-alpha * y * pred)
            # Normalizar
            w = w / np.sum(w)

            # Guardar
            self.estimators_.append(stump)
            self.alphas_.append(alpha)
            self.errors_.append(err)
            self.sample_weights_history_.append(w.copy())

        return self

    def decision_function(self, X):
        # suma ponderada de clasificadores
        if not self.estimators_:
            raise RuntimeError("El modelo no está entrenado.")
        f = np.zeros(X.shape[0])
        for alpha, stump in zip(self.alphas_, self.estimators_):
            f += alpha * stump.predict(X)
        return f

    def predict(self, X):
        f = self.decision_function(X)
        return np.where(f >= 0, 1, -1)