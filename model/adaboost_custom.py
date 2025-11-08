import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier

#preprocessing
class ExpectedColumns(BaseEstimator, TransformerMixin):
    """
    Valida que las columnas esperadas estén presentes en el DataFrame.
    """
    def __init__(self, expected_cols):
        self.expected_cols = expected_cols

    def fit(self, X, y=None):
        missing_cols = [col for col in self.expected_cols if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas esperadas: {missing_cols}")
        return self

    def transform(self, X):
        return X[self.expected_cols]
    
    def __str__(self):
        return (f"ExpectedColumns(\n"
                f"  Expected columns: {self.expected_cols},\n"
                f"  Number of columns: {len(self.expected_cols)}\n"
                f")")


class TitanicCategoriesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, categories_features):
        self.categories_features = categories_features
        self.categories_transformers = [
            OneHotEncoder(handle_unknown='error', sparse_output=False, drop='first') for _ in categories_features
        ]

    def fit(self, X, y=None):
        # fit each encoder with a 2D DataFrame (single column)
        for col, transformer in zip(self.categories_features, self.categories_transformers):
            transformer.fit(X[[col]])
        return self

    def transform(self, X):
        X_copy = X.copy()
        encoded_dfs = []
        for col, transformer in zip(self.categories_features, self.categories_transformers):
            # transform using a 2D DataFrame and build a DataFrame with proper column names
            encoded = transformer.transform(X_copy[[col]])
            # obtener nombres de columnas coherentes con la salida del encoder
            try:
                # scikit-learn >= 1.0
                col_names = transformer.get_feature_names_out([col])
            except Exception:
                # fallback: construir nombres a partir de categories_ respetando 'drop'
                cats = transformer.categories_[0]
                drop = getattr(transformer, "drop", None)
                if drop == "first":
                    cats_used = cats[1:]
                elif isinstance(drop, (list, tuple, np.ndarray)):
                    drop_idx = np.asarray(drop)
                    cats_used = [c for i, c in enumerate(cats) if i not in drop_idx]
                else:
                    cats_used = cats
                col_names = [f"{col}_{c}" for c in cats_used]
            encoded_df = pd.DataFrame(encoded, columns=col_names, index=X_copy.index)
            encoded_dfs.append(encoded_df)
            # drop the original categorical column
            X_copy = X_copy.drop(columns=[col])
        if encoded_dfs:
            X_copy = pd.concat([X_copy] + encoded_dfs, axis=1)
        return X_copy

    def __str__(self):
        info = "TitanicCategoriesTransformer(\n"
        info += f"  Categorical features: {self.categories_features},\n"
        info += f"  Encoder: OneHotEncoder(handle_unknown='error', sparse_output=False, drop='first'),\n"
        
        if hasattr(self.categories_transformers[0], 'categories_'):
            info += "  Fitted categories per feature:\n"
            for col, transformer in zip(self.categories_features, self.categories_transformers):
                if hasattr(transformer, 'categories_'):
                    cats = transformer.categories_[0]
                    info += f"    {col}: {list(cats)}\n"
        else:
            info += "  Status: Not fitted yet\n"
        
        info += ")"
        return info

class TitanicOutliersTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        if "age" in X_copy.columns:
            X_copy["age"] = np.where(X_copy["age"] > 100, np.nan, X_copy["age"])
        if "fare" in X_copy.columns:
            Q1, Q3 = X_copy["fare"].quantile(0.25), X_copy["fare"].quantile(0.75)
            IQR = Q3 - Q1
            upper = Q3 + 1.5 * IQR
            X_copy["fare"] = np.where(X_copy["fare"] > upper, upper, X_copy["fare"])
        return X_copy
    
    def __str__(self):
        return (f"TitanicOutliersTransformer(\n"
                f"  Age: Remove values > 100 (set to NaN),\n"
                f"  Fare: Cap at Q3 + 1.5*IQR (calculated on transform)\n"
                f")")



class TitanicScalingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features):
        self.numerical_features = numerical_features
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.numerical_features])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.numerical_features] = self.scaler.transform(X_copy[self.numerical_features])
        return X_copy

    def __str__(self):
        return (f"TitanicScalingTransformer(\n"
                f"  Numerical features: {self.numerical_features},\n"
                f"  Scaler: {self.scaler.__class__.__name__},\n"
                f"  Mean: {self.scaler.mean_ if hasattr(self.scaler, 'mean_') else 'Not fitted'},\n"
                f"  Scale: {self.scaler.scale_ if hasattr(self.scaler, 'scale_') else 'Not fitted'}\n"
                f")")


class TitanicImputationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, numerical_features, categorical_features):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.imputer_num = SimpleImputer(strategy="median")
        self.imputer_cat = SimpleImputer(strategy="most_frequent")

    def fit(self, X, y=None):
        self.imputer_num.fit(X[self.numerical_features])
        self.imputer_cat.fit(X[self.categorical_features])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.numerical_features] = self.imputer_num.transform(X_copy[self.numerical_features])
        X_copy[self.categorical_features] = self.imputer_cat.transform(X_copy[self.categorical_features])
        return X_copy

    def __str__(self):
        return (f"TitanicImputationTransformer(\n"
                f"  Numerical features: {self.numerical_features},\n"
                f"  Numerical strategy: {self.imputer_num.strategy},\n"
                f"  Categorical features: {self.categorical_features},\n"
                f"  Categorical strategy: {self.imputer_cat.strategy}\n"
                f")")



class TitanicPipeline(BaseEstimator, TransformerMixin):
    """
        Hace todo el pipeline para procesar los datos del dataset
    """
    def __init__(self):
        self.pipeline = None
        self.transformTarget = {0:-1,1:1}
        self.inverseTransformTarget = {-1:0,1:1}

    def fit(self, X, y=None):
        expectedColumnsStep = ExpectedColumns(
            ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
        )
        numFeatures = ["age", "fare", "sibsp", "parch"]
        categoriesFeatures = ["pclass", "sex", "embarked"]

        titanicCategoricalTransformer = TitanicCategoriesTransformer(categoriesFeatures)
        titanicOutliersTransformer = TitanicOutliersTransformer()
        titanicScalingTransformer = TitanicScalingTransformer(numFeatures)
        titanicImputationTransformer = TitanicImputationTransformer(numFeatures, categoriesFeatures)
        self.pipeline = Pipeline(
            steps=[
                ("expected_columns", expectedColumnsStep),
                ("outliers", titanicOutliersTransformer),
                ("imputation", titanicImputationTransformer),
                ("categorical", titanicCategoricalTransformer),
                ("scaling", titanicScalingTransformer),
            ]
        )
        self.pipeline.fit(X)
        return self

    def transform(self, X, y=None):
        if self.pipeline:
            return self.pipeline.transform(X)
        raise ValueError("Pipeline is not fitted yet.")

    def __str__(self) -> str:
        """
        Returns a string representation of the TitanicPipeline showing its configuration.
        """
        info = "TitanicPipeline(\n"
        info += f"  Target transform: {self.transformTarget},\n"
        info += f"  Inverse target transform: {self.inverseTransformTarget},\n"
        
        if self.pipeline is not None:
            info += "  Pipeline steps:\n"
            for name, step in self.pipeline.steps:
                info += f"    - {name}:\n"
                step_str = str(step)
                for line in step_str.split('\n'):
                    info += f"      {line}\n"
        else:
            info += "  Pipeline: Not fitted yet\n"
        
        info += ")"
        return info


#model
class SimpleAdaBoost:
    def __init__(self, base_estimator=None, n_estimators=50, verbose=False, seed=0):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator or DecisionTreeClassifier(max_depth=1, random_state=seed)
        self.alphas = []
        self.models = []
        self.errors = []
        self.verbose = verbose
        self.seed = seed

    def fit(self, X, y):
        n_samples = X.shape[0]
        D = np.ones(n_samples) / n_samples

        for t in range(self.n_estimators):
            h_t = self.base_estimator.__class__(**self.base_estimator.get_params())
            h_t.fit(X, y, sample_weight=D)

            y_pred = h_t.predict(X)
            err_t = np.sum(D * (y_pred != y)) / np.sum(D)
            err_t = max(err_t, 1e-10)  # evita dividir entre cero

            alpha_t = 0.5 * np.log((1 - err_t) / err_t)

            # Actualización de pesos
            D *= np.exp(-alpha_t * y * y_pred)
            D /= np.sum(D)

            # Guardar resultados
            self.models.append(h_t)
            self.alphas.append(alpha_t)
            self.errors.append(err_t)

            if self.verbose:
                print(f"Iter {t+1}/{self.n_estimators}: ε={err_t:.4f}, α={alpha_t:.4f}")

        return self

    def predict(self, X, return_scores=False):
        pred = np.zeros(X.shape[0])
        for alpha, model in zip(self.alphas, self.models):
            pred += alpha * model.predict(X)

        if return_scores:
            return pred

        y_pred = np.sign(pred)
        y_pred[y_pred == 0] = 1
        y_pred = (y_pred + 1) / 2
        return y_pred.astype(int)
