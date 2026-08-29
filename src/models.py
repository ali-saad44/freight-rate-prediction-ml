"""Model implementations and ensemble wrappers."""
import math
import random
from .features import FEATURE_NAMES

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class RidgeRegressorModel:
    """Fast Regularized Ridge Regression Baseline."""
    def __init__(self, alpha=5.0):
        self.alpha = alpha
        self.weights = None
        self.mean_x = None
        self.std_x = None

    def fit(self, X, y):
        n = len(X)
        p = len(X[0])
        p_aug = p + 1
        
        # Calculate feature means and stds for normalization
        self.mean_x = [sum(X[i][j] for i in range(n)) / n for j in range(p)]
        self.std_x = [math.sqrt(sum((X[i][j] - self.mean_x[j]) ** 2 for i in range(n)) / n) + 1e-8 for j in range(p)]
        
        # Single-pass accumulation of XtX and Xty
        XtX = [[0.0] * p_aug for _ in range(p_aug)]
        Xty = [0.0] * p_aug
        
        for i in range(n):
            row = [1.0] + [(X[i][j] - self.mean_x[j]) / self.std_x[j] for j in range(p)]
            y_i = y[i]
            for j in range(p_aug):
                rj = row[j]
                Xty[j] += rj * y_i
                for k in range(j, p_aug):
                    XtX[j][k] += rj * row[k]
                    
        # Fill symmetric lower triangle and add regularization
        for j in range(p_aug):
            for k in range(j):
                XtX[j][k] = XtX[k][j]
            if j > 0:
                XtX[j][j] += self.alpha
                
        # Gaussian elimination solver
        A = [row[:] + [Xty[i]] for i, row in enumerate(XtX)]
        for i in range(p_aug):
            pivot = A[i][i]
            if abs(pivot) < 1e-12:
                pivot = 1e-12
            for k in range(i, p_aug + 1):
                A[i][k] /= pivot
            for j in range(p_aug):
                if j != i:
                    factor = A[j][i]
                    for k in range(i, p_aug + 1):
                        A[j][k] -= factor * A[i][k]
                        
        self.weights = [A[i][p_aug] for i in range(p_aug)]
        return self

    def predict(self, X):
        p = len(self.mean_x)
        preds = []
        for row in X:
            norm_row = [1.0] + [(row[j] - self.mean_x[j]) / self.std_x[j] for j in range(p)]
            pred = sum(norm_row[j] * self.weights[j] for j in range(len(self.weights)))
            preds.append(max(10.0, pred))
        return preds


class FastDecisionTree:
    """O(N log N) Fast regression tree node for GBDT."""
    def __init__(self, max_depth=3, min_samples_split=40):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = 0.0

    def fit(self, X, residuals, depth=0):
        n_samples = len(residuals)
        total_sum = sum(residuals)
        self.value = total_sum / max(1, n_samples)
        
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return
            
        n_features = len(X[0])
        best_score = 0.0
        best_feat = None
        best_thresh = None
        
        sub_features = random.sample(range(n_features), min(6, n_features))
        
        for f_idx in sub_features:
            pairs = sorted((X[i][f_idx], residuals[i]) for i in range(n_samples))
            if pairs[-1][0] - pairs[0][0] < 1e-6:
                continue
                
            left_sum = 0.0
            left_count = 0
            step = max(1, n_samples // 12)
            
            for i in range(n_samples - 1):
                left_sum += pairs[i][1]
                left_count += 1
                
                if left_count % step != 0 or pairs[i][0] == pairs[i+1][0]:
                    continue
                    
                right_count = n_samples - left_count
                if left_count < 10 or right_count < 10:
                    continue
                    
                right_sum = total_sum - left_sum
                score = (left_sum * left_sum) / left_count + (right_sum * right_sum) / right_count
                
                if score > best_score:
                    best_score = score
                    best_feat = f_idx
                    best_thresh = (pairs[i][0] + pairs[i+1][0]) / 2.0
                    
        if best_feat is not None:
            self.feature_idx = best_feat
            self.threshold = best_thresh
            
            left_idx = [i for i in range(n_samples) if X[i][best_feat] <= best_thresh]
            right_idx = [i for i in range(n_samples) if X[i][best_feat] > best_thresh]
            
            if left_idx and right_idx:
                self.left = FastDecisionTree(self.max_depth, self.min_samples_split)
                self.left.fit([X[i] for i in left_idx], [residuals[i] for i in left_idx], depth + 1)
                
                self.right = FastDecisionTree(self.max_depth, self.min_samples_split)
                self.right.fit([X[i] for i in right_idx], [residuals[i] for i in right_idx], depth + 1)

    def predict_one(self, x):
        if self.feature_idx is None or self.left is None or self.right is None:
            return self.value
        if x[self.feature_idx] <= self.threshold:
            return self.left.predict_one(x)
        return self.right.predict_one(x)


class StandaloneGBDTRegressor:
    """Pure Python Gradient Boosted Decision Tree Regressor."""
    def __init__(self, n_estimators=20, learning_rate=0.2, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.base_pred = 0.0
        self.trees = []

    def fit(self, X, y):
        n = len(y)
        self.base_pred = sum(y) / n
        current_preds = [self.base_pred] * n
        self.trees = []
        sample_size = min(2000, n)
        
        for stage in range(self.n_estimators):
            residuals = [y[i] - current_preds[i] for i in range(n)]
            indices = random.sample(range(n), sample_size)
            X_sub = [X[i] for i in indices]
            res_sub = [residuals[i] for i in indices]
            
            tree = FastDecisionTree(max_depth=self.max_depth)
            tree.fit(X_sub, res_sub)
            self.trees.append(tree)
            
            for i in range(n):
                current_preds[i] += self.learning_rate * tree.predict_one(X[i])
                
        return self

    def predict(self, X):
        preds = []
        for x in X:
            val = self.base_pred
            for tree in self.trees:
                val += self.learning_rate * tree.predict_one(x)
            preds.append(max(10.0, val))
        return preds


class LightGBMRegressorModel:
    """LightGBM Wrapper."""
    def __init__(self, n_estimators=400, learning_rate=0.04, num_leaves=31):
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': 42,
            'verbose': -1
        }
        self.model = None
        self.fallback = None

    def fit(self, X, y):
        if HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(X, y)
        else:
            self.fallback = StandaloneGBDTRegressor(n_estimators=20, learning_rate=0.2, max_depth=3)
            self.fallback.fit(X, y)
        return self

    def predict(self, X):
        if HAS_LIGHTGBM and self.model is not None:
            raw_preds = self.model.predict(X)
            return [max(10.0, float(p)) for p in raw_preds]
        return self.fallback.predict(X)


class CatBoostRegressorModel:
    """CatBoost Wrapper."""
    def __init__(self, iterations=400, learning_rate=0.04, depth=6):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.model = None
        self.fallback = None

    def fit(self, X, y):
        if HAS_CATBOOST:
            self.model = cb.CatBoostRegressor(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                verbose=False,
                random_seed=42
            )
            self.model.fit(X, y)
        else:
            self.fallback = RidgeRegressorModel(alpha=10.0)
            self.fallback.fit(X, y)
        return self

    def predict(self, X):
        if HAS_CATBOOST and self.model is not None:
            raw_preds = self.model.predict(X)
            return [max(10.0, float(p)) for p in raw_preds]
        return self.fallback.predict(X)


class EnsembleRegressorModel:
    """Weighted Ensemble."""
    def __init__(self, models=None, weights=None):
        self.models = models or []
        self.weights = weights or []

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X):
        if not self.models:
            return []
        w_sum = sum(self.weights)
        norm_weights = [w / w_sum for w in self.weights]
        
        all_preds = [m.predict(X) for m in self.models]
        n_samples = len(X)
        
        blended = []
        for i in range(n_samples):
            val = sum(norm_weights[m_idx] * all_preds[m_idx][i] for m_idx in range(len(self.models)))
            blended.append(max(10.0, val))
        return blended
