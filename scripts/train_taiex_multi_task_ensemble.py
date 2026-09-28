#!/usr/bin/env python3
"""
TAIEX Pure NumPy Multi-Task Ensemble ML Pipeline
- Zero External Package Dependencies (Pure Python + NumPy)
- Multi-Horizon Dual-Task Prediction (1d, 3d, 5d):
  1. Classification: Up (+1), Consolidation (0), Down (-1)
  2. Regression: Future Close Price (收盤價數值)
- Models implemented:
  - Bagging (Random Forest Classifier & Regressor)
  - Boosting (GBDT Multi-Class Classifier & Regressor)
  - NeuralNet (Temporal MLP Classifier & Regressor)
  - Stacking (Meta-Learner Ensemble)
- Outputs:
  - Classification: Accuracy, Macro F1-score, 3x3 Confusion Matrix
  - Regression: RMSE, MAE, R²
  - Risk Metrics: Sharpe Ratio, Max Drawdown (MDD), Calmar Ratio
  - Formatted terminal comparison tables
"""

import os
import csv
import json
import time
import numpy as np

# ==================== 1. Classification & Regression Metrics ====================

def compute_classification_metrics(y_true, y_pred):
    labels = [-1, 0, 1]
    acc = float(np.mean(y_true == y_pred))
    
    # 3x3 Confusion Matrix
    # cm[row][col] -> row: Actual, col: Predicted
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        r_idx = labels.index(t)
        c_idx = labels.index(p)
        cm[r_idx, c_idx] += 1
        
    f1_list = []
    for i in range(3):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        
    macro_f1 = float(np.mean(f1_list))
    return {
        "Accuracy": round(acc, 4),
        "Macro_F1": round(macro_f1, 4),
        "Confusion_Matrix": cm.tolist()
    }

def compute_regression_metrics(price_true, price_pred):
    errors = price_pred - price_true
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))
    
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((price_true - np.mean(price_true)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-9)))
    
    return {
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "R2": round(r2, 4)
    }

def compute_risk_metrics(y_pred, returns_test):
    position = np.where(y_pred == 1, 1.0, np.where(y_pred == -1, -0.5, 0.0))
    strat_returns = position * returns_test

    mean_ret = np.mean(strat_returns)
    std_ret = np.std(strat_returns) + 1e-9
    rf_daily = 0.015 / 252.0
    sharpe = float((mean_ret - rf_daily) / std_ret * np.sqrt(252))

    cum_returns = np.cumprod(1.0 + strat_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (running_max - cum_returns) / running_max
    mdd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    ann_ret = float(np.mean(strat_returns) * 252)
    calmar = float(ann_ret / mdd) if mdd > 1e-6 else (ann_ret * 100.0)

    return {
        "Sharpe": round(sharpe, 2),
        "MDD_pct": round(mdd * 100.0, 2),
        "Calmar": round(calmar, 2)
    }


# ==================== 2. Pure NumPy Classifiers & Regressors ====================

class PureTree:
    def __init__(self, max_depth=5, min_samples=10, is_regression=False):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.is_regression = is_regression
        self.tree = None

    def _gini(self, y):
        if len(y) == 0: return 0.0
        _, counts = np.unique(y, return_counts=True)
        return 1.0 - np.sum((counts / len(y)) ** 2)

    def _best_split(self, X, y, feat_indices):
        best_score = 1e12
        best_f, best_t = None, None
        n = len(y)

        for f in feat_indices:
            x_col = X[:, f]
            threshs = np.percentile(x_col, np.linspace(10, 90, 8))
            for t in threshs:
                left = x_col <= t
                right = ~left
                if np.sum(left) < 5 or np.sum(right) < 5: continue

                if self.is_regression:
                    score = (np.var(y[left]) * np.sum(left) + np.var(y[right]) * np.sum(right)) / n
                else:
                    score = (self._gini(y[left]) * np.sum(left) + self._gini(y[right]) * np.sum(right)) / n

                if score < best_score:
                    best_score = score
                    best_f = f
                    best_t = t
        return best_f, best_t

    def _build(self, X, y, depth=0, feat_indices=None):
        n = len(y)
        if feat_indices is None: feat_indices = list(range(X.shape[1]))

        if self.is_regression:
            if depth >= self.max_depth or n < self.min_samples or np.var(y) < 1e-8:
                return {"leaf": True, "val": float(np.mean(y))}
        else:
            u_labels, counts = np.unique(y, return_counts=True)
            if depth >= self.max_depth or len(u_labels) <= 1 or n < self.min_samples:
                probs = {c: float(np.sum(y == c)/n) for c in [-1, 0, 1]}
                best_cls = int(u_labels[np.argmax(counts)])
                return {"leaf": True, "class": best_cls, "probs": probs}

        f, t = self._best_split(X, y, feat_indices)
        if f is None:
            if self.is_regression:
                return {"leaf": True, "val": float(np.mean(y))}
            else:
                u_labels, counts = np.unique(y, return_counts=True)
                probs = {c: float(np.sum(y == c)/n) for c in [-1, 0, 1]}
                best_cls = int(u_labels[np.argmax(counts)])
                return {"leaf": True, "class": best_cls, "probs": probs}

        left_m = X[:, f] <= t
        right_m = ~left_m
        return {
            "leaf": False, "f": int(f), "t": float(t),
            "left": self._build(X[left_m], y[left_m], depth + 1, feat_indices),
            "right": self._build(X[right_m], y[right_m], depth + 1, feat_indices)
        }

    def fit(self, X, y, feat_indices=None):
        self.tree = self._build(X, y, feat_indices=feat_indices)

    def _eval(self, node, x):
        if node["leaf"]: return node
        if x[node["f"]] <= node["t"]:
            return self._eval(node["left"], x)
        else:
            return self._eval(node["right"], x)

    def predict(self, X):
        res = []
        for x in X:
            node = self._eval(self.tree, x)
            res.append(node["val"] if self.is_regression else node["class"])
        return np.array(res)

    def predict_proba(self, X):
        res = []
        for x in X:
            node = self._eval(self.tree, x)
            res.append([node["probs"].get(c, 0.0) for c in [-1, 0, 1]])
        return np.array(res)


class BaggingModel:
    def __init__(self, n_estimators=25, max_depth=5, is_regression=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.is_regression = is_regression
        self.trees = []

    def fit(self, X, y):
        n, m = X.shape
        n_sub = max(1, int(m * 0.7))
        self.trees = []
        np.random.seed(42)
        for _ in range(self.n_estimators):
            b_idx = np.random.choice(n, size=n, replace=True)
            f_idx = np.random.choice(m, size=n_sub, replace=False)
            tree = PureTree(max_depth=self.max_depth, is_regression=self.is_regression)
            tree.fit(X[b_idx], y[b_idx], feat_indices=f_idx)
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([t.predict(X) for t in self.trees])
        if self.is_regression:
            return np.mean(preds, axis=0)
        else:
            probs = self.predict_proba(X)
            return np.array([-1, 0, 1])[np.argmax(probs, axis=1)]

    def predict_proba(self, X):
        probs = np.array([t.predict_proba(X) for t in self.trees])
        return np.mean(probs, axis=0)


class BoostingModel:
    def __init__(self, n_estimators=25, lr=0.08, max_depth=3, is_regression=False):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.is_regression = is_regression
        self.models = [] if is_regression else {c: [] for c in [-1, 0, 1]}

    def fit(self, X, y):
        n = len(y)
        if self.is_regression:
            F = np.mean(y) * np.ones(n)
            self.f0 = float(np.mean(y))
            for _ in range(self.n_estimators):
                residual = y - F
                tree = PureTree(max_depth=self.max_depth, is_regression=True)
                tree.fit(X, residual)
                F += self.lr * tree.predict(X)
                self.models.append(tree)
        else:
            classes = [-1, 0, 1]
            Y_oh = np.zeros((n, 3))
            for i, c in enumerate(classes): Y_oh[:, i] = (y == c).astype(float)
            F = np.zeros((n, 3))

            for _ in range(self.n_estimators):
                exp_F = np.exp(F - np.max(F, axis=1, keepdims=True))
                probs = exp_F / np.sum(exp_F, axis=1, keepdims=True)
                res = Y_oh - probs

                for idx, c in enumerate(classes):
                    tree = PureTree(max_depth=self.max_depth, is_regression=True)
                    tree.fit(X, res[:, idx])
                    F[:, idx] += self.lr * tree.predict(X)
                    self.models[c].append(tree)

    def predict(self, X):
        if self.is_regression:
            F = self.f0 * np.ones(len(X))
            for tree in self.models:
                F += self.lr * tree.predict(X)
            return F
        else:
            probs = self.predict_proba(X)
            return np.array([-1, 0, 1])[np.argmax(probs, axis=1)]

    def predict_proba(self, X):
        F = np.zeros((len(X), 3))
        for idx, c in enumerate([-1, 0, 1]):
            for tree in self.models[c]:
                F[:, idx] += self.lr * tree.predict(X)
        exp_F = np.exp(F - np.max(F, axis=1, keepdims=True))
        return exp_F / np.sum(exp_F, axis=1, keepdims=True)


class NeuralNetModel:
    def __init__(self, hidden=32, lr=0.01, epochs=120, is_regression=False):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.is_regression = is_regression

    def fit(self, X, y):
        self.mean_x = np.mean(X, axis=0, keepdims=True)
        self.std_x = np.std(X, axis=0, keepdims=True) + 1e-6
        X_norm = (X - self.mean_x) / self.std_x
        n, m = X_norm.shape

        np.random.seed(42)
        self.W1 = np.random.randn(m, self.hidden) * np.sqrt(2.0/m)
        self.b1 = np.zeros((1, self.hidden))

        if self.is_regression:
            self.mean_y = float(np.mean(y))
            self.std_y = float(np.std(y)) + 1e-6
            y_norm = (y - self.mean_y) / self.std_y
            Y_target = y_norm.reshape(-1, 1)

            self.W2 = np.random.randn(self.hidden, 1) * np.sqrt(2.0/self.hidden)
            self.b2 = np.zeros((1, 1))

            for _ in range(self.epochs):
                z1 = np.dot(X_norm, self.W1) + self.b1
                a1 = np.maximum(0, z1)
                z2 = np.dot(a1, self.W2) + self.b2

                dz2 = (z2 - Y_target) / n
                dW2 = np.dot(a1.T, dz2)
                db2 = np.sum(dz2, axis=0, keepdims=True)

                da1 = np.dot(dz2, self.W2.T)
                dz1 = da1 * (z1 > 0)
                dW1 = np.dot(X_norm.T, dz1)
                db1 = np.sum(dz1, axis=0, keepdims=True)

                self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        else:
            self.W2 = np.random.randn(self.hidden, 3) * np.sqrt(2.0/self.hidden)
            self.b2 = np.zeros((1, 3))
            Y_oh = np.zeros((n, 3))
            for idx, c in enumerate([-1, 0, 1]): Y_oh[:, idx] = (y == c).astype(float)

            for _ in range(self.epochs):
                z1 = np.dot(X_norm, self.W1) + self.b1
                a1 = np.maximum(0, z1)
                z2 = np.dot(a1, self.W2) + self.b2
                exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
                probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

                dz2 = (probs - Y_oh) / n
                dW2 = np.dot(a1.T, dz2)
                db2 = np.sum(dz2, axis=0, keepdims=True)

                da1 = np.dot(dz2, self.W2.T)
                dz1 = da1 * (z1 > 0)
                dW1 = np.dot(X_norm.T, dz1)
                db1 = np.sum(dz1, axis=0, keepdims=True)

                self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2

    def predict(self, X):
        X_norm = (X - self.mean_x) / self.std_x
        z1 = np.dot(X_norm, self.W1) + self.b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, self.W2) + self.b2
        if self.is_regression:
            return (z2.flatten() * self.std_y) + self.mean_y
        else:
            exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
            probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
            return np.array([-1, 0, 1])[np.argmax(probs, axis=1)]

    def predict_proba(self, X):
        X_norm = (X - self.mean_x) / self.std_x
        z1 = np.dot(X_norm, self.W1) + self.b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, self.W2) + self.b2
        exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class StackingModel:
    def __init__(self, m1, m2, m3, is_regression=False):
        self.m1, self.m2, self.m3 = m1, m2, m3
        self.is_regression = is_regression

    def fit(self, X, y):
        n = len(y)
        if self.is_regression:
            p1 = self.m1.predict(X).reshape(-1, 1)
            p2 = self.m2.predict(X).reshape(-1, 1)
            p3 = self.m3.predict(X).reshape(-1, 1)
            meta_X = np.hstack([p1, p2, p3])
            meta_X_b = np.hstack([meta_X, np.ones((n, 1))])
            self.weights = np.linalg.pinv(meta_X_b.T @ meta_X_b) @ meta_X_b.T @ y
        else:
            p1 = self.m1.predict_proba(X)
            p2 = self.m2.predict_proba(X)
            p3 = self.m3.predict_proba(X)
            meta_X = np.hstack([p1, p2, p3])

            Y_oh = np.zeros((n, 3))
            for idx, c in enumerate([-1, 0, 1]): Y_oh[:, idx] = (y == c).astype(float)

            np.random.seed(42)
            self.W = np.random.randn(9, 3) * 0.1
            self.b = np.zeros((1, 3))
            lr = 0.05
            for _ in range(150):
                z = np.dot(meta_X, self.W) + self.b
                exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)
                dz = (probs - Y_oh) / n
                self.W -= lr * (meta_X.T @ dz)
                self.b -= lr * np.sum(dz, axis=0, keepdims=True)

    def predict(self, X):
        if self.is_regression:
            p1 = self.m1.predict(X).reshape(-1, 1)
            p2 = self.m2.predict(X).reshape(-1, 1)
            p3 = self.m3.predict(X).reshape(-1, 1)
            meta_X = np.hstack([p1, p2, p3])
            meta_X_b = np.hstack([meta_X, np.ones((len(X), 1))])
            return meta_X_b @ self.weights
        else:
            probs = self.predict_proba(X)
            return np.array([-1, 0, 1])[np.argmax(probs, axis=1)]

    def predict_proba(self, X):
        p1 = self.m1.predict_proba(X)
        p2 = self.m2.predict_proba(X)
        p3 = self.m3.predict_proba(X)
        meta_X = np.hstack([p1, p2, p3])
        z = np.dot(meta_X, self.W) + self.b
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


# ==================== Main Runner ====================

def main():
    start_time = time.time()
    csv_path = os.path.expanduser("~/storage/downloads/taiex_ml_training_data.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.expanduser("~/stock-analysis/data/taiex_ml_training_data.csv")

    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader: rows.append(r)

    stationary_features = [
        "Open_Ratio", "High_Ratio", "Low_Ratio",
        "Volume_Ratio_5", "Volume_Ratio_20", "Volume_Ratio_60",
        "BIAS_3", "BIAS_5", "BIAS_10", "BIAS_20", "BIAS_60", "BIAS_120",
        "BB_Width", "BB_PercentB",
        "K", "D", "RSI", "Divergence_Signal"
    ]
    available_features = [f for f in stationary_features if f in rows[0]]

    N = len(rows)
    X_raw, labels_raw, closes_raw, dates = [], [], [], []
    for r in rows:
        dates.append(r["Date"])
        closes_raw.append(float(r["Close"]))
        labels_raw.append(int(r["Label"]) if r["Label"] != "" else None)
        vec = []
        valid = True
        for f in available_features:
            val = r.get(f, "")
            if val == "" or val is None: valid = False; break
            vec.append(float(val))
        X_raw.append(vec if valid else None)

    closes_arr = np.array(closes_raw)
    horizons = [1, 2, 3, 4, 5]
    all_res = {}

    print("\n" + "="*88)
    print("      TAIEX MULTI-TASK ENSEMBLE PIPELINE (CLASSIFICATION + PRICE REGRESSION)")
    print("="*88)

    for h in horizons:
        X_h, y_cls_h, y_reg_ratio_h, target_close_h, base_close_h, ret_h, date_h = [], [], [], [], [], [], []
        for i in range(N - h):
            if X_raw[i] is not None and labels_raw[i + h] is not None:
                X_h.append(X_raw[i])
                y_cls_h.append(labels_raw[i + h])
                c_now = closes_arr[i]
                c_future = closes_arr[i + h]
                y_reg_ratio_h.append(c_future / c_now)
                target_close_h.append(c_future)
                base_close_h.append(c_now)
                ret_h.append((c_future - c_now) / c_now)
                date_h.append(dates[i])

        X_arr = np.array(X_h, dtype=np.float32)
        y_cls_arr = np.array(y_cls_h, dtype=np.int32)
        y_reg_ratio_arr = np.array(y_reg_ratio_h, dtype=np.float32)
        target_close_arr = np.array(target_close_h, dtype=np.float32)
        base_close_arr = np.array(base_close_h, dtype=np.float32)
        ret_arr = np.array(ret_h, dtype=np.float32)

        split_idx = int(len(X_arr) * 0.8)
        X_tr, X_te = X_arr[:split_idx], X_arr[split_idx:]
        y_cls_tr, y_cls_te = y_cls_arr[:split_idx], y_cls_arr[split_idx:]
        y_reg_tr, y_reg_te = y_reg_ratio_arr[:split_idx], y_reg_ratio_arr[split_idx:]
        
        target_close_te = target_close_arr[split_idx:]
        base_close_te = base_close_arr[split_idx:]
        ret_te = ret_arr[split_idx:]

        models_names = ["Bagging (RandomForest)", "Boosting (GBDT)", "NeuralNet (Temporal MLP)", "Stacking (Meta-Ensemble)"]
        horizon_results = {}

        # Classification Models
        c_bag = BaggingModel(n_estimators=30, max_depth=5, is_regression=False)
        c_bag.fit(X_tr, y_cls_tr)

        c_boost = BoostingModel(n_estimators=25, lr=0.08, max_depth=3, is_regression=False)
        c_boost.fit(X_tr, y_cls_tr)

        c_nn = NeuralNetModel(hidden=32, lr=0.02, epochs=100, is_regression=False)
        c_nn.fit(X_tr, y_cls_tr)

        c_stack = StackingModel(c_bag, c_boost, c_nn, is_regression=False)
        c_stack.fit(X_tr, y_cls_tr)

        cls_models = [c_bag, c_boost, c_nn, c_stack]

        # Regression Models
        r_bag = BaggingModel(n_estimators=30, max_depth=5, is_regression=True)
        r_bag.fit(X_tr, y_reg_tr)

        r_boost = BoostingModel(n_estimators=25, lr=0.08, max_depth=3, is_regression=True)
        r_boost.fit(X_tr, y_reg_tr)

        r_nn = NeuralNetModel(hidden=32, lr=0.01, epochs=120, is_regression=True)
        r_nn.fit(X_tr, y_reg_tr)

        r_stack = StackingModel(r_bag, r_boost, r_nn, is_regression=True)
        r_stack.fit(X_tr, y_reg_tr)

        reg_models = [r_bag, r_boost, r_nn, r_stack]

        print(f"\n▶【預測未來 {h} 日 (Horizon = {h}d)】(測試集 {len(X_te)} 筆, {date_h[split_idx]} ~ {date_h[-1]}):")
        print("="*100)
        print(f"{'Model Name':<24} | {'Acc (%)':<7} {'F1-macro':<8} | {'RMSE (點)':<10} {'MAE (點)':<9} {'R²':<7} | {'Sharpe':<6} {'MDD (%)':<8} {'Calmar':<6}")
        print("-" * 100)

        for name, cm_mod, rm_mod in zip(models_names, cls_models, reg_models):
            # Classification
            pred_cls = cm_mod.predict(X_te)
            cls_m = compute_classification_metrics(y_cls_te, pred_cls)

            # Regression
            pred_ratio = rm_mod.predict(X_te)
            pred_close_price = base_close_te * pred_ratio
            reg_m = compute_regression_metrics(target_close_te, pred_close_price)

            # Risk
            risk_m = compute_risk_metrics(pred_cls, ret_te)

            horizon_results[name] = {
                "Classification": cls_m,
                "Regression": reg_m,
                "Risk": risk_m
            }

            print(f"{name:<24} | {cls_m['Accuracy']*100:>6.2f}% {cls_m['Macro_F1']:>8.4f} | {reg_m['RMSE']:>10.2f} {reg_m['MAE']:>9.2f} {reg_m['R2']:>7.4f} | {risk_m['Sharpe']:>6.2f} {risk_m['MDD_pct']:>7.2f}% {risk_m['Calmar']:>6.2f}")

        print("\n  [分類任務 3x3 混淆矩陣 (Rows: Actual [-1, 0, 1], Cols: Predicted [-1, 0, 1])]")
        for name in models_names:
            cm = horizon_results[name]["Classification"]["Confusion_Matrix"]
            print(f"   - {name:<24}: 下跌[-1]={cm[0]}, 盤整[0]={cm[1]}, 上漲[1]={cm[2]}")

        all_res[f"{h}d"] = horizon_results

    total_time = time.time() - start_time
    print("\n" + "="*100)
    print(f" [Pipeline Executed Successfully] Total Time: {total_time:.2f} seconds.")
    print("="*100)

    out_file = os.path.expanduser("~/stock-analysis/models/taiex_multi_task_ensemble_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_res, f, ensure_ascii=False, indent=2)
    print(f"[Report JSON Saved] {out_file}\n")

if __name__ == "__main__":
    main()
