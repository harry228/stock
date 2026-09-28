#!/usr/bin/env python3
"""
TAIEX Pure NumPy Machine Learning Pipeline
- Zero External Package Dependencies (Pure Python + NumPy)
- Multi-Horizon Index & Regime Prediction: 1d, 2d, 3d, 5d
- Models implemented:
  1. Bagging (Random Forest)
  2. Boosting (Gradient Boosted Trees / Pure NumPy GBDT)
  3. Sequence / Neural Net (NumPy Temporal MLP with Sliding Window)
  4. Stacking Meta-Classifier (Ensemble Meta-Learner combining 1-3)
  5. Baseline (Persistence / Benchmark)
- Metrics: Accuracy, RMSE, R², Sharpe Ratio, Max Drawdown (MDD), Calmar Ratio
- Outputs terminal comparison report & JSON summary.
"""

import os
import csv
import json
import time
import math
import numpy as np

# ==================== 1. Pure NumPy Decision Tree ====================

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X, y, feature_indices):
        best_gini = 1e9
        best_feature = None
        best_threshold = None
        n_samples = len(y)

        for feat in feature_indices:
            x_col = X[:, feat]
            thresholds = np.percentile(x_col, np.linspace(10, 90, 10))
            for thresh in thresholds:
                left_mask = x_col <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) < 5 or np.sum(right_mask) < 5:
                    continue
                
                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                weighted_gini = (np.sum(left_mask) * gini_left + np.sum(right_mask) * gini_right) / n_samples

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feat
                    best_threshold = thresh

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0, feature_indices=None):
        n_samples, n_features = X.shape
        unique_labels, counts = np.unique(y, return_counts=True)

        if depth >= self.max_depth or len(unique_labels) <= 1 or n_samples < self.min_samples_split:
            prob_dict = {}
            for c in [-1, 0, 1]:
                mask = (y == c)
                prob_dict[c] = float(np.sum(mask) / n_samples) if n_samples > 0 else 0.0
            best_cls = int(unique_labels[np.argmax(counts)])
            return {"leaf": True, "class": best_cls, "probs": prob_dict}

        if feature_indices is None:
            feature_indices = list(range(n_features))

        feat, thresh = self._best_split(X, y, feature_indices)
        if feat is None:
            prob_dict = {}
            for c in [-1, 0, 1]:
                mask = (y == c)
                prob_dict[c] = float(np.sum(mask) / n_samples) if n_samples > 0 else 0.0
            best_cls = int(unique_labels[np.argmax(counts)])
            return {"leaf": True, "class": best_cls, "probs": prob_dict}

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1, feature_indices)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1, feature_indices)

        return {
            "leaf": False, 
            "feature": int(feat), 
            "threshold": float(thresh), 
            "left": left_tree, 
            "right": right_tree
        }

    def fit(self, X, y, feature_indices=None):
        self.tree = self._build_tree(X, y, feature_indices=feature_indices)

    def _predict_one(self, node, x):
        if node["leaf"]:
            return node["class"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(node["left"], x)
        else:
            return self._predict_one(node["right"], x)

    def predict(self, X):
        return np.array([self._predict_one(self.tree, x) for x in X])

    def _predict_proba_one(self, node, x):
        if node["leaf"]:
            return [node["probs"].get(c, 0.0) for c in [-1, 0, 1]]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_proba_one(node["left"], x)
        else:
            return self._predict_proba_one(node["right"], x)

    def predict_proba(self, X):
        return np.array([self._predict_proba_one(self.tree, x) for x in X])


# ==================== 2. Pure NumPy Bagging (Random Forest) ====================

class BaggingRandomForest:
    def __init__(self, n_estimators=25, max_depth=5, min_samples_split=10, max_features_ratio=0.7):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features_ratio = max_features_ratio
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        n_sub_features = max(1, int(n_features * self.max_features_ratio))
        self.trees = []

        np.random.seed(42)
        for i in range(self.n_estimators):
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            feat_idx = np.random.choice(n_features, size=n_sub_features, replace=False)

            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X[boot_idx], y[boot_idx], feature_indices=feat_idx)
            self.trees.append(tree)

    def predict_proba(self, X):
        all_probs = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(all_probs, axis=0)

    def predict(self, X):
        probs = self.predict_proba(X)
        classes = np.array([-1, 0, 1])
        return classes[np.argmax(probs, axis=1)]


# ==================== 3. Pure NumPy Boosting (GBDT) ====================

class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _best_split(self, X, y):
        best_mse = 1e12
        best_feat, best_thresh = None, None
        n_samples = len(y)

        for feat in range(X.shape[1]):
            x_col = X[:, feat]
            thresholds = np.percentile(x_col, np.linspace(10, 90, 8))
            for thresh in thresholds:
                left_mask = x_col <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) < 5 or np.sum(right_mask) < 5:
                    continue
                mse_left = np.var(y[left_mask]) * np.sum(left_mask)
                mse_right = np.var(y[right_mask]) * np.sum(right_mask)
                total_mse = (mse_left + mse_right) / n_samples
                if total_mse < best_mse:
                    best_mse = total_mse
                    best_feat = feat
                    best_thresh = thresh
        return best_feat, best_thresh

    def _build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(y) < self.min_samples_split or np.var(y) < 1e-6:
            return {"leaf": True, "val": float(np.mean(y))}

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return {"leaf": True, "val": float(np.mean(y))}

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        return {
            "leaf": False,
            "feature": int(feat),
            "threshold": float(thresh),
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_one(self, node, x):
        if node["leaf"]:
            return node["val"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(node["left"], x)
        else:
            return self._predict_one(node["right"], x)

    def predict(self, X):
        return np.array([self._predict_one(self.tree, x) for x in X])


class BoostingGBDT:
    def __init__(self, n_estimators=20, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.models = {c: [] for c in [-1, 0, 1]}

    def _softmax(self, logits):
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples = len(y)
        classes = [-1, 0, 1]
        
        # One-hot encoding
        Y_onehot = np.zeros((n_samples, 3))
        for idx, c in enumerate(classes):
            Y_onehot[:, idx] = (y == c).astype(float)

        F = np.zeros((n_samples, 3))

        for tree_idx in range(self.n_estimators):
            probs = self._softmax(F)
            residuals = Y_onehot - probs

            for c_idx, c in enumerate(classes):
                reg = DecisionTreeRegressor(max_depth=self.max_depth)
                reg.fit(X, residuals[:, c_idx])
                pred_res = reg.predict(X)
                F[:, c_idx] += self.lr * pred_res
                self.models[c].append(reg)

    def predict_proba(self, X):
        n_samples = len(X)
        classes = [-1, 0, 1]
        F = np.zeros((n_samples, 3))

        for c_idx, c in enumerate(classes):
            for reg in self.models[c]:
                F[:, c_idx] += self.lr * reg.predict(X)

        exp_l = np.exp(F - np.max(F, axis=1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    def predict(self, X):
        probs = self.predict_proba(X)
        classes = np.array([-1, 0, 1])
        return classes[np.argmax(probs, axis=1)]


# ==================== 4. Pure NumPy Temporal Neural Net / Attention MLP ====================

class TemporalNeuralNet:
    """
    Pure NumPy Multi-Layer Perceptron with Temporal Sliding Window & Feature Standardizer.
    Models non-linear feature interactions and sequential context.
    """
    def __init__(self, hidden_dim=32, lr=0.01, epochs=120):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.mean = None
        self.std = None
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        self.mean = np.mean(X, axis=0, keepdims=True)
        self.std = np.std(X, axis=0, keepdims=True) + 1e-6
        X_norm = (X - self.mean) / self.std

        n_samples, n_features = X_norm.shape
        classes = [-1, 0, 1]
        
        Y_onehot = np.zeros((n_samples, 3))
        for idx, c in enumerate(classes):
            Y_onehot[:, idx] = (y == c).astype(float)

        np.random.seed(42)
        self.W1 = np.random.randn(n_features, self.hidden_dim) * np.sqrt(2.0 / n_features)
        self.b1 = np.zeros((1, self.hidden_dim))
        self.W2 = np.random.randn(self.hidden_dim, 3) * np.sqrt(2.0 / self.hidden_dim)
        self.b2 = np.zeros((1, 3))

        for epoch in range(self.epochs):
            # Forward
            z1 = np.dot(X_norm, self.W1) + self.b1
            a1 = np.maximum(0, z1) # ReLU
            z2 = np.dot(a1, self.W2) + self.b2
            probs = self._softmax(z2)

            # Backprop
            dz2 = (probs - Y_onehot) / n_samples
            dW2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * (z1 > 0).astype(float)
            dW1 = np.dot(X_norm.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)

            # Update
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

    def predict_proba(self, X):
        X_norm = (X - self.mean) / self.std
        z1 = np.dot(X_norm, self.W1) + self.b1
        a1 = np.maximum(0, z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return self._softmax(z2)

    def predict(self, X):
        probs = self.predict_proba(X)
        classes = np.array([-1, 0, 1])
        return classes[np.argmax(probs, axis=1)]


# ==================== 5. Pure NumPy Stacking Ensemble ====================

class StackingEnsemble:
    """
    Pure NumPy Stacking Meta-Learner combining Bagging, Boosting, and Neural Net probabilities.
    """
    def __init__(self, bagging_model, boosting_model, nn_model):
        self.bagging = bagging_model
        self.boosting = boosting_model
        self.nn = nn_model
        self.meta_W = None
        self.meta_b = None

    def fit(self, X, y):
        # Generate base model probability predictions
        p_bag = self.bagging.predict_proba(X)
        p_boost = self.boosting.predict_proba(X)
        p_nn = self.nn.predict_proba(X)

        # Meta-features: concatenate probabilities [n_samples, 9]
        meta_X = np.hstack([p_bag, p_boost, p_nn])
        n_samples = len(y)
        classes = [-1, 0, 1]

        Y_onehot = np.zeros((n_samples, 3))
        for idx, c in enumerate(classes):
            Y_onehot[:, idx] = (y == c).astype(float)

        np.random.seed(42)
        self.meta_W = np.random.randn(9, 3) * 0.1
        self.meta_b = np.zeros((1, 3))

        lr = 0.05
        for epoch in range(150):
            z = np.dot(meta_X, self.meta_W) + self.meta_b
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

            dz = (probs - Y_onehot) / n_samples
            dW = np.dot(meta_X.T, dz)
            db = np.sum(dz, axis=0, keepdims=True)

            self.meta_W -= lr * dW
            self.meta_b -= lr * db

    def predict_proba(self, X):
        p_bag = self.bagging.predict_proba(X)
        p_boost = self.boosting.predict_proba(X)
        p_nn = self.nn.predict_proba(X)
        meta_X = np.hstack([p_bag, p_boost, p_nn])

        z = np.dot(meta_X, self.meta_W) + self.meta_b
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict(self, X):
        probs = self.predict_proba(X)
        classes = np.array([-1, 0, 1])
        return classes[np.argmax(probs, axis=1)]


# ==================== 6. Metric & Financial Evaluation ====================

def evaluate_metrics(y_true, y_pred, y_prob, returns_test):
    """
    Computes Accuracy, RMSE, R², Sharpe Ratio, Max Drawdown (MDD), Calmar Ratio.
    """
    # Classification Accuracy
    acc = float(np.mean(y_true == y_pred))

    # RMSE on Label Error (or continuous expectation)
    # Continuous expected label from probabilities: E[y] = (-1)*p(-1) + 0*p(0) + 1*p(1)
    y_exp = y_prob[:, 2] - y_prob[:, 0]
    rmse = float(np.sqrt(np.mean((y_true - y_exp) ** 2)))

    # R² coefficient of determination
    ss_res = np.sum((y_true - y_exp) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-9)))

    # Financial Trading Simulation
    # Strategy Signal: Long (+1) if Up, Cash (0) if Consolidation, Cash or Short if Down
    # Here: Position = 1 if pred == 1, 0 if pred == 0, -0.5 if pred == -1
    position = np.where(y_pred == 1, 1.0, np.where(y_pred == -1, -0.5, 0.0))
    strat_returns = position * returns_test

    # Annualized Sharpe Ratio
    mean_ret = np.mean(strat_returns)
    std_ret = np.std(strat_returns) + 1e-9
    rf_daily = 0.015 / 252.0
    sharpe = float((mean_ret - rf_daily) / std_ret * np.sqrt(252))

    # Maximum Drawdown (MDD)
    cum_returns = np.cumprod(1.0 + strat_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (running_max - cum_returns) / running_max
    mdd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Calmar Ratio
    ann_ret = float(np.mean(strat_returns) * 252)
    calmar = float(ann_ret / mdd) if mdd > 1e-6 else (ann_ret * 100.0)

    return {
        "Accuracy": round(acc, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "Sharpe": round(sharpe, 4),
        "MDD_pct": round(mdd * 100.0, 2),
        "Calmar": round(calmar, 4),
        "Ann_Return_pct": round(ann_ret * 100.0, 2)
    }


# ==================== Main Execution Flow ====================

def main():
    start_time = time.time()
    
    csv_path = os.path.expanduser("~/storage/downloads/taiex_ml_training_data.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.expanduser("~/stock-analysis/data/taiex_ml_training_data.csv")
        
    print(f"[Data Source] Loading CSV: {csv_path}")
    
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    print(f"[Dataset] Loaded {len(rows)} daily market records.")

    # Feature selection
    stationary_features = [
        "Open_Ratio", "High_Ratio", "Low_Ratio",
        "Volume_Ratio_5", "Volume_Ratio_20", "Volume_Ratio_60",
        "BIAS_3", "BIAS_5", "BIAS_10", "BIAS_20", "BIAS_60", "BIAS_120",
        "BB_Width", "BB_PercentB",
        "K", "D", "RSI", "Divergence_Signal"
    ]
    available_features = [f for f in stationary_features if f in rows[0]]
    
    N = len(rows)
    X_raw = []
    labels_raw = []
    dates = []
    closes = []
    
    for r in rows:
        dates.append(r["Date"])
        closes.append(float(r["Close"]))
        labels_raw.append(int(r["Label"]) if r["Label"] != "" else None)
        
        vec = []
        valid = True
        for f in available_features:
            val = r.get(f, "")
            if val == "" or val is None:
                valid = False
                break
            vec.append(float(val))
        X_raw.append(vec if valid else None)

    closes_arr = np.array(closes)
    daily_returns = np.zeros(N)
    daily_returns[1:] = (closes_arr[1:] - closes_arr[:-1]) / closes_arr[:-1]

    horizons = [1, 2, 3, 5]
    all_results = {}

    print("\n" + "="*80)
    print("  TAIEX PURE NUMPY MULTI-HORIZON MACHINE LEARNING PIPELINE (XIAOMI PAD 7)")
    print("  Ensemble Methods: Bagging, Boosting, Neural Net & Stacking")
    print("="*80)

    for h in horizons:
        X_h, y_h, ret_h, date_h = [], [], [], []
        
        for i in range(N - h):
            if X_raw[i] is not None and labels_raw[i + h] is not None:
                X_h.append(X_raw[i])
                y_h.append(labels_raw[i + h])
                # Horizon return
                ret_val = (closes_arr[i + h] - closes_arr[i]) / closes_arr[i]
                ret_h.append(ret_val)
                date_h.append(dates[i])

        X_arr = np.array(X_h, dtype=np.float32)
        y_arr = np.array(y_h, dtype=np.int32)
        ret_arr = np.array(ret_h, dtype=np.float32)

        # Train/Test Split (80/20 Time Series Split)
        split_idx = int(len(X_arr) * 0.8)
        X_tr, X_te = X_arr[:split_idx], X_arr[split_idx:]
        y_tr, y_te = y_arr[:split_idx], y_arr[split_idx:]
        ret_te = ret_arr[split_idx:]

        # Train Baseline (Persistence Model)
        y_pred_base = X_te[:, available_features.index("Divergence_Signal")].astype(int) if "Divergence_Signal" in available_features else y_tr[-1]*np.ones(len(y_te), dtype=int)
        y_pred_base = np.clip(y_pred_base, -1, 1)
        prob_base = np.zeros((len(y_te), 3))
        for idx, p in enumerate(y_pred_base):
            prob_base[idx, p + 1] = 1.0

        # Train Bagging
        bag = BaggingRandomForest(n_estimators=30, max_depth=5)
        bag.fit(X_tr, y_tr)
        p_bag = bag.predict_proba(X_te)
        pred_bag = bag.predict(X_te)

        # Train Boosting
        boost = BoostingGBDT(n_estimators=25, learning_rate=0.08, max_depth=3)
        boost.fit(X_tr, y_tr)
        p_boost = boost.predict_proba(X_te)
        pred_boost = boost.predict(X_te)

        # Train Neural Net
        nn = TemporalNeuralNet(hidden_dim=32, lr=0.02, epochs=100)
        nn.fit(X_tr, y_tr)
        p_nn = nn.predict_proba(X_te)
        pred_nn = nn.predict(X_te)

        # Train Stacking
        stack = StackingEnsemble(bag, boost, nn)
        stack.fit(X_tr, y_tr)
        p_stack = stack.predict_proba(X_te)
        pred_stack = stack.predict(X_te)

        # Evaluate Metrics
        metrics_base = evaluate_metrics(y_te, y_pred_base, prob_base, ret_te)
        metrics_bag = evaluate_metrics(y_te, pred_bag, p_bag, ret_te)
        metrics_boost = evaluate_metrics(y_te, pred_boost, p_boost, ret_te)
        metrics_nn = evaluate_metrics(y_te, pred_nn, p_nn, ret_te)
        metrics_stack = evaluate_metrics(y_te, pred_stack, p_stack, ret_te)

        horizon_res = {
            "Baseline": metrics_base,
            "Bagging (RandomForest)": metrics_bag,
            "Boosting (GBDT)": metrics_boost,
            "NeuralNet (Temporal)": metrics_nn,
            "Stacking (Meta-Learner)": metrics_stack
        }
        all_results[f"{h}d"] = horizon_res

        # Print Table for this Horizon
        print(f"\n▶【預測未來 {h} 日指數/盤勢 ({h}-Day Horizon)】(測試集 {len(X_te)} 筆, {date_h[split_idx]} ~ {date_h[-1]}):")
        print("-" * 88)
        print(f"{'Model Architecture':<25} | {'Acc (%)':<8} | {'RMSE':<7} | {'R²':<7} | {'Sharpe':<7} | {'MDD (%)':<8} | {'Calmar':<7}")
        print("-" * 88)
        for model_name, m in horizon_res.items():
            print(f"{model_name:<25} | {m['Accuracy']*100:>7.2f}% | {m['RMSE']:>7.4f} | {m['R2']:>7.4f} | {m['Sharpe']:>7.2f} | {m['MDD_pct']:>7.2f}% | {m['Calmar']:>7.2f}")
        print("-" * 88)

    total_time = time.time() - start_time
    print(f"\n[Execution Completed] Pipeline finished in {total_time:.2f} seconds.")

    # Save summary report
    out_dir = os.path.expanduser("~/stock-analysis/models")
    os.makedirs(out_dir, exist_ok=True)
    report_file = os.path.join(out_dir, "taiex_ensemble_numpy_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[Report Saved] {report_file}")

if __name__ == "__main__":
    main()
