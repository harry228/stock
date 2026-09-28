#!/usr/bin/env python3
"""
TAIEX Multi-Horizon ML Model Training Script (Tablet Optimized)
Zero-dependency (Pure Python + NumPy).
Trains 1-day, 3-day, and 5-day market regime classification models on Xiaomi Pad 7 (Termux).
"""

import os
import csv
import json
import time
from datetime import datetime
import numpy as np

# ==================== Pure NumPy Decision Tree & Random Forest ====================

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
        best_gini = 1.0
        best_feature = None
        best_threshold = None
        n_samples = len(y)

        for feat in feature_indices:
            x_col = X[:, feat]
            # Select percentile candidates for speed
            thresholds = np.percentile(x_col, np.linspace(10, 90, 12))
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
        num_labels = len(np.unique(y))

        if depth >= self.max_depth or num_labels <= 1 or n_samples < self.min_samples_split:
            classes, counts = np.unique(y, return_counts=True)
            return {
                "leaf": True, 
                "class": int(classes[np.argmax(counts)]), 
                "probs": {int(c): float(cnt/n_samples) for c, cnt in zip(classes, counts)}
            }

        if feature_indices is None:
            feature_indices = list(range(n_features))

        feat, thresh = self._best_split(X, y, feature_indices)
        if feat is None:
            classes, counts = np.unique(y, return_counts=True)
            return {
                "leaf": True, 
                "class": int(classes[np.argmax(counts)]), 
                "probs": {int(c): float(cnt/n_samples) for c, cnt in zip(classes, counts)}
            }

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

    def _predict_proba_one(self, node, x, all_classes):
        if node["leaf"]:
            return [node["probs"].get(c, 0.0) for c in all_classes]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_proba_one(node["left"], x, all_classes)
        else:
            return self._predict_proba_one(node["right"], x, all_classes)

    def predict_proba(self, X, all_classes=[-1, 0, 1]):
        return np.array([self._predict_proba_one(self.tree, x, all_classes) for x in X])


class RandomForest:
    def __init__(self, n_estimators=35, max_depth=5, min_samples_split=10, max_features_ratio=0.7):
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

    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        final_preds = []
        for i in range(X.shape[0]):
            classes, counts = np.unique(all_preds[:, i], return_counts=True)
            final_preds.append(classes[np.argmax(counts)])
        return np.array(final_preds)

    def predict_proba(self, X, all_classes=[-1, 0, 1]):
        all_probs = np.array([tree.predict_proba(X, all_classes) for tree in self.trees])
        return np.mean(all_probs, axis=0)

# ==================== Main Script Logic ====================

def main():
    start_time = time.time()
    
    csv_path = os.path.expanduser("~/storage/downloads/taiex_ml_training_data.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.expanduser("~/stock-analysis/data/taiex_ml_training_data.csv")
        
    print(f"Loading CSV dataset: {csv_path}")
    
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
            
    print(f"Total Rows Loaded: {len(rows)}")
    
    # 1. Define Stationary Features
    stationary_features = [
        "Open_Ratio", "High_Ratio", "Low_Ratio",
        "Volume_Ratio_5", "Volume_Ratio_20", "Volume_Ratio_60",
        "BIAS_3", "BIAS_5", "BIAS_10", "BIAS_20", "BIAS_60", "BIAS_120",
        "BB_Width", "BB_PercentB",
        "K", "D", "RSI", "Divergence_Signal"
    ]
    
    # Ensure features exist in CSV
    available_features = [f for f in stationary_features if f in rows[0]]
    print(f"Selected {len(available_features)} Stationary Features:")
    for idx, f in enumerate(available_features):
        print(f"  {idx+1:2d}. {f}")

    # Prepare dataset arrays
    # Extract feature matrix X and targets
    N = len(rows)
    X_raw = []
    labels_raw = []
    dates = []
    
    for r in rows:
        dates.append(r["Date"])
        labels_raw.append(int(r["Label"]) if r["Label"] != "" else None)
        
        feat_vec = []
        valid = True
        for f in available_features:
            val_str = r.get(f, "")
            if val_str == "" or val_str is None:
                valid = False
                break
            feat_vec.append(float(val_str))
            
        if valid:
            X_raw.append(feat_vec)
        else:
            X_raw.append(None)
            
    # Function to train and evaluate for target horizon
    results = {}
    
    horizons = [1, 2, 3, 4, 5]
    label_map = {1: "上漲 (Up)", 0: "盤整 (Consolidation)", -1: "下跌 (Down)"}
    
    print("\n" + "="*60)
    print("      TAIEX ML MULTI-HORIZON TRAINING (XIAOMI PAD 7)")
    print("="*60)
    
    latest_row_idx = N - 1
    latest_date = dates[latest_row_idx]
    
    for h in horizons:
        # Construct dataset for horizon h:
        # Target for row i is label at row i + h
        X_h = []
        y_h = []
        date_h = []
        
        for i in range(N - h):
            if X_raw[i] is not None and labels_raw[i + h] is not None:
                X_h.append(X_raw[i])
                y_h.append(labels_raw[i + h])
                date_h.append(dates[i])
                
        X_arr = np.array(X_h, dtype=np.float32)
        y_arr = np.array(y_h, dtype=np.int32)
        
        # Chronological Split (80% Train, 20% Test)
        split_idx = int(len(X_arr) * 0.8)
        X_train, X_test = X_arr[:split_idx], X_arr[split_idx:]
        y_train, y_test = y_arr[:split_idx], y_arr[split_idx:]
        
        t0 = time.time()
        rf = RandomForest(n_estimators=30, max_depth=5, min_samples_split=10)
        rf.fit(X_train, y_train)
        train_time = time.time() - t0
        
        # Test Set Evaluation
        preds = rf.predict(X_test)
        acc = np.mean(preds == y_test)
        
        # Calculate per-class accuracy / precision
        class_acc = {}
        for c in [-1, 0, 1]:
            mask = (y_test == c)
            if np.sum(mask) > 0:
                class_acc[label_map[c]] = float(np.mean(preds[mask] == c))
            else:
                class_acc[label_map[c]] = 0.0
                
        # Latest Inference for Horizon h
        # Latest feature vector (row N - 1)
        latest_x = np.array([X_raw[latest_row_idx]], dtype=np.float32)
        latest_pred = int(rf.predict(latest_x)[0])
        latest_probs = rf.predict_proba(latest_x, all_classes=[-1, 0, 1])[0]
        
        prob_dict = {
            "Down (-1)": round(float(latest_probs[0]), 4),
            "Consolidation (0)": round(float(latest_probs[1]), 4),
            "Up (1)": round(float(latest_probs[2]), 4)
        }
        
        results[f"{h}d"] = {
            "horizon": f"{h} 天後",
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "test_accuracy": round(float(acc), 4),
            "train_time_sec": round(float(train_time), 3),
            "per_class_accuracy": class_acc,
            "latest_prediction": label_map[latest_pred],
            "latest_confidence": prob_dict
        }
        
        print(f"\n▶【預測未來 {h} 日盤勢 ({h}-Day Horizon)】:")
        print(f"  - 訓練樣本數 : {len(X_train)} 筆 ({date_h[0]} ~ {date_h[split_idx-1]})")
        print(f"  - 測試樣本數 : {len(X_test)} 筆 ({date_h[split_idx]} ~ {date_h[-1]})")
        print(f"  - 訓練耗時   : {train_time:.3f} 秒")
        print(f"  - 測試集準確率: {acc * 100:.2f}%")
        print(f"  - 最新 ({latest_date}) 推理未來 {h} 天預測: 【 {label_map[latest_pred]} 】")
        print("    信心度機率:")
        for state, prob in prob_dict.items():
            bar = "█" * int(prob * 20)
            print(f"      - {state:20s}: {prob*100:6.2f}%  {bar}")

    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"  平板總訓練與推理完成！總耗時: {total_time:.2f} 秒")
    print("="*60)
    
    # Save training report to models directory
    model_dir = os.path.expanduser("~/stock-analysis/models")
    os.makedirs(model_dir, exist_ok=True)
    report_path = os.path.join(model_dir, "taiex_tablet_models_report.json")
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"訓練模型報告已寫入: {report_path}\n")

if __name__ == "__main__":
    main()
