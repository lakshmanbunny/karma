import pickle
import json
import pandas as pd
import random
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, roc_curve
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Import XGBoost and LightGBM
import xgboost as xgb
import lightgbm as lgb

def train_model():
    print("Starting model training...")

    os.makedirs('app/models', exist_ok=True)
    
    data_path = 'data_generation/karma_logs.json'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Please run data_generation/generate_dataset.py first")
        return
    
    with open(data_path, 'r') as f:
        logs = json.load(f)
    
    print(f"Loaded {len(logs)} log entries")
    
    config_path = 'app/config.json'
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    mutual_pairs_path = 'app/models/mutual_pairs.pkl'
    if os.path.exists(mutual_pairs_path):
        with open(mutual_pairs_path, 'rb') as f:
            mutual_pairs = pickle.load(f)
    else:
        user_ids = set()
        for log in logs:
            user_ids.add(log.get("from_user", ""))
            user_ids.add(log.get("to_user", ""))
        user_ids = list(user_ids)
        
        mutual_pairs = set()
        for _ in range(30):
            if len(user_ids) >= 2:
                u1, u2 = random.sample(user_ids, 2)
                mutual_pairs.add((u1, u2))
                mutual_pairs.add((u2, u1))
        
        with open(mutual_pairs_path, 'wb') as f:
            pickle.dump(mutual_pairs, f)
    
    print(f"Using {len(mutual_pairs)} mutual pairs")
    
    data = []
    for log in logs:
        try:
            timestamp = datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00'))
            features = {
                "from_user": log.get("from_user", "unknown"),
                "to_user": log.get("to_user", "unknown"),
                "type": log.get("karma_type", "unknown"),
                "content": log.get("content", ""),
                "post_id": log.get("post_id", "unknown"),
                "is_bot": 1 if log.get("from_user", "").startswith('bot_') else 0,
                "is_mutual": 1 if (log.get("from_user", ""), log.get("to_user", "")) in mutual_pairs else 0,
                "hour_of_day": timestamp.hour,
                "day_of_week": timestamp.weekday(),
                "content_length": len(log.get("content", "")) if log.get("content") else 0,
                "label": log.get("label", "normal")
            }
            data.append(features)
        except Exception as e:
            print(f"Error processing log entry: {e}")
            continue
    
    df = pd.DataFrame(data)
    print(f"Created dataframe with {len(df)} rows")
    
    df['content'] = df['content'].fillna('')
    
    vectorizer = CountVectorizer(max_features=100)
    content_features = vectorizer.fit_transform(df['content'])
    
    content_df = pd.DataFrame(
        content_features.toarray(),
        columns=[f'content_word_{i}' for i in range(content_features.shape[1])]
    )
    
    with open('app/models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    X = df.drop(['label', 'content'], axis=1)
    y = df['label'].apply(lambda x: 1 if x == 'karma-fraud' else 0)
    
    X = pd.get_dummies(X, columns=['from_user', 'to_user', 'type', 'post_id'])
    X = pd.concat([X, content_df], axis=1)
    
    feature_names = X.columns.tolist()
    with open('app/models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    print(f"Saved {len(feature_names)} feature names")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Base models
    rf = RandomForestClassifier(random_state=42)
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    lgbm = lgb.LGBMClassifier(random_state=42)

    # Define models dict for comparison and stacking
    models = {
        "RandomForest": rf,
        "XGBoost": xgb_model,
        "LightGBM": lgbm,
    }

    # Add stacking classifier
    stacking_clf = StackingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgb_model),
            ('lgbm', lgbm)
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        n_jobs=-1,
        passthrough=True
    )
    models["Stacking"] = stacking_clf

    results = {}

    plt.figure(figsize=(8, 6))
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]  # probability for ROC curve
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "f1_score": f1,
            "roc_auc": roc_auc
        }
        
        print(f"{name} evaluation:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Models')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('app/models/roc_curve_comparison.png')
    plt.show()

    # Decide best model based on ROC AUC
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']

    print(f"\nBest model based on ROC AUC: {best_model_name} with ROC AUC = {results[best_model_name]['roc_auc']:.4f}")

    # Save best model
    with open('app/models/model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # Update config
    config["model_version"] = f"1.0.{datetime.now().strftime('%Y%m%d')}"
    config["training_timestamp"] = datetime.now().isoformat()
    config["best_model"] = best_model_name
    config["metrics"] = {
        best_model_name: {
            "accuracy": results[best_model_name]['accuracy'],
            "f1_score": results[best_model_name]['f1_score'],
            "roc_auc": results[best_model_name]['roc_auc']
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print("Best model saved and config updated.")
    print("Training complete!")

if __name__ == "__main__":
    train_model()
