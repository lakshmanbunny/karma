import pickle
import json
import pandas as pd
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from datetime import datetime

# Define mutual_pairs at the top of the file
user_ids = [f"stu_{i:04d}" for i in range(220)]  # Assuming 220 users as in the dataset generation
mutual_pairs = set()
for _ in range(30):  # 30 mutual groups
    u1, u2 = random.sample(user_ids, 2)
    mutual_pairs.add((u1, u2))
    mutual_pairs.add((u2, u1))

# Load the dataset
with open('../data_generation/karma_logs.json', 'r') as f:
    logs = json.load(f)

# Preprocess the dataset
data = []
for log in logs:
    timestamp = datetime.fromisoformat(log["timestamp"].replace('Z', '+00:00'))
    features = {
        "from_user": log["from_user"],
        "to_user": log["to_user"],
        "type": log["karma_type"],
        "content": log["content"] if log["content"] else "",
        "post_id": log["post_id"] if log["post_id"] else "",
        "is_bot": 1 if log["from_user"].startswith('bot_') else 0,
        "is_mutual": 1 if (log["from_user"], log["to_user"]) in mutual_pairs else 0,
        "hour_of_day": timestamp.hour,
        "day_of_week": timestamp.weekday(),
        "content_length": len(log["content"]) if log["content"] else 0,
        "label": log["label"]
    }
    data.append(features)

df = pd.DataFrame(data)

# Prepare features and labels
X = df.drop('label', axis=1)
y = df['label'].apply(lambda x: 1 if x == 'karma-fraud' else 0)

# Convert categorical features to numerical
X = pd.get_dummies(X, columns=['from_user', 'to_user', 'type', 'post_id'])

# Save the feature names for later use
feature_names = X.columns.tolist()
with open('../app/models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")

# Save the model
with open('../app/models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
