import pickle
import json
import random
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# Add these at the top of the file
junk_comments = [
    "Nice post bro!", "🔥🔥🔥", "Cool!", "Wow!", "Sub 4 sub?", "Check mine!", "Good", "Awesome bro",
    "Follow me!", "Like my page", "Great stuff", "Amazing!", "Keep it up", "Nice one",
    "Thanks", "Appreciate it", "Well done", "Nice work", "Good job"
]

user_ids = [f"stu_{i:04d}" for i in range(220)]  # Assuming 220 users as in the dataset generation

class FraudDetector:
    def __init__(self, model_path, vectorizer_path, config_path, feature_names_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        with open(feature_names_path, 'rb') as f:
            self.feature_names = pickle.load(f)

    def detect_fraud(self, karma_log):
        # Convert the karma_log to a list of activities if it's not already
        if isinstance(karma_log, dict) and "karma_log" in karma_log:
            activities = karma_log["karma_log"]
        else:
            activities = karma_log

        # Process each activity
        results = []
        for activity in activities:
            features = self._extract_features(activity)
            # Convert features to a DataFrame with the same columns as training
            feature_df = pd.DataFrame([features])
            # One-hot encode categorical features
            feature_df = pd.get_dummies(feature_df, columns=['from_user', 'to_user', 'type', 'post_id'])

            # Ensure all feature columns are present (fill missing with 0)
            for col in self.feature_names:
                if col not in feature_df.columns:
                    feature_df[col] = 0

            # Reorder columns to match training
            feature_df = feature_df[self.feature_names]

            # Convert to numpy array
            feature_vector = feature_df.values[0]

            fraud_score = self.model.predict_proba([feature_vector])[0][1]
            suspicious_activities = self._identify_suspicious_activities(activity, fraud_score)
            results.append({
                "activity_id": activity.get("activity_id", ""),
                "fraud_score": fraud_score,
                "suspicious_activities": suspicious_activities
            })

        # Calculate overall fraud score (average of all activity scores)
        overall_score = sum(result["fraud_score"] for result in results) / len(results) if results else 0

        # Combine all suspicious activities
        all_suspicious = []
        for result in results:
            all_suspicious.extend(result["suspicious_activities"])

        return overall_score, all_suspicious

    def _extract_features(self, activity):
        # Extract features from a single activity
        features = {
            "from_user": activity.get("from_user", ""),
            "to_user": activity.get("to_user", ""),
            "type": activity.get("type", ""),
            "content": activity.get("content", ""),
            "post_id": activity.get("post_id", ""),
            "is_bot": 1 if activity.get("from_user", "").startswith('bot_') else 0,
            "is_mutual": 1 if (activity.get("from_user", ""), activity.get("to_user", "")) in mutual_pairs else 0,
            "hour_of_day": datetime.fromisoformat(activity.get("timestamp", "").replace('Z', '+00:00')).hour if activity.get("timestamp") else 0,
            "day_of_week": datetime.fromisoformat(activity.get("timestamp", "").replace('Z', '+00:00')).weekday() if activity.get("timestamp") else 0,
            "content_length": len(activity.get("content", ""))
        }
        return features

    def _identify_suspicious_activities(self, activity, fraud_score):
        suspicious_activities = []
        if fraud_score > self.config["fraud_threshold"]:
            if activity.get("type") == "upvote_received" and activity.get("from_user", "").startswith('bot_'):
                suspicious_activities.append({
                    "activity_id": activity.get("activity_id", ""),
                    "reason": "mutual upvote",
                    "score": fraud_score
                })
            if activity.get("type") == "comment" and activity.get("content", "") in junk_comments:
                suspicious_activities.append({
                    "activity_id": activity.get("activity_id", ""),
                    "reason": "junk comment",
                    "score": fraud_score
                })
            if len(activity.get("content", "")) < 5:  # Short comments
                suspicious_activities.append({
                    "activity_id": activity.get("activity_id", ""),
                    "reason": "short comment",
                    "score": fraud_score
                })
        return suspicious_activities

# Mutual buddy pairs (should be loaded from the dataset or config)
mutual_pairs = set()
for _ in range(30):  # 30 mutual groups
    u1, u2 = random.sample(user_ids, 2)
    mutual_pairs.add((u1, u2))
    mutual_pairs.add((u2, u1))
