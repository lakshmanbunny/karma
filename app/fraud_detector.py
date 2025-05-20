import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
import os

class FraudDetector:
    def __init__(self, model_path, config_path, feature_names_path):
        """
        Initialize the fraud detector with model and configuration
        
        Args:
            model_path: Path to the trained model pickle file
            config_path: Path to the configuration JSON file
            feature_names_path: Path to the feature names pickle file
        """
        # Load the model if it exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        # Load the configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # Load feature names
        if not os.path.exists(feature_names_path):
            raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")
            
        with open(feature_names_path, 'rb') as f:
            self.feature_names = pickle.load(f)
            
        # Get junk comments from config
        self.junk_comments = self.config.get("junk_comments", [])
        
        # Load mutual pairs if available or initialize empty set
        self.mutual_pairs_path = os.path.join(os.path.dirname(model_path), "mutual_pairs.pkl")
        if os.path.exists(self.mutual_pairs_path):
            with open(self.mutual_pairs_path, 'rb') as f:
                self.mutual_pairs = pickle.load(f)
        else:
            self.mutual_pairs = set()

    def detect_fraud(self, karma_log):
        """
        Detect fraudulent activities in the karma log
        
        Args:
            karma_log: Dictionary containing user_id and list of activities or list of activities
            
        Returns:
            Tuple of (overall_fraud_score, list_of_suspicious_activities)
        """
        # Convert the karma_log to a list of activities if it's not already
        if isinstance(karma_log, dict) and "karma_log" in karma_log:
            activities = karma_log["karma_log"]
        else:
            activities = karma_log
            
        # Handle empty activities list
        if not activities:
            return 0.0, []

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
                    
            # Make sure we only include columns present in training
            for col in list(feature_df.columns):
                if col not in self.feature_names:
                    feature_df.drop(col, axis=1, inplace=True)

            # Reorder columns to match training
            feature_df = feature_df.reindex(columns=self.feature_names, fill_value=0)

            # Make prediction
            if len(feature_df) > 0:
                # Convert to numpy array and predict
                feature_vector = feature_df.values
                fraud_score = self.model.predict_proba(feature_vector)[0][1]
            else:
                fraud_score = 0.0
                
            # Find suspicious activities
            suspicious_activities = self._identify_suspicious_activities(activity, fraud_score)
            
            results.append({
                "activity_id": activity.get("activity_id", ""),
                "fraud_score": float(fraud_score),  # Convert numpy float to Python float
                "suspicious_activities": suspicious_activities
            })

        # Calculate overall fraud score (average of all activity scores)
        overall_score = sum(result["fraud_score"] for result in results) / len(results) if results else 0.0

        # Combine all suspicious activities
        all_suspicious = []
        for result in results:
            all_suspicious.extend(result["suspicious_activities"])

        return float(overall_score), all_suspicious

    def _extract_features(self, activity):
        """Extract features from a single activity"""
        # Parse timestamp safely
        timestamp = None
        if activity.get("timestamp"):
            try:
                timestamp = datetime.fromisoformat(activity["timestamp"].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                timestamp = datetime.now()  # Default to current time on error
        else:
            timestamp = datetime.now()
            
        # Extract features
        features = {
            "from_user": activity.get("from_user", "unknown"),
            "to_user": activity.get("to_user", "unknown"),
            "type": activity.get("type", "unknown"),
            "content": activity.get("content", ""),
            "post_id": activity.get("post_id", "unknown"),
            "is_bot": 1 if activity.get("from_user", "").startswith('bot_') else 0,
            "is_mutual": 1 if (activity.get("from_user", ""), activity.get("to_user", "")) in self.mutual_pairs else 0,
            "hour_of_day": timestamp.hour,
            "day_of_week": timestamp.weekday(),
            "content_length": len(activity.get("content", "")) if activity.get("content") else 0
        }
        return features

    def _identify_suspicious_activities(self, activity, fraud_score):
        """Identify reasons why an activity might be fraudulent"""
        suspicious_activities = []
        
        if fraud_score > self.config.get("fraud_threshold", 0.7):
            # Check for bot upvotes
            if activity.get("type") == "upvote_received" and activity.get("from_user", "").startswith('bot_'):
                suspicious_activities.append({
                    "activity_id": activity.get("activity_id", ""),
                    "reason": "bot upvote",
                    "score": float(fraud_score)
                })
                
            # Check for mutual upvotes
            if activity.get("type") == "upvote_received" and (activity.get("from_user", ""), activity.get("to_user", "")) in self.mutual_pairs:
                suspicious_activities.append({
                    "activity_id": activity.get("activity_id", ""),
                    "reason": "mutual upvote",
                    "score": float(fraud_score)
                })
                
            # Check for junk comments
            if activity.get("type") == "comment" and activity.get("content") in self.junk_comments:
                suspicious_activities.append({
                    "activity_id": activity.get("activity_id", ""),
                    "reason": "junk comment",
                    "score": float(fraud_score)
                })
                
            # Check for short comments
            if activity.get("type") == "comment" and len(activity.get("content", "")) < 5:
                suspicious_activities.append({
                    "activity_id": activity.get("activity_id", ""),
                    "reason": "short comment",
                    "score": float(fraud_score)
                })
                
        return suspicious_activities