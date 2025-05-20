from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
import json
import os
import numpy as np
from app.fraud_detector import FraudDetector

app = FastAPI()

class Activity(BaseModel):
    activity_id: str
    type: str
    from_user: str
    timestamp: str
    source: str = None
    post_id: str = None
    content: str = None

class KarmaLog(BaseModel):
    user_id: str
    karma_log: List[Activity]

class FraudResponse(BaseModel):
    user_id: str
    fraud_score: float
    suspicious_activities: List[Dict[str, Any]]
    status: str

# Initialize FraudDetector
fraud_detector = None
try:
    with open('app/config.json', 'r') as f:
        config = json.load(f)
    fraud_detector = FraudDetector(
        model_path='app/models/model.pkl',
        vectorizer_path='app/models/vectorizer.pkl',
        config_path='app/config.json',
        feature_names_path='app/models/feature_names.pkl'
    )
except Exception as e:
    print(f"Error initializing FraudDetector: {e}")

@app.post("/analyze", response_model=FraudResponse)
async def analyze_karma(log: KarmaLog):
    if fraud_detector is None:
        raise HTTPException(status_code=500, detail="Fraud detector not initialized")

    # Convert the log to a dictionary
    log_dict = log.dict()

    # Add the user_id as to_user for each activity
    for activity in log_dict["karma_log"]:
        activity["to_user"] = log_dict["user_id"]

    try:
        fraud_score, suspicious_activities = fraud_detector.detect_fraud(log_dict)
        status = "flagged" if fraud_score > config["fraud_threshold"] else "clean"

        return {
            "user_id": log.user_id,
            "fraud_score": float(fraud_score),
            "suspicious_activities": suspicious_activities,
            "status": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting fraud: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/version")
async def version():
    return {"version": "1.0"}
