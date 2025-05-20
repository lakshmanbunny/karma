from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import os
from app.fraud_detector import FraudDetector

app = FastAPI(title="Karma Fraud Detector API")

class Activity(BaseModel):
    activity_id: str
    type: str
    from_user: str
    timestamp: str
    to_user: Optional[str] = None
    source: Optional[str] = None
    post_id: Optional[str] = None
    content: Optional[str] = None

class KarmaLog(BaseModel):
    user_id: str
    karma_log: List[Activity]

class FraudResponse(BaseModel):
    user_id: str
    fraud_score: float
    suspicious_activities: List[Dict[str, Any]]
    status: str

# Global variables
config = {}
fraud_detector = None

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    global config, fraud_detector
    
    # Make sure model directory exists
    os.makedirs('app/models', exist_ok=True)
    
    # Load configuration
    config_path = 'app/config.json'
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        config = {"fraud_threshold": 0.7}  # Default config
    
    # Check if model files exist
    model_path = 'app/models/model.pkl'
    feature_names_path = 'app/models/feature_names.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(feature_names_path):
        print("Warning: Model files not found. The API will start but analyze endpoint will fail.")
        return
    
    # Initialize fraud detector
    try:
        fraud_detector = FraudDetector(
            model_path=model_path,
            config_path=config_path,
            feature_names_path=feature_names_path
        )
        print("Fraud detector initialized successfully")
    except Exception as e:
        print(f"Error initializing fraud detector: {e}")

@app.post("/analyze", response_model=FraudResponse)
async def analyze_karma(log: KarmaLog):
    """
    Analyze a karma log for fraudulent activities
    """
    global fraud_detector, config
    
    if fraud_detector is None:
        raise HTTPException(status_code=500, 
                           detail="Fraud detector not initialized. Make sure model files exist in app/models/")

    # Convert the Pydantic model to a dictionary
    log_dict = log.dict()

    # Add the user_id as to_user for each activity if not already present
    for activity in log_dict["karma_log"]:
        if not activity.get("to_user"):
            activity["to_user"] = log_dict["user_id"]

    try:
        fraud_score, suspicious_activities = fraud_detector.detect_fraud(log_dict)
        status = "flagged" if fraud_score > config.get("fraud_threshold", 0.7) else "clean"

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
    """Health check endpoint"""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/version")
async def version():
    """Get the API version"""
    global config
    return {
        "version": "1.0", 
        "model_version": config.get("model_version", "unknown"),
        "config_version": config.get("config_version", "unknown")
    }