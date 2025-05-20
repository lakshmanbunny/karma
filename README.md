# Karma Fraud Detector

A system to detect fraudulent karma gain activities in a social platform.

## Setup

1. Install Docker.
2. Clone this repository.
3. Generate the training dataset:
   ```bash
   cd data_generation
   python generate_dataset.py
   ```
4. Train the model:
   ```bash
   python scripts/train_model.py
   ```
5. Build the Docker image:
   ```bash
   docker build -t karma-fraud-detector .
   ```
6. Run the container:
   ```bash
   docker run -p 8000:8000 -v $(pwd)/app/models:/app/app/models karma-fraud-detector
   ```

## API Endpoints

- `POST /analyze`: Analyze karma logs for fraud
- `GET /health`: Health check endpoint
- `GET /version`: Get the API version

## Example API Request

```json
{
  "user_id": "stu_0001",
  "karma_log": [
    {
      "activity_id": "act_1234",
      "type": "upvote_received",
      "from_user": "stu_0002",
      "timestamp": "2024-05-20T14:30:00Z",
      "post_id": "post_1000"
    }
  ]
}
```