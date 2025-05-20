import random
import json
import os
import pickle
from datetime import datetime, timedelta
from faker import Faker

def generate_dataset():
    """Generate synthetic karma logs dataset for training"""
    fake = Faker()

    print("Generating synthetic dataset...")

    # Configuration
    NUM_USERS = 220
    NUM_LOGS = 3000
    BOT_USER_COUNT = 10
    FRAUD_RATIO = 0.2  # 20% of logs will be fraudulent

    # User IDs
    user_ids = [f"stu_{i:04d}" for i in range(NUM_USERS)]
    bot_ids = [f"bot_{i:04d}" for i in range(BOT_USER_COUNT)]
    all_users = user_ids + bot_ids

    # Activity ID tracker for uniqueness
    used_activity_ids = set()

    # Karma types and weights
    karma_types = ["upvote_received", "comment", "post"]
    karma_weights = [0.5, 0.3, 0.2]

    # Load junk comments from config if exists
    config_path = '../app/config.json' if os.path.exists('../app/config.json') else 'app/config.json'
    junk_comments = []
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            try:
                config = json.load(f)
                junk_comments = config.get("junk_comments", [])
            except:
                pass

    if not junk_comments:
        junk_comments = [
            "Nice post bro!", "🔥🔥🔥", "Cool!", "Wow!", "Sub 4 sub?", "Check mine!", "Good", "Awesome bro",
            "Follow me!", "Like my page", "Great stuff", "Amazing!", "Keep it up", "Nice one",
            "Thanks", "Appreciate it", "Well done", "Nice work", "Good job"
        ]

    normal_comments = [
        "I found your explanation helpful.", "Thanks for sharing this post!",
        "Can you elaborate more on this topic?", "Interesting perspective.",
        "Very well written and informative.", "This is insightful, thank you.",
        "I learned something new today.", "Great analysis!",
        "Well-researched article.", "This is thought-provoking.",
        "I agree with your points.", "Excellent breakdown of the topic.",
        "This is a comprehensive guide.", "I'll be referring to this post.",
        "Your perspective is refreshing.", "This is a valuable contribution."
    ]

    post_titles = [
        "How to get started with Python",
        "Understanding Machine Learning Basics",
        "The Future of AI",
        "Web Development Trends in 2025",
        "Data Science for Beginners",
        "Advanced Python Techniques",
        "Building REST APIs with FastAPI",
        "Introduction to Docker",
        "Mastering Git and GitHub",
        "The Art of Debugging"
    ]

    # Generate mutual buddy pairs
    mutual_pairs = set()
    while len(mutual_pairs) < 60:  # 30 pairs = 60 mutual entries
        u1, u2 = random.sample(user_ids, 2)
        mutual_pairs.add((u1, u2))
        mutual_pairs.add((u2, u1))

    # Distribute log counts more evenly across users
    user_activity_counts = {uid: 0 for uid in all_users}

    def get_random_user(exclude=None):
        candidates = [u for u in all_users if u != exclude]
        # Bias selection toward less active users
        weights = [1 / (user_activity_counts[u] + 1) for u in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]

    def generate_unique_activity_id():
        while True:
            act_id = f"act_{random.randint(10000, 99999)}"
            if act_id not in used_activity_ids:
                used_activity_ids.add(act_id)
                return act_id

    base_time = datetime.utcnow() - timedelta(days=30)

    def generate_log():
        """Generate a single karma log entry"""
        from_user = get_random_user()
        to_user = get_random_user(exclude=from_user)

        # Update usage
        user_activity_counts[from_user] += 1
        user_activity_counts[to_user] += 1

        # Random timestamp within last 30 days
        timestamp = base_time + timedelta(minutes=random.randint(0, 60 * 24 * 30))

        karma_type = random.choices(karma_types, weights=karma_weights)[0]

        post_id = f"post_{random.randint(1000, 9999)}" if karma_type in ["upvote_received", "comment"] else None
        content = None

        if karma_type == "comment":
            content = random.choice(junk_comments) if random.random() < 0.4 else random.choice(normal_comments)
        elif karma_type == "post":
            content = random.choice(post_titles)

        activity_id = generate_unique_activity_id()

        label = "normal"
        fraud_reasons = []

        # Fraud detection logic
        if karma_type == "upvote_received":
            if (from_user.startswith("bot_") or (from_user, to_user) in mutual_pairs) and random.random() < 0.8:
                label = "karma-fraud"
                fraud_reasons.append("mutual upvote")

            elif random.random() < 0.05:
                label = "karma-fraud"
                fraud_reasons.append("karma burst")

        if karma_type == "comment" and content in junk_comments:
            if random.random() < 0.7:
                label = "karma-fraud"
                fraud_reasons.append("junk comment")

        if random.random() < FRAUD_RATIO and not fraud_reasons:
            label = "karma-fraud"
            fraud_reasons.append("random fraud")

        # Add noise: revert some fraud to normal
        if label == "karma-fraud" and random.random() < 0.05:
            label = "normal"
            fraud_reasons = []

        return {
            "activity_id": activity_id,
            "timestamp": timestamp.isoformat() + "Z",
            "from_user": from_user,
            "to_user": to_user,
            "karma_type": karma_type,
            "content": content,
            "post_id": post_id,
            "label": label,
            "fraud_reasons": fraud_reasons
        }

    logs = [generate_log() for _ in range(NUM_LOGS)]

    os.makedirs("data_generation", exist_ok=True)

    output_path = "data_generation/karma_logs.json"
    with open(output_path, "w") as f:
        json.dump(logs, f, indent=2)

    fraud_count = sum(1 for log in logs if log['label'] == 'karma-fraud')
    fraud_percentage = fraud_count / len(logs) * 100

    print(f"✅ Dataset generated: {output_path} ({len(logs)} entries)")
    print(f"📊 Fraud ratio: {fraud_percentage:.2f}% ({fraud_count} / {len(logs)})")
    print(f"👤 Users: {NUM_USERS} students + {BOT_USER_COUNT} bots")
    print(f"🤝 Mutual pairs: {len(mutual_pairs) // 2} pairs")

    # Save mutual pairs for training
    os.makedirs("app/models", exist_ok=True)
    with open("app/models/mutual_pairs.pkl", "wb") as f:
        pickle.dump(mutual_pairs, f)
    
    print(f"💾 Mutual pairs saved to app/models/mutual_pairs.pkl for training")

if __name__ == "__main__":
    generate_dataset()
