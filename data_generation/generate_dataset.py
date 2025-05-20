import random
import json
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()

# Configuration
NUM_USERS = 220
NUM_LOGS = 3000
BOT_USER_COUNT = 10
FRAUD_RATIO = 0.2  # 20% of logs will be fraudulent

# User IDs
user_ids = [f"stu_{i:04d}" for i in range(NUM_USERS)]
bot_ids = [f"bot_{i:04d}" for i in range(BOT_USER_COUNT)]
all_users = user_ids + bot_ids

# Karma types and their weights
karma_types = ["upvote_received", "comment", "post"]
karma_weights = [0.5, 0.3, 0.2]

# Comment templates
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

# Post templates
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

def generate_log():
    from_user = random.choice(all_users)
    to_user = random.choice(all_users)
    while to_user == from_user:
        to_user = random.choice(all_users)

    timestamp = datetime.utcnow() - timedelta(minutes=random.randint(0, 50000))
    karma_type = random.choices(karma_types, weights=karma_weights)[0]

    post_id = f"post_{random.randint(1000, 9999)}" if karma_type in ["upvote_received", "post"] else None
    content = None

    if karma_type == "comment":
        if random.random() < 0.4:  # 40% chance of being junk
            content = random.choice(junk_comments)
        else:
            content = random.choice(normal_comments)
    elif karma_type == "post":
        content = random.choice(post_titles)

    # Fraud labeling logic
    label = "normal"
    fraud_reasons = []

    # Mutual upvotes (50% chance if both users are bots or mutuals)
    if karma_type == "upvote_received" and (from_user.startswith("bot_") or (from_user, to_user) in mutual_pairs):
        if random.random() < 0.8:
            label = "karma-fraud"
            fraud_reasons.append("mutual upvote")

    # Junk comments
    if karma_type == "comment" and content in junk_comments:
        label = "karma-fraud"
        fraud_reasons.append("junk comment")

    # Sudden burst (simulate by short interval time)
    if karma_type == "upvote_received" and random.random() < 0.05:
        label = "karma-fraud"
        fraud_reasons.append("karma burst")

    # Ensure we have the desired fraud ratio
    if label == "normal" and random.random() < FRAUD_RATIO:
        label = "karma-fraud"
        fraud_reasons.append("random fraud")

    return {
        "timestamp": timestamp.isoformat() + "Z",
        "from_user": from_user,
        "to_user": to_user,
        "karma_type": karma_type,
        "content": content,
        "post_id": post_id,
        "label": label,
        "fraud_reasons": fraud_reasons
    }

# Mutual buddy generation (mutual karma groups)
mutual_pairs = set()
for _ in range(30):  # 30 mutual groups
    u1, u2 = random.sample(user_ids, 2)
    mutual_pairs.add((u1, u2))
    mutual_pairs.add((u2, u1))

# Generate logs
logs = [generate_log() for _ in range(NUM_LOGS)]

# Save to JSON file
with open("karma_logs.json", "w") as f:
    json.dump(logs, f, indent=2)

print(f"✅ Dataset generated: karma_logs.json ({len(logs)} entries)")
print(f"📊 Fraud ratio: {sum(1 for log in logs if log['label'] == 'karma-fraud') / len(logs):.2%}")
