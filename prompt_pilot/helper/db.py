from pymongo import MongoClient
import os

# MongoDB configuration
MONGO_URI = os.getenv(
    "MONGO_URI", "mongodb://localhost:27017/"
)  # You can modify this if needed
client = MongoClient(MONGO_URI)
db = client.ci_chat  # This will be your database name

# Collection for storing chat histories
chat_histories = db.chat_histories  # The collection where chat history will be stored
