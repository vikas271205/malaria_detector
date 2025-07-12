import firebase_admin
from firebase_admin import credentials, firestore
import os
import json

# 🔐 Read from environment variable
cred_json = os.environ.get("FIREBASE_CREDENTIALS")

if not cred_json:
    raise ValueError("FIREBASE_CREDENTIALS environment variable is not set")

# Parse the JSON string into a Python dict
cred_dict = json.loads(cred_json)

# Pass dict directly to firebase_admin
cred = credentials.Certificate(cred_dict)

# Initialize Firebase Admin SDK
firebase_admin.initialize_app(cred)
db = firestore.client()
