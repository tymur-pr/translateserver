import os
from functools import wraps
from flask import request, jsonify
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY", "NOT_SET")

def reqire_key(f):
    @wraps(f)
    def decorated_func(*args, **kwargs):
        api_key = request.headers.get("x-api-key")
        if not api_key or api_key != API_KEY:
            return jsonify({"error": "Unauthorized. Invalid or missing API key."}), 401
        return f(*args, **kwargs)
    return decorated_func


