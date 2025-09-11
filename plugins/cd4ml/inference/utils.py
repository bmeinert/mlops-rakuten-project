import hashlib
import time
import random

def generate_id(number_of_digits) -> str:
    base = f"{time.time()}_{random.random()}"
    hash_object = hashlib.sha256(base.encode())
    short_hash = hash_object.hexdigest()[:number_of_digits]
    return short_hash