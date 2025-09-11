from fastapi import FastAPI, Depends, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from hashlib import sha256
from datetime import datetime, timedelta
from typing import Dict
import time
from prometheus_client import generate_latest, Counter, Histogram

# --- FastAPI Setup ---
auth_app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- JWT Configuration ---
SECRET_KEY = "your_secret_key_here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- Simple In-Memory User Database ---
users_db: Dict[str, Dict[str, str]] = {
    "admin": {"username": "admin", "password": sha256("admin123".encode()).hexdigest(), "role": "admin"},
    "user": {"username": "user", "password": sha256("user123".encode()).hexdigest(), "role": "user"},
}

# --- JWT Utilities ---
def create_access_token(data: dict, expires_delta: timedelta = None):
    """
    Generate a JWT access token with optional expiration.

    Parameters:
        data (dict): Payload to include in the token (e.g., user claims).
        expires_delta (timedelta, optional): Token lifespan. Defaults to 15 minutes.

    Returns:
        str: Encoded JWT as a string.
    """
    to_encode = data.copy()
    expire = datetime.now() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str = Depends(oauth2_scheme)):
    """
    Decode and validate a JWT access token.

    Extracts the `sub` (username) claim, verifies its presence in the user database,
    and returns the associated user object if valid.

    Parameters:
        token (str): JWT token provided via the OAuth2 scheme.

    Returns:
        dict: Authenticated user data from `users_db`.

    Raises:
        HTTPException (401): If the token is invalid, expired, or the user does not exist.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None or username not in users_db:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return users_db[username]
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# --- Models ---
class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"

# ----------------------------------------------------
# Prometheus Metrics
# ----------------------------------------------------

# Counter for number of requests
REQUEST_COUNT = Counter(
    'auth_service_requests_total',
    'Total number of requests to auth_service',
    ['method', 'endpoint', 'status_code'] 
)

# Histogram for request latency
REQUEST_LATENCY = Histogram(
    'auth_service_request_latency_seconds',
    'Request latency in seconds for auth_service',
    ['method', 'endpoint']
)


# Middleware for collecting metrics for every request
@auth_app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware to measure request processing time and expose Prometheus metrics.

    Records the number of HTTP requests (`REQUEST_COUNT`) and their latency
    (`REQUEST_LATENCY`) labeled by method and endpoint path.

    Parameters:
        request (Request): Incoming FastAPI request object.
        call_next (Callable): Handler to process the request and produce a response.

    Returns:
        Response: The original response, after metric recording.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # Collecting of general request metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(process_time)

    return response

# ----------------------------------------------------
# Auth Endpoints
# ----------------------------------------------------

@auth_app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and issue a JWT access token.

    Validates the username and password against the in-memory user database.
    Returns a bearer token if authentication succeeds.

    Parameters:
        form_data (OAuth2PasswordRequestForm): Username and password submitted via form.

    Returns:
        dict: {
            "access_token": str,
            "token_type": "bearer"
        }

    Raises:
        HTTPException (401): If authentication fails due to incorrect credentials.
    """
    user = users_db.get(form_data.username)
    hashed_pw = sha256(form_data.password.encode()).hexdigest()
    if not user or user["password"] != hashed_pw:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

# --- Admin-only Endpoints ---
def admin_required(user=Depends(verify_token)):
    """
    Dependency to restrict access to admin-only endpoints.
    """
    if user["role"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admins only")
    return user

@auth_app.post("/users")
def create_user(user_data: UserCreate, user=Depends(admin_required)):
    """
    Create a new user account (admin-only).

    Requires authentication and admin privileges. Adds a new user to the in-memory `users_db`
    with hashed password and assigned role.

    Parameters:
        user_data (UserCreate): Object containing `username`, `password`, and `role`.
        user (dict): The authenticated admin user (injected via `admin_required`).

    Returns:
        dict: Success message upon user creation.

    Raises:
        HTTPException (400): If the username already exists.
        HTTPException (403): If the requester is not an admin.
    """
    if user_data.username in users_db:
        raise HTTPException(status_code=400, detail="User already exists")
    users_db[user_data.username] = {
        "username": user_data.username,
        "password": sha256(user_data.password.encode()).hexdigest(),
        "role": user_data.role
    }
    return {"msg": "User created successfully."}

@auth_app.delete("/users/{username}")
def delete_user(username: str, user=Depends(admin_required)):
    """
    Delete an existing user account (admin-only).

    Requires authentication and admin privileges. Removes the specified user
    from the in-memory `users_db`.

    Parameters:
        username (str): Username of the account to delete.
        user (dict): The authenticated admin user (injected via `admin_required`).

    Returns:
        dict: Success message upon user deletion.

    Raises:
        HTTPException (404): If the user does not exist.
        HTTPException (403): If the requester is not an admin.
    """
    if username not in users_db:
        raise HTTPException(status_code=404, detail="User not found")
    del users_db[username]
    return {"msg": "User deleted successfully."}

# --- Health Check Endpoint ---
@auth_app.get("/health")
def health_check():
    """
    Health check endpoint for the authentication service.

    Returns a simple status message indicating the service is running.

    Returns:
        dict: {
            "status": "ok"
        }
    """
    return {"status": "ok"}


# ----------------------------------------------------
# Prometheus Metrics Endpoint
# ----------------------------------------------------

@auth_app.get("/metrics")
async def metrics():
    """
    Expose Prometheus metrics for the authentication service.

    Returns metrics in plain text format compatible with Prometheus scraping.

    Returns:
        Response: Prometheus-formatted metrics with media type:
        'text/plain; version=0.0.4; charset=utf-8'
    """
    return Response(content=generate_latest(), media_type="text/plain; version=0.0.4; charset=utf-8")