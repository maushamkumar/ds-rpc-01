from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Dict
import jwt
from datetime import datetime, timedelta
import os

from app.auth.authentication import UserManager, UserNotFoundError, PasswordMismatchError
from app.auth.rbac import RolePermissions
from app.rag.document_loader import DocumentLoader
from app.rag.embeddings import EmbeddingManager
from app.rag.generation import ResponseGenerator
from app.config.settings import settings
from app.logger.log import logging

# --- Configuration ---
SECRET_KEY = "your-secret-key"  # Replace with a strong, secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Internal RAG Chatbot API",
    description="API for an internal chatbot with role-based access control.",
    version="1.0.0"
)

logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class Token(BaseModel):
    access_token: str
    token_type: str

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    context: List[Dict]


# --- Dependencies & Managers ---
user_manager = UserManager()
embedding_manager = EmbeddingManager(
    model_name=settings.embedding_model,
    persist_directory=settings.chroma_persist_directory
)
response_generator = ResponseGenerator(api_key=settings.groq_api_key)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- Utility Functions ---
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise credentials_exception
        return {"username": username, "role": role}
    except jwt.PyJWTError:
        raise credentials_exception

# --- API Endpoints ---
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        user = user_manager.authenticate(form_data.username, form_data.password)
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user["username"], "role": user["role"]},
            expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    except (UserNotFoundError, PasswordMismatchError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    user_role = current_user["role"]
    allowed_departments = RolePermissions.get_allowed_sources(user_role)

    # Search for relevant documents
    search_results = embedding_manager.search_chunks(
        query=request.query,
        departments=allowed_departments,
        n_results=5
    )

    # Generate a response
    response_text = response_generator.generate_response(
        query=request.query,
        context_documents=search_results,
        user_role=user_role
    )

    # Extract sources
    sources = sorted(list(set(doc["metadata"]["source"] for doc in search_results)))

    return ChatResponse(response=response_text, sources=sources, context=search_results)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting up the application...")
    # Pre-load and embed documents for all departments on startup
    data_dir = "app/data"
    loader = DocumentLoader(data_dir)
    departments = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for dept in departments:
        logger.info(f"Processing department: {dept}")
        try:
            docs = loader.load_documents_by_department(dept)
            embedding_manager.add_documents(dept, docs)
        except Exception as e:
            logger.error(f"Failed to process department {dept}: {e}")

if __name__ == "__main__":
    import uvicorn
    import os
    # This part is for local development and debugging.
    # In production, you would use a Gunicorn or similar server.
    uvicorn.run(app, host="0.0.0.0", port=8000)