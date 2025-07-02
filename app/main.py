# from typing import Dict

# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.security import HTTPBasic, HTTPBasicCredentials


# app = FastAPI()
# security = HTTPBasic()

# # Dummy user database
# users_db: Dict[str, Dict[str, str]] = {
#     "Tony": {"password": "password123", "role": "engineering"},
#     "Bruce": {"password": "securepass", "role": "marketing"},
#     "Sam": {"password": "financepass", "role": "finance"},
#     "Peter": {"password": "pete123", "role": "engineering"},
#     "Sid": {"password": "sidpass123", "role": "marketing"},
#     "Natasha": {"passwoed": "hrpass123", "role": "hr"}
# }


# # Authentication dependency
# def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
#     username = credentials.username
#     password = credentials.password
#     user = users_db.get(username)
#     if not user or user["password"] != password:
#         raise HTTPException(status_code=401, detail="Invalid credentials")
#     return {"username": username, "role": user["role"]}


# # Login endpoint
# @app.get("/login")
# def login(user=Depends(authenticate)):
#     return {"message": f"Welcome {user['username']}!", "role": user["role"]}


# # Protected test endpoint
# @app.get("/test")
# def test(user=Depends(authenticate)):
#     return {"message": f"Hello {user['username']}! You can now chat.", "role": user["role"]}


# # Protected chat endpoint
# @app.post("/chat")
# def query(user=Depends(authenticate), message: str = "Hello"):
#     return "Implement this endpoint."


# app.py

# from config.settings import Settings

# def main():
#     config = Settings()
#     print("Groq Key:", config.groq_api_key)
#     print("Model:", config.embedding_model)

# if __name__ == "__main__":
#     main()


# app/main.py

# from auth.authentication import UserManager

# def test_auth():
#     um = UserManager()

#     test_cases = [
#         ("finance_user", "finance123"),
#         ("marketing_user", "wrongpassword"),
#         ("ceo", "ceo123"),
#         ("employee", "emp123"),
#         ("hr_user", "hr123"),
#         ("invalid_user", "somepass"),
#     ]

#     for username, password in test_cases:
#         result = um.authenticate(username, password)
#         if result:
#             print(f"✅ Authenticated {username} → Role: {result['role']}")
#         else:
#             print(f"❌ Authentication failed for {username}")

# if __name__ == "__main__":
#     test_auth()


# app/main.py

# from auth.rbac import RolePermissions

# def test_permissions():
#     roles = ["finance", "marketing", "hr", "engineering", "c_level", "employee", "guest"]
#     sources = ["finance", "marketing", "hr", "engineering", "general", "unknown"]

#     print("📋 Testing get_allowed_sources:")
#     for role in roles:
#         allowed = RolePermissions.get_allowed_sources(role)
#         print(f"  🔑 {role}: {allowed}")

#     print("\n🔐 Testing can_access_source:")
#     test_cases = [
#         ("finance", "finance"),
#         ("finance", "hr"),
#         ("employee", "general"),
#         ("employee", "finance"),
#         ("c_level", "marketing"),
#         ("guest", "general"),
#         ("guest", "finance"),
#     ]

#     for role, source in test_cases:
#         result = RolePermissions.can_access_source(role, source)
#         print(f"  {role} → {source}: {'✅ Access' if result else '❌ Denied'}")

# if __name__ == "__main__":
#     test_permissions()




# app/main.py

# from rag.document_loader import DocumentLoader
# import pprint

# def test_document_loader():
#     data_dir = "app/data/"  # Adjust if needed
#     department = "finance"      # Try with 'hr' too

#     loader = DocumentLoader(data_dir)
#     documents = loader.load_documents_by_department(department)

#     print(f"\n📄 Loaded {len(documents)} documents for department: {department}\n")
#     for doc in documents:
#         print("Source:", doc['source'])
#         print("Department:", doc['department'])
#         print("Path:", doc['metadata']['file_path'])
#         print("Content Preview:", doc['content'][:200], "...\n")  # print first 200 chars

# if __name__ == "__main__":
#     test_document_loader()


# app/main.py

# from app.rag.document_loader import DocumentLoader
# from app.rag.embeddings import EmbeddingManager

# def test_embedding_flow():
#     department = "hr"
#     data_path = "app/data"

#     # 1. Load documents
#     loader = DocumentLoader(data_path)
#     documents = loader.load_documents_by_department(department)
#     print(f"✅ Loaded {len(documents)} documents.")

#     # 2. Embed & store in ChromaDB
#     embedder = EmbeddingManager()
#     embedder.add_documents(department, documents)
#     print(f"✅ Embedded and stored documents for department: {department}")

#     # 3. Test a search query
#     query = "annual financial report"
#     results = embedder.search(query, departments=[department])
#     print(f"\n🔍 Search Results for: '{query}'\n")

#     for res in results:
#         print("📄 Content:", res['content'][:150])
#         print("📁 Source:", res['metadata']['source'])
#         print("🏷️ Chunk ID:", res['metadata']['chunk_id'])
#         print("📏 Distance:", res['distance'])
#         print("--------")

# if __name__ == "__main__":
#     test_embedding_flow()


# from dotenv import load_dotenv
# import os

# from rag.document_loader import DocumentLoader
# from rag.embeddings import EmbeddingManager
# from rag.generation import ResponseGenerator  # <- your current code

# def test_response_generator():
#     # Step 1: Load env
#     load_dotenv()
#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         raise ValueError("GROQ_API_KEY not found in environment!")

#     # Step 2: Load documents
#     department = "finance"
#     loader = DocumentLoader("data")
#     docs = loader.load_documents_by_department(department)

#     # Step 3: Embed & store
#     embedder = EmbeddingManager()
#     embedder.add_documents(department, docs)

#     # Step 4: Search relevant docs
#     query = "What is the total budget for 2024?"
#     top_docs = embedder.search(query, departments=[department])

#     # Step 5: Generate response
#     generator = ResponseGenerator(api_key)
#     answer = generator.generate_response(query, context_documents=top_docs, user_role="finance")

#     print("\n🧠 AI Answer:\n")
#     print(answer)

# if __name__ == "__main__":
#     test_response_generator()


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