# DS RPC 01: Internal chatbot with role based access control

This is the starter repository for Codebasics's [Resume Project Challenge](https://codebasics.io/challenge/codebasics-gen-ai-data-science-resume-project-challenge) of building a RAG based Internal Chatbot with role based access control. Please fork this repository to get started.

Basic Authentication using FastAPI's `HTTPBasic` has been implemented in `main.py` for learners to get started with.

Visit the challenge page to learn more: [DS RPC-01](https://codebasics.io/challenge/codebasics-gen-ai-data-science-resume-project-challenge)
![alt text](resources/RPC_01_Thumbnail.jpg)
### Roles Provided
 - **engineering**
 - **finance**
 - **general**
 - **hr**
 - **marketing**


## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ds-rpc-01.git
    cd ds-rpc-01
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file** in the root directory of the project and add your Groq API key:
    ```
    GROQ_API_KEY="your-groq-api-key"
    ```

## Running the Application

1.  **Start the backend server:**
    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 &  
    ```

2.  **In a new terminal, start the frontend application:**
    ```bash
    streamlit run frontend/streamlit_app.py
    ```

3.  Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Login Credentials

You can use the following username/password combinations to log in with different roles:

*   **Finance:** `finance_user` / `finance123`
*   **Marketing:** `marketing_user` / `marketing123`
*   **HR:** `hr_user` / `hr123`
*   **Engineering:** `engineering_user` / `eng123`
*   **C-Level:** `ceo` / `ceo123`
*   **Employee:** `employee` / `emp123`
*   **General:** `general_user` / `general123`


### 2.2 Project Structure
```
fintech-chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py                 
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── authentication.py   
│   │   └── rbac.py            
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── document_loader.py
│   │   ├── embeddings.py      
│   │   ├── retrieval.py       
│   │   ├── query_processor.py       
│   │   └── generation.py      
│   ├── data/
│   │   ├── finance/           # Finance documents
│   │   ├── marketing/         # Marketing documents
│   │   ├── hr/               # HR documents
│   │   ├── engineering/      # Engineering documents
│   │   └── general/          # General company documents
│   ├── exception/
│   │   ├── __init__.py                  
│   │   └── exception_handler.py/          
│   ├── logger/
│   │   ├── __init__.py
│   │   └── log.py/          
│   └── config/
│       ├── __init__.py
│       └── settings.py       
|
├── frontend/
│   └── streamlit_app.py      
├── requirements.txt
├── .env
└── README.md
```
