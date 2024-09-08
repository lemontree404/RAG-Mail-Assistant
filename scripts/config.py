import os

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGCHAIN_API_KEY'] = 'YOUR-API-KEY'
os.environ['LANGCHAIN_PROJECT'] = "mail_assistant"

OLLAMA_BASE_URL = 'http://localhost:11434'
USER = 'GMAIL-ID'
VECTORESTORE_DIR = 'vectorstore'
LOAD_VECTORSTORE = True
LOCAL_MAIL_STORE = 'mails.txt'

# PSQL

PSQL_DB_NAME = 'PSQL-DB-NAME'
PSQL_USER = 'PSQL-USER'
PSQL_PASSWORD = 'PSQL-PASSWORD'
PSQL_HOST = 'localhost'
PSQL_PORT = 5432