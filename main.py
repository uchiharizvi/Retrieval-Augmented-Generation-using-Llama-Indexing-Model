from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.settings import Settings
import warnings
import sys
from chunkers.tokensplitter import chunk_documents

# Suppress LibreSSL warnings
warnings.filterwarnings("ignore", message=".*LibreSSL.*")

# Load document(s)
documents = SimpleDirectoryReader("./data").load_data()

# Import the desired chunking strategy
# Update this line to switch chunkers easily

# Apply chunking
nodes = chunk_documents(documents)

# Setup embedding + LLM
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = Ollama(model="llama3")

Settings.embed_model = embed_model
Settings.llm = llm

# Build vector index
index = VectorStoreIndex(nodes)

# Query loop
while True:
    query = input("\nAsk about the book: ")
    if query.lower() in ["exit", "quit"]:
        sys.exit(0)
    engine = index.as_query_engine(similarity_top_k=3)
    response = engine.query(query)
    print("\n📘 Answer:\n", response)