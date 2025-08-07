#SimpleDirectoryReader: Loads all .txt, .md, .pdf files in the ./data/ folder.
#VectorStoreIndex: Builds a vector index (i.e., searchable embedding database).
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex 
#This replaces the old ServiceContext. You set your global llm and embed_model here.
from llama_index.core.settings import Settings
#Uses a local sentence-transformer model to convert chunks into vector embeddings.
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#Connects to the local Ollama server to use models like llama3, mistral, etc.
from llama_index.llms.ollama import Ollama
#This is the chunker — it splits your docs into smaller pieces (aka “nodes”).
from llama_index.core.node_parser import SentenceSplitter
#Suppresses annoying OpenSSL/LibreSSL compatibility warnings on macOS.
import warnings 
warnings.filterwarnings("ignore", message=".*LibreSSL.*")
#Load and chunk documents | Loads all readable files in ./data.
documents = SimpleDirectoryReader("./data").load_data()
# THIS is the chunking strategy 
# It chunks text into ~512 tokens per piece
# Each chunk overlaps 50 tokens with the previous one (helps with context continuity)
# It splits by sentence boundaries where possible for better semantic breaks
chunker = SentenceSplitter(chunk_size=512, chunk_overlap=50)#Fixed Size Chunks, 512 tokens, 50 overlap
# Actually applies the chunking and gives you a list of chunked text “nodes” (with metadata).
nodes = chunker.get_nodes_from_documents(documents)

#Embedding + LLM setup
# This model converts each chunk into a numerical vector for comparison.
# This is what makes semantic search work.
emded_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
# This is your generator — it’s the brain that reads the retrieved chunks and answers your query.
llm = Ollama(model="llama3")
# Registers both models with LlamaIndex so they get used throughout.
Settings.embed_model = emded_model
Settings.llm = llm

# Builds a vector index from all the document chunks (embeddings stored in memory).
# This is the retriever part of RAG.
index = VectorStoreIndex(nodes)

#Query Loop
# Waits for a user query.
# Retrieves top 3 similar chunks (via vector similarity).
# Feeds them to the LLM (Ollama) to generate a final response.
while True:
    query = input("Ask Something: ")
    if query.lower() in ["exit", "quit"]:
        break
    engine = index.as_query_engine(similarity_top_k=3)  # Top 3 similar nodes
    response = engine.query(query)
    print(f"Response: {response}")
                                  