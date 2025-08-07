# chunkers/token_splitter.py

from llama_index.core.node_parser import TokenTextSplitter

def chunk_documents(documents, chunk_size=512, chunk_overlap=50):
    chunker = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = chunker.get_nodes_from_documents(documents)
    print(f"🔹 Total chunks created with TokenTextSplitter: {len(nodes)}")
    print("📌 First chunk preview:\n")
    print(nodes[0].get_content()[:500], "...\n")
    return nodes