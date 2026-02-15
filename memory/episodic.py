import chromadb
from chromadb.utils import embedding_functions
import uuid
import time
from colorama import Fore

class EpisodicMemory:
    def __init__(self, path="./memory_db"):
        self.client = chromadb.PersistentClient(path=path)
        
        # We use a standard embedding function (all-MiniLM-L6-v2 is built-in and fast)
        # This converts text into a list of numbers (vectors)
        self.embedding_func = embedding_functions.DefaultEmbeddingFunction()
        
        # Create or get the collection (think of this as a "Table" in SQL)
        self.collection = self.client.get_or_create_collection(
            name="episodic_logs",
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity for "meaning"
        )
        print(Fore.GREEN + f"✔ Memory System Loaded. Total Memories: {self.collection.count()}")

    def add_memory(self, text, metadata=None):
        """
        Stores a memory.
        text: The raw content (or summary)
        metadata: Dict containing timestamp, speaker, emotion, etc.
        """
        if metadata is None:
            metadata = {}
            
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = time.time()
            
        # Generate a unique ID for the memory
        mem_id = str(uuid.uuid4())
        
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[mem_id]
        )
        # print(Fore.CYAN + f"  → [Memory Stored]: {text[:30]}...") 

    def recall(self, query_text, n_results=3):
        """
        Searches for memories conceptually related to 'query_text'
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        # Chroma returns a complex object; let's simplify it for our brain
        memories = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                memories.append({"text": doc, "meta": meta})
                
        return memories

    def count(self):
        return self.collection.count()