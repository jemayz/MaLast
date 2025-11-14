import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import logging
import uuid6

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class indexing:
    def __init__(self, persist_directory="chroma_db"):
        self.embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.persist_directory = persist_directory
        
        # Create persistent ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
    
    def index_documents(self, documents, collection_name="Agentic_retrieval", domain=None, top_k=5):
        """Index documents with domain-specific collections"""
        
        # Create domain-specific collection name
        if domain:
            collection_name = f"{domain}_{collection_name}"
        
        try:
            # Try to get existing collection
            existing_collection = self.chroma_client.get_collection(name=collection_name)
            logger.info(f"📚 Found existing collection '{collection_name}' with {existing_collection.count()} documents")
            
            # Create vector store from existing collection
            vector_store = Chroma(
                client=self.chroma_client,
                collection_name=collection_name,
                embedding_function=self.embedding_function
            )
            
        except Exception:
            # Collection doesn't exist, create new one
            logger.info(f"🔄 Creating collection '{collection_name}' with {len(documents)} documents...")
            logger.info(f"⏰ Estimated time: {self._estimate_time(len(documents))} minutes")
            
            vector_store = Chroma(
                client=self.chroma_client,
                collection_name=collection_name,
                embedding_function=self.embedding_function
            )
            
            # Add documents in batches with progress tracking
            batch_size = 100
            total_batches = (len(documents) + batch_size - 1) // batch_size
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                batch_num = i // batch_size + 1
                
                vector_store.add_documents(
                    documents=batch,
                    ids=[str(uuid6.uuid6()) for _ in batch]
                )
                
                logger.info(f"⚡ Progress: {batch_num}/{total_batches} batches completed")
            
            logger.info(f"✅ Successfully indexed {len(documents)} documents to '{collection_name}'")

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        
        return retriever
    
    def _estimate_time(self, num_documents):
        """Estimate indexing time based on document count"""
        # Rough estimate: ~1000 docs = 15 minutes
        return max(1, (num_documents // 1000) * 15)
    
    def add_new_documents(self, documents, collection_name="Agentic_retrieval"):
        """Add new documents to existing collection"""
        try:
            logger.info(f"Add new document'")
            vector_store = Chroma(
                client=self.chroma_client,
                collection_name=collection_name,
                embedding_function=self.embedding_function
            )
            
            vector_store.add_documents(
                documents=documents, 
                ids=[str(uuid6.uuid6()) for _ in documents]
            )
            
            logger.info(f"Added {len(documents)} new documents to '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            return False
    
    def reset_collection(self, collection_name="Agentic_retrieval"):
        """Delete existing collection to start fresh"""
        try:
            self.chroma_client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {str(e)}")
            return False