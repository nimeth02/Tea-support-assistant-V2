from pinecone import Pinecone
# from pinecone import ServerlessSpec
from typing import Dict, Any,List
from agent.config import settings
from agent.db.base_db import BaseDatabase
import time
from urllib3.exceptions import MaxRetryError, NameResolutionError
import socket
import cohere


class PineconeDatabase(BaseDatabase):
    """Pinecone vector database connection for FAQ agent."""
    
    def __init__(self):
        self.client = None
        self.index = None
        self.max_retries = 3
        self.retry_delay = 2  
    
    def _check_internet_connection(self):
        """Check if there's an internet connection."""
        try:
            # Try to resolve a known domain
            socket.gethostbyname('google.com')
            return True
        except socket.gaierror:
            return False
    
    def connect(self):
        """Initialize Pinecone connection and ensure index exists."""
        if not self._check_internet_connection():
            raise ConnectionError("No internet connection available. Please check your network connection.")
            
        retries = 0
        while retries < self.max_retries:
            try:
                print(f"Attempting to connect to Pinecone (attempt {retries + 1}/{self.max_retries})")
                
                # Initialize Pinecone client
                self.client = Pinecone(api_key=settings.PINECONE_API_KEY, environment="us-east-1")
                
                # Get index name from settings
                index_name = settings.PINECONE_DISEASE_INDEX_NAME
                print(f"Checking for index: {index_name}")
                
                # List available indexes
                try:
                    available_indexes = self.client.list_indexes().names()
                    # if index_name not in available_indexes:
                    #     print(f"Creating new index: {index_name}")
                    #     self.client.create_index(
                    #         name=index_name,
                    #         dimension=1024,  
                    #         metric="cosine",
                    #         spec=ServerlessSpec(cloud="aws", region=settings.PINECONE_REGION)
                    #     )
                except Exception as e:
                    print(f"Error listing/creating indexes: {str(e)}")
                    raise
                
                # Initialize index
                self.index = self.client.Index(index_name)
                print(f"Successfully connected to Pinecone index: {index_name}")
                return
                
            except (MaxRetryError, NameResolutionError) as e:
                retries += 1
                print(f"Connection attempt {retries} failed: {str(e)}")
                if retries < self.max_retries:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    print("Max retries reached. Could not connect to Pinecone.")
                    print("Please check your internet connection and DNS settings.")
                    raise
            except Exception as e:
                print(f"Unexpected error connecting to Pinecone: {str(e)}")
                raise
    
    def close(self):
        """Pinecone client doesn't require explicit closing (managed by Pinecone)."""
        self.client = None
        self.index = None
        print("Pinecone connection closed")
    
    def get_session(self):
        """Return the Pinecone index client."""
        if not self.index:
            self.connect()
        return self.index

    async def query(self, query: str, top_k: int = 5, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Query Pinecone index with a text query and return top-k matching records.
        
        Args:
            query (str): Text query (e.g., symptom description).
            top_k (int): Number of top results to return.
            include_metadata (bool): Whether to include metadata in results.
        
        Returns:
            Dict containing matches with IDs, scores, and metadata.
        """
        try:
            co = cohere.Client(settings.COHERE_API_KEY)

            print(query)

            response = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
            )
            print(response)

            query_embedding = response.embeddings[0]

            print(query_embedding)
            
            # Query Pinecone index
            result = self.get_session().query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=include_metadata,
                namespace="tea-disease"
            )
            print(result)
            return result
        except Exception as e:
            print(f"Error querying Pinecone: {str(e)}")
            return {"matches": [], "error": str(e)}    