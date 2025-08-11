from langchain_groq import ChatGroq
from app.core.config import settings
import httpx
import os

class LLMService:
    def __init__(self):
        try:
            if not settings.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not configured")
                
            # Verify API key format (starts with 'gsk_')
            if not settings.GROQ_API_KEY.startswith('gsk_'):
                raise ValueError("Invalid Groq API key format")
                
            self.llm = ChatGroq(
                model=settings.GROQ_MODEL,
                api_key=settings.GROQ_API_KEY,
                timeout=30.0,
                max_retries=3  # Add retry mechanism
            )
            print("Groq LLM initialized successfully")
            
            # Test connection immediately
            # self._test_connection()
        except Exception as e:
            print(f"Failed to initialize Groq LLM: {str(e)}")
            raise

    def _test_connection(self):
        """Test the connection with a simple request"""
        try:
            test_prompt = "Hello groq"
            response = self.llm.invoke(test_prompt)
            print(response)
            print("Groq connection test successful")
        except Exception as e:
            print(f"Groq connection test failed: {str(e)}")
            raise RuntimeError(f"Groq service unavailable: {str(e)}")

    def get_llm(self):
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        return self.llm


