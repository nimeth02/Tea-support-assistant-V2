from fastapi import APIRouter, Depends, HTTPException
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.core.dependancies import get_pinecone_index 

router = APIRouter()

@router.get("/welcome")
async def read_root():
    return {"message": "Welcome to the Chat API"}

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest ):
    try:
        chat_service = ChatService()
        userId=3
        response = await chat_service.process_chat(request,userId)
        print(response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))