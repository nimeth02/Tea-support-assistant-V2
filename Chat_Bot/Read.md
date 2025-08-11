python -m uvicorn main:app --reload





🚀 Tea Farming Support Assistant – AI-Powered Farming Help 🌱

I’ve been working on an AI-driven Tea Farming Support Assistant – a modular chatbot designed to answer farmers’ questions on tea cultivation.

💡 How it works:
A central ChatService (built with FastAPI) classifies a farmer’s query using an LLM (Groq API) and routes it to the right expert module:

🛡 DiseaseService – Tea Disease Advisor

Tools:

TeaDiseaseRAG – Retrieves local disease & pest info from Pinecone vector DB

DuckDuckGoWebSearch – Falls back on web search for broader info

Purpose: Combine local knowledge + global sources for accurate disease diagnosis.

🌾 FertilizeService – Fertilizer & Soil Advisor

Tools:

WeatherForecast – 14-day rainfall & precipitation prediction

DosageCalculator – Mock ML model (scikit-learn/joblib) for optimal fertilizer dosage

TeaLandDataByUser – Retrieves farmer’s land size, location, soil type

GetFertilizeHistoryByUser – Fetches past fertilization records

Purpose: Provide personalized fertilizer plans based on land, weather, and history.

📄 HelpService – Company Policy & Support

Tools:

HelpRAG – Retrieves company policies & FAQ answers from Pinecone vector DB

Purpose: Answer queries about tea collection dates, loans, advances, and policies.

⚙ Key Technologies:

Backend: FastAPI (async, modular design)

Frontend: Next.js (interactive UI)

AI Frameworks: LangChain, LangGraph

LLM: Groq API

Vector Search: Pinecone

External APIs: Weather forecast integration

Mock ML Model: Fertilizer dosage prediction (scikit-learn/joblib)