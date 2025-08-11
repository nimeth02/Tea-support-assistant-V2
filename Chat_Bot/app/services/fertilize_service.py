from langchain.tools import Tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from app.services.llm_service import LLMService
from typing import Dict, Any, List
import requests
from app.core.config import settings
from app.mocks.tea_land_mock_db import tea_land_data
from app.mocks.fertilize_history_mock_db import fertilize_history_data
from datetime import datetime
import joblib
import numpy as np
from pathlib import Path
import os


class FertilizeService:
    def __init__(self):
        self.llm = LLMService().get_llm()
        self.tools = self._initialize_tools()
        self.agent = self._initialize_agent()

    def _initialize_tools(self):
        """Define the three fertilizer tools"""
        return [
            Tool(
                name="WeatherFforecast",
                func=lambda location: self._weather_forecast(location),
                description="Fetches a 14-day weather forecast (rain chance and precipitation) for a given location. Input: location (string, e.g., 'Kandy'). Use TeaLandDataByUser first to get the location for a userId.",
            ),
            Tool(
                name="DosageCalculator",
                func=lambda userId: self._dosage_calculator_by_user(int(userId)),
                description="Calculates fertilizer dosage in kg for a userId. Input: userId (integer, e.g., 3)",
            ),
            Tool(
                name="TeaLandDataByUser",
                func=lambda userId: self._tea_land_data_by_user(int(userId)),
                description="Retrieves tea land data (e.g., location, land size) for a userId. Input: userId (integer, e.g., 3).",
            ),
            Tool(
                name="GetFertilizeHistoryByUser",
                func=lambda userId: self._get_fertilize_history_by_user(int(userId)),
                description="Fetches fertilization history for a userId. Input: userId (integer, e.g., 3).",
            )
        ]

    def _initialize_agent(self):
        template = """You are a fertilizer expert for tea plants. Answer the following questions as best you can. 
                You have access to the following tools:

                {tools}

                Use the following format:

                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question

                Guidelines:
                - Use "TeaLandDataByUser" first if the query requires user-specific data (e.g., location or land details) with the userId.
                - Use "WeatherFforecast" for weather-related queries, but only after getting the location from "TeaLandDataByUser".
                - Use "DosageCalculator" for fertilizer dosage questions.
                - Use "GetFertilizeHistoryByUser" for fertilization history queries, using the userId.
                - If the query is unrelated to fertilizers or tea plants (e.g., "What's the weather in London?"), respond with: "I can only assist with tea plant fertilization questions. Try asking about fertilizer dosage, weather for your tea land, or fertilization history."
                - If no answer is found after 3 tool calls, respond with: "Sorry, I could not find an answer. Please rephrase or try asking something like: 'How much fertilizer should I use?', 'What’s the weather for my tea land?', or 'When was my last fertilization?'"
                - Final answer should be very simple understandable for low knowleded farmer

                Question: {input}
                UserId: {userId}
                Thought:{agent_scratchpad}"""

        prompt = ChatPromptTemplate.from_template(template).partial(
            tool_names=", ".join([tool.name for tool in self.tools])
        )
        return create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)

    async def fertilize(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the fertilizer service query using the agent."""
        query = state["query"]
        userId= state["userId"]

        try:
            executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,  # Enable verbose logging for debugging
                max_iterations=5,  # Limit iterations to prevent infinite loops
                timeout = 60 ,
                handle_parsing_errors=True,
            )
            result = executor.invoke({"input": query, "userId": userId})

            print("agent result", result)

            # Format response
            return {
                "intent": state["intent"],
                "history": state["history"]
                + [f"User: {query}", f"Bot: {result['output']}"],
                "query": query,
                "response": result["output"],
            }

        except Exception as e:
            
            # User-friendly fallback response
            fallback_response = (
                "Sorry, I couldn’t process your request right now. "
                "Please try rephrasing your question or ask something like:\n"
                "- 'How much fertilizer should I use for my tea plants?'\n"
                "- 'What’s the weather forecast for Kandy?'\n"
                "- 'When was my last fertilization?'"
            )
            return {
                "intent": state["intent"],
                "history": state["history"] + [f"User: {query}", f"Bot: {fallback_response}"],
                "query": query,
                "response": fallback_response,
            }

    def _weather_forecast(self, location: str) -> dict:
        """Return list of daily rain chance and precipitation for 14 days."""
        print("_weather_forecast")
        print(location)
        url = f"{settings.WHETHER_BASE_URL}/forecast.json"
        params = {"key": settings.WHETHER_API, "q": location, "days": 14}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        daily = data.get("forecast", {}).get("forecastday", [])
        result = []
        for day in daily:
            day_info = day.get("day", {})
            result.append(
                {
                    "date": day.get("date"),
                    "chance_of_rain": day_info.get("daily_chance_of_rain"),
                    "total_precip_mm": day_info.get("totalprecip_mm"),
                }
            )

        # forecast = fetch_daily_rain_forecast("Kandy", days=14)
        forecast = result
        print(forecast)
        return forecast

    def _dosage_calculator_by_user(self, userId: int) -> str:
        """Tool to calculate fertilizer amount dosage"""
        print("dosage calculation")
        land_data = None
        user_id = int(userId)
        print(user_id)
        if user_id:
            land_data = next(
                (item for item in tea_land_data if item["userId"] == user_id), None
            )

        if not land_data:
            return "Could not find land data. " + default_response

        # 3. Get most recent fertilization date
        fert_history = [f for f in fertilize_history_data if f["userId"] == user_id]
        if not fert_history:
            days_since = 365  # Default to 1 year if no history
        else:
            latest_fert = max(
                fert_history,
                key=lambda x: datetime.strptime(x["fertilizedDate"], "%Y-%m-%d"),
            )
            days_since = (
                datetime.now()
                - datetime.strptime(latest_fert["fertilizedDate"], "%Y-%m-%d")
            ).days

        try:
            base_dir = Path(__file__).resolve().parent.parent
            model_path = os.path.join(base_dir, "ml_models", "tea_fertilizer_model.pkl")

            model_artifacts = joblib.load(model_path)
            model = model_artifacts["model"]
            scaler = model_artifacts["scaler"]
            poly = model_artifacts["poly"]
        except Exception as e:
            print(f"Model loading failed: {e}")
            return "Model error. " + default_response

        # 5. Prepare input data
        input_data = np.array(
            [
                [
                    land_data["landSize"],
                    land_data["plantAmount"],
                    land_data["degree"],
                    days_since,
                ]
            ]
        )

        print(input_data)

        # 6. Transform and predict
        input_scaled = scaler.transform(input_data)
        input_poly = poly.transform(input_scaled)
        dosage = round(float(model.predict(input_poly)[0]), 2)

        print("dosage", dosage)
        # 7. Format response
        return (
            f"Recommended fertilizer dosage: {dosage} kg\n"
            f"Calculation based on:\n"
            f"- Location: {land_data['location']}\n"
            f"- Land size: {land_data['landSize']} acres\n"
            f"- Plant count: {land_data['plantAmount']}\n"
            f"- Slope: {land_data['degree']}/10\n"
            f"- Days since last fertilization: {days_since}\n"
            f"(User ID: {user_id})"
        )

    def _tea_land_data_by_user(self, userId: int) -> List[Dict[str, Any]]:
        """
        Get tea land data.
        This function finds tea land records according to the userId.
        The query parameter is expected to be a string representation of the userId."""
        print("_tea_land_data_by_user")
        # print("Available tea_land_data:", tea_land_data)
        # print(userId)
        user_id = int(userId)

        result = [data for data in tea_land_data if data["userId"] == user_id]
        print(result)
        if not result:
            return [{"error": f"No data found for userId {user_id}"}]
        return result

    def _get_fertilize_history_by_user(self, userId: int) -> List[Dict[str, Any]]:
        """
        Get fertilizer history data.
        This function finds fertilizer records according to the userId.
        The query parameter is expected to be a string representation of the userId.
        """
        print("_get_fertilize_history_by_user")
        user_id = int(userId)

        result = [
            data for data in fertilize_history_data if data["userId"] == user_id
        ]
        print(result)

        if not result:
            return [
                {"error": f"No fertilizer history found for userId {user_id}"}
            ]

        sorted_history = sorted(
                result,
                key=lambda x: datetime.strptime(x["fertilizedDate"], "%Y-%m-%d"),
                reverse=True
            )
            # Format each record as a user-friendly string
        formatted_entries = [
                f"You fertilized on {datetime.strptime(entry['fertilizedDate'], '%Y-%m-%d').strftime('%B %d, %Y')} with {entry['fertilizedAmount']} kg"
                for entry in sorted_history
            ]
            # Join entries with periods
        return ". ".join(formatted_entries) + "."


    # def fetch_daily_rain_forecast(location: str, days: int = 14) -> List[Dict[str, any]]:
    #     url = f"{settings.WHETHER_BASE_URL}/forecast.json"
    #     params = {"key": settings.WHETHER_API, "q": city, "days": 14}
    #     response = requests.get(url, params=params, timeout=10)
    #     response.raise_for_status()
    #     data = response.json()

    #     daily = data.get("forecast", {}).get("forecastday", [])
    #     result = []
    #     for day in daily:
    #         day_info = day.get("day", {})
    #         result.append({
    #             "date": day.get("date"),
    #             "chance_of_rain": day_info.get("daily_chance_of_rain"),
    #             "total_precip_mm": day_info.get("totalprecip_mm")
    #         })
    #     return result
