from langchain_core.prompts import ChatPromptTemplate


clasiify_intent_prompt = ChatPromptTemplate.from_template(
                "Classify the intent of the following query as 'fertilize','deceses','maintaince','Help' in tea cultivation:\n"
                "Query: {query}\n"
                "Respond with only the intent (e.g., 'faq')."
            )