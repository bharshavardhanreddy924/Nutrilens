import os
from langchain_groq import ChatGroq

class IndianDietBot:
    def __init__(self, api_key: str):
        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name="mixtral-8x7b-32768"
        )

    def respond_to_query(self, user_query: str) -> str:
        prompt = f"""
        You are an Indian diet expert. Answer the following query concisely:
        
        {user_query}
        
        If the query is about meal recommendations, include:
        1. Dish name
        2. Main ingredients
        3. Approximate nutritional information
        
        If the query is unrelated to Indian diets, please provide a general response stating that i am only well-versed as a diet recommender.
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    # Replace with your actual API key or set it as an environment variable
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_api_key_here")
    
    if not GROQ_API_KEY or GROQ_API_KEY == "your_api_key_here":
        print("Error: Please set your GROQ API key before running the bot.")
        return
    
    bot = IndianDietBot(GROQ_API_KEY)
    print("Welcome to Indian Diet Bot! Type your query or 'exit' to quit.")
    
    while True:
        user_query = input("\nYou: ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        
        response = bot.respond_to_query(user_query)
        print("\nBot:", response)

if __name__ == "__main__":
    main()
