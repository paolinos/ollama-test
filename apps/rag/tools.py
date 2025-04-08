from langchain.tools import Tool
import random
from langchain_community.tools import DuckDuckGoSearchRun
from huggingface_hub import list_models
from assistant import create_assistant_with_tools
from langchain_core.messages import HumanMessage

search_tool = DuckDuckGoSearchRun()
#results = search_tool.invoke("Who's the current President of France?")
#print(results)


def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# Initialize the tool
weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location."
)


def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # List models from the specified author, sorted by downloads
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"

# Initialize the tool
hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description="Fetches the most downloaded model from a specific author on the Hugging Face Hub."
)
# Example usage
#print(hub_stats_tool("facebook")) # Example: Get the most downloaded model by Facebook

if __name__ == "__main__":
    alfred = create_assistant_with_tools([search_tool, weather_info_tool, hub_stats_tool])

    # Questions about people and need to have the tools
    messages = [HumanMessage(content="Who is Facebook and what's their most popular model?")]
    response = alfred.invoke({"messages": messages})

    print("🎩 Alfred's Response:")
    print(response['messages'][-1].content)