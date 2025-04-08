"""
Example of Rag using 
https://huggingface.co/learn/agents-course/unit3/agentic-rag/introduction

Using Ollama and Qwen coder models instruct

for this example you need to start the container and then load the model one time:
```sh
docker exec -it ollama ollama run qwen2.5-coder:7b-instruct
```
"""
import asyncio
from assistant import create_assistant_with_tools
from langchain_core.messages import HumanMessage
from retriever import guest_info_tool
from tools import search_tool, weather_info_tool, hub_stats_tool

def log_question_answer(question, answer):
    print("- Question:",question,"\n   🎩 Alfred's Response:\n       ", answer, "\n")

async def main():
    print("🎩 Alfred's A Gala Agent is ready\n=================================\n\n")
    alfred = create_assistant_with_tools([guest_info_tool, search_tool, weather_info_tool, hub_stats_tool])

    questions = [
        "Tell me about 'Lady Ada Lovelace'", 
        "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?",
        "One of our guests is from Qwen. What can you tell me about their most popular model?",
        "I need to speak with 'Dr. Nikola Tesla' about recent advancements in wireless energy. Can you help me prepare for this conversation?"
    ]
    for q in questions:
        response = alfred.invoke({"messages": q})
        #print("- Question:",q,"\n   🎩 Alfred's Response:\n       ", response['messages'][-1].content, "\n")
        log_question_answer(q, response['messages'][-1].content)
        # NOTE: added delays to avoid rate limit issues with DuckDuckGo
        await asyncio.sleep(1)
    

    #  Advanced Features: Conversation Memory
    print("\nAdvanced Features: Conversation Memory:\n")
    # First interaction
    q = "Tell me about 'Lady Ada Lovelace'. What's her background and how is she related to me?"
    response = alfred.invoke({"messages": [HumanMessage(content=q)]})
    log_question_answer(q, response['messages'][-1].content)
    await asyncio.sleep(1)

    # Second interaction (referencing the first)
    q="What projects is she currently working on?"
    response = alfred.invoke({"messages": response["messages"] + [HumanMessage(content=q)]})
    log_question_answer(q, response['messages'][-1].content)
    await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
    