"""
Using Ollama and Qwen coder models to create an app

Remember to create a virtualenv before to install packages

# Required packages: 
pip install ollama==0.4.7 asyncio==3.4.3 tqdm==4.67.1 psutil==7.0.0

# Run app
python ./apps/create_chat.py
"""
import asyncio
import time
from tqdm import tqdm
import psutil
from ollama import AsyncClient

# NOTE: Remember to download the model inside the container first.
LLM_MODEL="qwen2.5-coder:1.5b"  # 1.5b 3b 7b 14b 32b

# An example of content to be processed
content="""
What is Chat?
Chat is a communication tool. 
It serves as a hub for instant messaging, enabling seamless collaboration among individuals and teams in workplaces, schools, or personal settings.

What does it do?
- The chat feature allows users to:
- Communicate instantly: Send messages to individuals or groups in real-time.
- Collaborate effectively: Share files, links, and ideas within conversations.
- Stay organized: Keep track of discussions, files, and tasks in one place.
- Connect remotely: Facilitate communication across different locations and time zones.
- Chat is designed to enhance productivity and foster collaboration, making it a versatile tool for modern communication.

Features of Chat:
- One-on-One and Group Chats: Start private conversations or create group chats for team discussions.
- File Sharing: Attach and share documents, images, and other files directly within the chat.
- Formatting Options: Customize messages with bold, italics, underlining, lists, and more.
- Emoji, GIFs, and Stickers: Add a fun and personal touch to your messages.
- Message History: Access previous conversations and shared files easily.
- Integration with Apps: Use third-party apps.
- Scheduling and Actions: Schedule meetings, set reminders, and perform other actions directly from the chat.
- Search Functionality: Quickly find messages, files, or people within the chat.
- Security Features: Ensure safe communication with encryption and compliance tools.

With all this information, you need to write a chat application using Python and FastAPI. 
you will create all the required endpoints and use MongoDB as the database.

- User authentication and authorization
- Group chats
- Private messages
- File sharing with other users
- Emoji, GIFs, and Stickers to messages
- Real-time updates (using WebSockets)
- Notifications for new messages
- Use uvicorn to run the app
- add grpc to create Group and Private messages
- add unit testing to all functionality
"""

async def run_model():
  """
  Run model with the content
  """
  response = await AsyncClient().chat(model=LLM_MODEL, messages=[
    {
      'role': 'user',
      'content': content,
    },
  ])
  return response['message']['content']
  
async def check_cpu_mem_usage():
  """
  CPU & RAM stats from https://stackoverflow.com/questions/276052/how-to-get-current-cpu-and-ram-usage-in-python
  """
  with tqdm(total=100, desc='cpu%', position=1) as cpubar, tqdm(total=100, desc='ram%', position=0) as rambar:
    while True:
      rambar.n=psutil.virtual_memory().percent
      cpubar.n=psutil.cpu_percent()
      cpubar.refresh()
      rambar.refresh()
      await asyncio.sleep(0.5)

async def main():
  """
  Run app async
  """
  delta_t = time.time()
  tasks = [
    asyncio.create_task(check_cpu_mem_usage()),
    asyncio.create_task(run_model())
  ]
  done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
  for td in done:
    result = td.result()

    # Stop cpu mem stats 
    for tp in tasks:
      tp.cancel()

    # wait & Print data
    await asyncio.sleep(0.5)
    
    print("="*50 + "\n    Application Result:\n"+ "="*50)
    print(result)
    print(f"Processing time: {time.time() - delta_t} seconds")


if __name__ == "__main__":
    asyncio.run(main())