import asyncio
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from mcp_use import MCPAgent, MCPClient

async def run_memory_chat():
    # Load environment variables
    load_dotenv()
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    # Config file path - change if needed
    config_file = "browser_mcp.json"

    print("Initializing chat...")

    # Create MCP client and agent with memory enabled
    client = MCPClient.from_config_file(config_file)
    llm = ChatGroq(model="qwen-qwq-32b")  # You can change the model

    # Create agent with memory_enabled=True
    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=30,
        memory_enabled=True
    )

    # Get input from user
    query = input("Enter your query: ")

    # Run the agent
    result = await agent.run(query, max_steps=30)
    print(f"\nResult: {result}")

if __name__ == "__main__":
    asyncio.run(run_memory_chat())
