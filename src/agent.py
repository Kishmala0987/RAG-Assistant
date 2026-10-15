#Step 7
from langchain.agents import create_agent
from src import config
from src.logger import get_logger

logger = get_logger(__name__)
def create_hr_agents(llm, tools):
    """Create HR agent that can call tools to  answer question"""
    logger.info("Creating Hr agent with tools", len(tools))
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=config.SYSTEM_PROMPT
    )
    logger.info("HR Agent Ready!!")
    return agent