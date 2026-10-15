"""Step 0b: LangSmith tracing.

LangSmith tracing needs no wiring in our own code - LangChain looks for
LANGSMITH_TRACING / LANGSMITH_ENDPOINT / LANGSMITH_API_KEY / LANGSMITH_PROJECT
directly in the environment (loaded from .env by config.py) and, if
tracing is turned on, automatically sends a trace of every LLM call,
tool call, and agent step to your LangSmith project.

This module doesn't turn tracing on - the env vars already do that. All
it does is log, once per run, whether tracing is active, so it's obvious
from the logs (see logger.py) whether this run was traced.
"""
from src import config
from src.logger import get_logger

logger = get_logger(__name__)

def check_langsmith_tracing() -> None:
    """Lof wheather langsimth tracing is enable for this run """
    tracing_on = config.LANGSMITH_TRACING.lower() == "true"
    if tracing_on and config.LANGSMITH_API_KEY:
        logger.info(
            f"LangSmith tracing enable for project {config.LANGSMITH_PROJECT}, Traces at {config.LANGSMITH_ENDPOINT}"
        )
    else:
        logger.info("LangSmith tracing is of, set it true in .env file")
