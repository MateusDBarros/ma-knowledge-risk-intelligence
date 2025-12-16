import logging, re
from typing import List

from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Agent(BaseModel):
    query: str
    context: List[str] = []
    answer: str = ""


def build_agent():

    llm