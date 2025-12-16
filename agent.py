import logging, re
import os
from typing import List
from langchain_ibm import WatsonxLLM
from langgraph.graph import StateGraph
from pydantic import BaseModel
from dotenv import load_dotenv
from utils import search_deals

load_dotenv()
env = os.environ.get("ENV")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DealAgentState(BaseModel):
    query: str
    context: List[str] = []
    answer: str = ""


def build_agent():
    llm = WatsonxLLM(
        model_id="ibm/granite-4-h-small",
        url=os.getenv("WML_URL"),
        apikey=os.getenv("WML_APIKEY"),
        project_id=os.getenv("PROJECT_ID")
    )

    graph = StateGraph(DealAgentState)

    def retrieve(state: DealAgentState) -> DealAgentState:
        results = search_deals(
            query=state.query,
            collection_name="ma_deals_knowledge",
            top_k=5,
            document_type="risks"
        )

        if not results:
            state.context = ["No relevant deal information was found."]
            return state

        state.context = [
            f"DEAL CONTEXT:\n{r['metadata']}\n{r['risks'] or r['summary']}"
            for r in results
        ]
        return state

    def answer(state: DealAgentState) -> DealAgentState:
        context = "\n\n".join(state.context)

        prompt = f"""
        You are an experienced M&A analyst.

        Analyze the information below and answer the question strictly based on it.

        Context:
        {context}

        Task:
        - Identify recurring risks across deals
        - Extract concrete lessons learned from outcomes
        - Focus on cross-border or integration-related issues when applicable

        Rules:
        - Do not repeat instructions
        - Do not describe the task
        - Do not request a JSON format
        - Write a direct analytical answer in plain text
        - If information is insufficient, say: "Insufficient evidence in the dataset."

        Answer:
        """

        state.answer = llm.invoke(prompt)
        return state

    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "answer")

    return graph.compile()
