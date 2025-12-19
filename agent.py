import json
import logging
import os
from typing import List
from langchain_ibm import WatsonxLLM
from langgraph.graph import StateGraph
from pydantic import BaseModel
from dotenv import load_dotenv
from utils import search_deals

load_dotenv()

logging.basicConfig(level=logging.INFO)
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
        project_id=os.getenv("PROJECT_ID"),
        params={
            "max_new_tokens": 512,
            "temperature": 0.2,
            "top_p": 0.9
        }
    )

    graph = StateGraph(DealAgentState)

    def retrieve(state: DealAgentState) -> DealAgentState:
        results = search_deals(
            query=state.query,
            collection_name="ma_deals_knowledge",
            top_k=6
        )

        if not results:
            state.context = ["No relevant deal information was found."]
            return state

        state.context = []
        for r in results:
            meta = json.loads(r["metadata"])

            state.context.append(
                f"""
DEAL OVERVIEW:
- Acquirer: {meta.get("acquirer")}
- Target: {meta.get("target")}
- Sector: {meta.get("sector")}
- Region: {meta.get("region")}
- Year: {meta.get("year")}

IDENTIFIED RISKS:
{r["risks"]}

OBSERVED OUTCOME:
{r["outcome"]}
""".strip()
            )

        return state

    def answer(state: DealAgentState) -> DealAgentState:
        context = "\n\n".join(state.context)

        prompt = f"""
    You are a senior M&A integration advisor reviewing multiple completed acquisitions.

    Your objective is to synthesize insights across deals, not to summarize individual transactions.

    Using the evidence below:
    - Identify recurring integration risk patterns
    - Explicitly link those risks to observed post-merger outcomes
    - Derive clear, actionable lessons that would materially improve future M&A integrations

    Context:
    {context}

    Guidelines:
    - Base conclusions on patterns observed across multiple deals
    - When similar risks appear in different sectors, treat them as systemic
    - Write a structured analytical response with short paragraphs
    - Each lesson should clearly state: risk → consequence → lesson learned

    Write a complete analytical answer. Do not be overly brief.

    Answer:
    """
        state.answer = llm.invoke(prompt)
        return state

    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "answer")

    return graph.compile()
