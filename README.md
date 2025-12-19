# M&A Deal Intelligence — RAG-based Analytical System
## Overview

This project implements a Retrieval-Augmented Generation (RAG) system designed to analyze historical M&A deals and extract evidence-based insights about integration risks, regulatory challenges, and post-merger outcomes.
The system combines vector search (Milvus) with LLM-based analytical reasoning (IBM Granite) to answer complex, qualitative questions such as:
What patterns emerge in deals with regulatory issues?
Which integration risks most often lead to delayed value realization?
How do cross-border constraints affect post-merger performance?
The focus is on grounded analysis: all conclusions are strictly derived from retrieved deal evidence, minimizing hallucination.


## Key Features
* Semantic retrieval over structured M&A deal data
* Explicit separation of:
* * Deal summaries
* * Pre-acquisition risks
* * Post-merger outcomes
* Evidence-driven analytical responses
* Emphasis on integration execution and regulatory impact

## Tech Stack
* Python 3.10+
* Milvus (vector database)
* Sentence Transformers (all-MiniLM-L6-v2)
* LangGraph (agent orchestration)
* IBM Granite (Watsonx)
* Pydantic
* python-dotenv

## Design Principles
* Retrieval-first, reasoning-second
* Strict grounding in retrieved evidence
* Clear linkage between risks and outcomes
* Analytical depth over generic Q&A responses

## Project Scope
**This project was developed as a portfolio and learning initiative, with emphasis on:**
* RAG system design
* Vector search correctness and reliability
* LLM-based analytical reasoning
* Enterprise-style data modeling

**Future extensions (not implemented) could include:**
* Knowledge graph integration
* Multi-hop retrieval
* Automated evaluation pipelines

## _**Disclaimer**_
All M&A deal data used in this project is synthetic and anonymized, created exclusively for educational and portfolio purposes.
This project does not represent real financial analysis or investment advice.