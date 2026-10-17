"""Answer generation: LangChain chain of prompt | model | parser over retrieved CV excerpts."""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langfuse import observe
from langfuse.langchain import CallbackHandler

from . import config
from .rag import Index

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a concise, honest career coach. Ground every answer in the CV excerpts below;
if the CV lacks something, say so instead of inventing it. When a job description is given,
tailor advice to it. Prefer concrete rewrites over generic tips.

CV excerpts:
{cv}

Job description:
{jd}"""),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

_chain = None


def chain():
    global _chain
    _chain = _chain or PROMPT | ChatOpenAI(model=config.CHAT_MODEL, temperature=config.TEMPERATURE) | StrOutputParser()
    return _chain


@observe(name="coach-answer")
def answer(question: str, index: Index, jd: str, history: list[dict]) -> str:
    inputs = {
        "cv": "\n---\n".join(index.search(question)),
        "jd": jd or "(none provided)",
        "history": [(m["role"], m["content"]) for m in history[-6:]],
        "question": question,
    }
    return chain().invoke(inputs, config={"callbacks": [CallbackHandler()]})
