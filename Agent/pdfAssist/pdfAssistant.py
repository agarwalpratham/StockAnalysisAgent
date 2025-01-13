import typer
from typing import Optional,List
from phi.assistant import Assistant
from rich.prompt import Prompt
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.lancedb import LanceDb
from phi.agent import Agent
from phi.vectordb.search import SearchType
from phi.model.groq import Groq

import os
from dotenv import load_dotenv
load_dotenv()

# os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
vector_db = LanceDb(
    table_name="recipes",
    uri="/tmp/lancedb",
    search_type=SearchType.keyword,
)

knowledge_base=PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=vector_db
)

knowledge_base.load(recreate=True)


def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    agent = Agent(
        model=Groq(id="llama-3.1-70b-versatile"),
        run_id=run_id,
        user_id=user,
        knowledge=knowledge_base,
        show_tool_calls=True,
        debug_mode=True,
    )

    if run_id is None:
        run_id = agent.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in ("exit", "bye"):
            break
        agent.print_response(message)

if __name__=="__main__":
    typer.run(pdf_assistant)
