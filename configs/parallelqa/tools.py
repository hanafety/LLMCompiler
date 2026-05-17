import os

from src.agents.tools import Tool
from src.docstore.wikipedia import DocstoreExplorer, ReActWikipedia

# Initialize with proxy from environment
_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
web_searcher = ReActWikipedia(proxy=_proxy)
docstore = DocstoreExplorer(web_searcher)

tools = [
    Tool(
        name="search",
        func=docstore.asearch,
        description=(
            "search(entity: str) -> str:\n"
            " - Executes an exact search for the entity on Wikipedia.\n"
            " - Returns the first paragraph if the entity is found.\n"
            " - `entity`: entity to search for on Wikipedia, e.g., Mount Everest, cheetah, San Francisco, etc."
        ),
        stringify_rule=lambda args: f"search({args[0]})",
    ),
]
