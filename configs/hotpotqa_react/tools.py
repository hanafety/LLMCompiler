import os

from src.agents.tools import Tool
from src.docstore.wikipedia import DocstoreExplorer, ReActWikipedia

# Initialize with proxy from environment
_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
web_searcher = ReActWikipedia(proxy=_proxy)
docstore = DocstoreExplorer(web_searcher)

tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description=("useful for when you need to ask with search"),
    ),
]
