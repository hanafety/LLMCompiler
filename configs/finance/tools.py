"""Finance benchmark tools.

Reuses parallelqa tools (search + math) for financial data queries.
"""

from configs.parallelqa.tools import tools, generate_tools

__all__ = ["tools", "generate_tools"]
