# HTTP Proxy Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add HTTP proxy support for Wikipedia API requests in ReActWikipedia class while keeping model API calls direct.

**Architecture:** Add a `proxy` parameter to `ReActWikipedia.__init__()` and use it in both synchronous (`requests.get`) and asynchronous (`aiohttp.ClientSession`) HTTP calls. Tool configuration files read proxy from environment variable `HTTPS_PROXY` or `HTTP_PROXY` and pass it to `ReActWikipedia`.

**Tech Stack:** Python, requests, aiohttp

---

## Task 1: Add proxy parameter to ReActWikipedia class

**Files:**
- Modify: `src/docstore/wikipedia.py`

- [ ] **Step 1: Add `proxy` parameter to `__init__`**

Modify the `__init__` method at line 25 to add the `proxy` parameter:

```python
def __init__(self, benchmark=False, skip_retry_when_postprocess=False, proxy=None) -> None:
    """Check that wikipedia package is installed."""
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "Could not import wikipedia python package. "
            "Please install it with `pip install wikipedia`."
        )
    self.page = None
    self.lookup_keyword = None
    self.lookup_list = None
    self.lookup_cnt = None

    self.benchmark = benchmark
    self.all_times = []

    # when True, always skip retry when postprocess
    self.skip_retry_when_postprocess = skip_retry_when_postprocess

    # HTTP proxy for Wikipedia requests
    self.proxy = proxy
```

- [ ] **Step 2: Add proxy support to synchronous `search()` method**

Modify the first `requests.get()` call at line 159:

```python
        s = time.time()
        entity = str(entity)
        entity_ = entity.replace(" ", "+")
        search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
        proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None
        response_text = requests.get(search_url, proxies=proxies).text
```

Modify the second `requests.get()` call at line 167:

```python
            alternative = self._get_alternative(result)
            entity_ = alternative.replace(" ", "+")
            search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
            response_text = requests.get(search_url, proxies=proxies).text
```

- [ ] **Step 3: Add proxy support to asynchronous `asearch()` method**

Modify the first `session.get()` call at lines 200-202:

```python
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, proxy=self.proxy) as response:
                response_text = await response.text()
```

Modify the second `session.get()` call at lines 210-212:

```python
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, proxy=self.proxy) as response:
                    response_text = await response.text()
```

- [ ] **Step 4: Commit the changes**

```bash
git add src/docstore/wikipedia.py
git commit -m "feat: add HTTP proxy support to ReActWikipedia

- Add proxy parameter to __init__
- Use proxy in synchronous search() method via requests.get(proxies=...)
- Use proxy in asynchronous asearch() method via aiohttp session.get(proxy=...)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 2: Update hotpotqa tools to pass proxy

**Files:**
- Modify: `configs/hotpotqa/tools.py`

- [ ] **Step 1: Add import and read proxy from environment**

Modify the file to add `os` import and read proxy:

```python
import os

from src.agents.tools import Tool
from src.docstore.wikipedia import DocstoreExplorer, ReActWikipedia


def generate_tools(args):
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    web_searcher = ReActWikipedia(proxy=proxy)
    if args.model_type == "vllm":
        # If we use LLaMA with vLLM for the movie recommendation task,
        # we frequently get the context length error, so we limit the
        # wikipedia context length to 400 and only return the first sentence.
        docstore = DocstoreExplorer(web_searcher, char_limit=400, one_sentence=True)
    else:
        docstore = DocstoreExplorer(web_searcher)

    tools = [
        Tool(
            name="search",
            func=docstore.asearch,
            description=(
                "search(entity: str) -> str:\n"
                " - Executes an exact search for the entity on Wikipedia.\n"
                " - Returns the first paragraph if the entity is found.\n"
            ),
            stringify_rule=lambda args: f"search({args[0]})",
        ),
    ]
    return tools
```

- [ ] **Step 2: Commit the changes**

```bash
git add configs/hotpotqa/tools.py
git commit -m "feat: pass HTTP proxy to ReActWikipedia in hotpotqa tools

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 3: Update hotpotqa_react tools to pass proxy

**Files:**
- Modify: `configs/hotpotqa_react/tools.py`

- [ ] **Step 1: Add import and read proxy from environment**

Modify the file:

```python
import os

from src.agents.tools import Tool
from src.docstore.wikipedia import DocstoreExplorer, ReActWikipedia


def generate_tools(args):
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    web_searcher = ReActWikipedia(proxy=proxy)
    if args.model_type == "vllm":
        # If we use LLaMA with vLLM for the movie recommendation task,
        # we frequently get the context length error, so we limit the
        # wikipedia context length to 400 and only return the first sentence.
        docstore = DocstoreExplorer(web_searcher, char_limit=400, one_sentence=True)
    else:
        docstore = DocstoreExplorer(web_searcher)

    tools = [
        Tool(
            name="Search",
            func=docstore.search,
            # NOTE: This description is not used
            description="useful for when you need to ask with search",
        ),
    ]
    return tools
```

- [ ] **Step 2: Commit the changes**

```bash
git add configs/hotpotqa_react/tools.py
git commit -m "feat: pass HTTP proxy to ReActWikipedia in hotpotqa_react tools

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 4: Update movie tools to pass proxy

**Files:**
- Modify: `configs/movie/tools.py`

- [ ] **Step 1: Add import and read proxy from environment**

Modify the file (note: `web_searcher` and `docstore` are module-level variables):

```python
import os

from src.agents.tools import Tool
from src.chains.llm_math_chain import LLMMathChain
from src.docstore.wikipedia import DocstoreExplorer, ReActWikipedia
from src.utils.model_utils import get_model

_MATH_DESCRIPTION = (
    "math(problem: str, context: Optional[list[str]]) -> float:\n"
    " - Solves the provided math problem.\n"
    ' - `problem` can be either a simple math problem (e.g. "1 + 3") or a word problem (e.g. "how many apples are there if there are 3 apples and 2 apples").\n'
    " - You cannot calculate multiple expressions in one call. For instance, `math('1 + 3, 2 + 4')` does not work. "
    "If you need to calculate multiple expressions, you need to call them separately like `math('1 + 3')` and then `math('2 + 4')`\n"
    " - Minimize the number of `math` actions as much as possible. For instance, instead of calling "
    '2. math("what is the 10% of $1") and then call 3. math("$1 + $2"), '
    'you MUST call 2. math("what is the 110% of $1") instead, which will reduce the number of math actions.\n'
    # Context specific rules below
    " - You can optionally provide a list of strings as `context` to help the agent solve the problem. "
    "If there are multiple contexts you need to answer the question, you can provide them as a list of strings.\n"
    " - `math` action will not see the output of the previous actions unless you provide it as `context`. "
    "You MUST provide the output of the previous actions as `context` if you need to do math on it.\n"
    " - You MUST NEVER provide `search` action's outputs as a variable in the `problem` argument. "
    "This is because `search` returns a text blob that contains the information about the entity, not a number or value. "
    "Therefore, when you need to provide an output of `search` action, you MUST provide it as a `context` argument to `math` action. "
    'For example, 1. search("Barack Obama") and then 2. math("age of $1") is NEVER allowed. '
    'Use 2. math("age of Barack Obama, context=["$1"]) instead.\n'
    " - When you ask a question about `context`, specify the units. "
    'For instance, "what is xx in height?" or "what is xx in millions?" instead of "what is xx?"\n'
)


def run_llm_math_chain_factory(llm_math_chain):
    async def run_llm_math_chain(question, context=None):
        if context is None:
            prompt = question
        else:
            if len(context) == 1:
                context_str = f"Context:\n{context[0]}"
            else:
                context_strs = []
                for i, c in enumerate(context):
                    context_strs.append(f"Context {i}:\n{c}")
                context_str = "\n\n".join(context_strs)
            prompt = (
                "Answer the Question based on the Context. When you write down a expression, it MUST ONLY consists of numbers and operators. "
                "Here are some guidelines that you will be PANALIZED if you don't follow:\n\n"
                "  - When you are asked for differences, you consider the absolute value of the difference. Difference of two numbers is always positive."
                "For instance, the difference between 1 and 2 is 1, not -1.\n"
                "  - When you are applying operations (e.g. difference, summation, ratio, etc.) between multiple values in the Context, you must unify the units of those numbers. "
                "For instance, you cannot add 1 meter to 1 foot.\n"
                "     - You must pick the values in the same units if all the values are available in the same units.\n"
                "     - If not, you must convert them to the same units before applying the operation.\n"
                "  - You MUST strictly follow the unit (e.g. meter, kilometer, million, etc.) you were asked.\n"
                "     - If the Context has the numbers in same units as the question, you can directly use them.\n"
                "     - If the Context has the numbers in different units than the question, you must convert them to the units asked in the question."
                "For example, if the question asks for the distance between two cities in kilometers, but the Context has the distance in miles, "
                "you must convert the distance to kilometers.\n"
                "  - If you are asked about a particular number in millions, billions, or any other unit, the number should be written without specifying the unit. "
                "For example, if you are asked for 100 millions, it should be written as 100, not 100 million or 100,000,000.\n"
                ' - Never introduce a variable. For instance "gazelle_max_speed * 1.4" is not allowed. Pick up a correct number from the given context.\n'
                "\n"
                f"{context_str}\n\n"
                f"Question: {question}\n\n"
            )
        response = llm_math_chain.run(prompt)
        response = response.split("Answer:")[1].strip()
        try:
            response = float(response)
            # round to 3 decimal places
            response = round(response, 3)
            response = str(response)
        except:
            pass
        return response

    return run_llm_math_chain


# Initialize with proxy from environment
_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
web_searcher = ReActWikipedia(proxy=_proxy)
docstore = DocstoreExplorer(web_searcher)


def generate_tools(args, model_name):
    llm_math_chain = get_model(
        model_type=args.model_type,
        api_key=args.api_key,
        vllm_port=args.vllm_port,
        stream=False,
        temperature=0,
    )
    llm_math_chain = LLMMathChain.from_llm(llm=llm_math_chain, verbose=True)
    return [
        Tool(
            name="search",
            func=docstore.asearch,
            description=(
                "search(entity: str) -> str:\n"
                " - Executes an exact search for the entity on Wikipedia.\n"
                " - Returns the first paragraph if the entity is found.\n"
            ),
            stringify_rule=lambda args: f"search({args[0]})",
        ),
        Tool(
            name="math",
            func=run_llm_math_chain_factory(llm_math_chain),
            description=_MATH_DESCRIPTION,
            stringify_rule=lambda args: f"math({args[0]})",  # drop context
        ),
    ]
```

- [ ] **Step 2: Commit the changes**

```bash
git add configs/movie/tools.py
git commit -m "feat: pass HTTP proxy to ReActWikipedia in movie tools

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 5: Update movie_react tools to pass proxy

**Files:**
- Modify: `configs/movie_react/tools.py`

- [ ] **Step 1: Add import and read proxy from environment**

Modify the file:

```python
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
```

- [ ] **Step 2: Commit the changes**

```bash
git add configs/movie_react/tools.py
git commit -m "feat: pass HTTP proxy to ReActWikipedia in movie_react tools

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 6: Update parallelqa tools to pass proxy

**Files:**
- Modify: `configs/parallelqa/tools.py`

- [ ] **Step 1: Add import and read proxy from environment**

Modify the file:

```python
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
```

- [ ] **Step 2: Commit the changes**

```bash
git add configs/parallelqa/tools.py
git commit -m "feat: pass HTTP proxy to ReActWikipedia in parallelqa tools

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Task 7: Update parallelqa_react tools to pass proxy

**Files:**
- Modify: `configs/parallelqa_react/tools.py`

- [ ] **Step 1: Add import and read proxy from environment**

Modify the file:

```python
import os

from src.agents.tools import Tool
from src.chains.llm_math_chain import LLMMathChain
from src.docstore.wikipedia import DocstoreExplorer, ReActWikipedia
from src.utils.model_utils import get_model


def run_llm_math_chain_factory(llm_math_chain):
    def run_llm_math_chain(args):
        # since llm math chain returns the answer with a prefix, we need to remove it
        try:
            response = llm_math_chain.run(args)
            try:
                r = response.split("Answer:")[-1]
                r = r.strip()
                r = float(r)
                # round to 3 decimal places
                r = round(r, 3)
                response = "Answer: " + str(r)
            except:
                pass
        except:
            response = "Error: Invalid expression. Try with a different expression."
        return response

    return run_llm_math_chain


# Initialize with proxy from environment
_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
web_searcher = ReActWikipedia(proxy=_proxy)
docstore = DocstoreExplorer(web_searcher)


def generate_tools(args, model_name):
    llm_math_chain = get_model(
        model_type=args.model_type,
        model_name=model_name,
        vllm_port=args.vllm_port,
        stream=False,
        temperature=0,
    )
    llm_math_chain = LLMMathChain.from_llm(llm=llm_math_chain, verbose=True)
    return [
        Tool(
            name="Search",
            func=docstore.search,
            description="",  # not used
        ),
        Tool(
            name="Calculate",
            func=run_llm_math_chain_factory(llm_math_chain),
            description="",  # not used
        ),
    ]
```

- [ ] **Step 2: Commit the changes**

```bash
git add configs/parallelqa_react/tools.py
git commit -m "feat: pass HTTP proxy to ReActWikipedia in parallelqa_react tools

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Usage

After implementation, use the proxy as follows:

```bash
# With proxy
HTTPS_PROXY=http://127.0.0.1:10808 python run_llm_compiler.py --benchmark_name hotpotqa --store results.json

# Without proxy (default behavior unchanged)
python run_llm_compiler.py --benchmark_name hotpotqa --store results.json
```
