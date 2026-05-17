# HTTP Proxy Support for Wikipedia Requests

**Date:** 2026-05-17

## Summary

Add HTTP proxy support for Wikipedia API requests in `ReActWikipedia` class. Model API calls (OpenAI, vLLM, etc.) will not use the proxy.

## Requirements

- Only Wikipedia and similar external HTTP resources go through proxy
- Model API requests (OpenAI, vLLM, Azure, Friendli) bypass proxy
- Proxy address: configurable via environment variable `HTTPS_PROXY` or `HTTP_PROXY`
- Default proxy: `http://127.0.0.1:10808`
- No behavior change when proxy is not configured

## Design

### 1. ReActWikipedia Class Changes

**File:** `src/docstore/wikipedia.py`

#### 1.1 Constructor

Add `proxy` parameter:

```python
def __init__(self, benchmark=False, skip_retry_when_postprocess=False, proxy=None) -> None:
    ...
    self.proxy = proxy
```

#### 1.2 Synchronous search() Method

Two `requests.get()` calls (lines 159, 167) need proxy support:

```python
proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None
response_text = requests.get(search_url, proxies=proxies).text
```

#### 1.3 Asynchronous asearch() Method

Two `session.get()` calls (lines 200-202, 210-212) need proxy support:

```python
async with aiohttp.ClientSession() as session:
    async with session.get(search_url, proxy=self.proxy) as response:
        response_text = await response.text()
```

### 2. Tool Configuration Changes

**Files:**
- `configs/hotpotqa/tools.py`
- `configs/hotpotqa_react/tools.py`
- `configs/movie/tools.py`
- `configs/movie_react/tools.py`
- `configs/parallelqa/tools.py`
- `configs/parallelqa_react/tools.py`

Read proxy from environment variable when creating `ReActWikipedia`:

```python
import os

proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
docstore = ReActWikipedia(proxy=proxy)
```

### 3. Usage

```bash
# With proxy
HTTPS_PROXY=http://127.0.0.1:10808 python run_llm_compiler.py --benchmark_name hotpotqa --store results.json

# Without proxy (default behavior)
python run_llm_compiler.py --benchmark_name hotpotqa --store results.json
```

## Files to Modify

| File | Change |
|------|--------|
| `src/docstore/wikipedia.py` | Add `proxy` parameter to `__init__`, use in `search()` and `asearch()` |
| `configs/hotpotqa/tools.py` | Pass proxy to `ReActWikipedia` |
| `configs/hotpotqa_react/tools.py` | Pass proxy to `ReActWikipedia` |
| `configs/movie/tools.py` | Pass proxy to `ReActWikipedia` |
| `configs/movie_react/tools.py` | Pass proxy to `ReActWikipedia` |
| `configs/parallelqa/tools.py` | Pass proxy to `ReActWikipedia` |
| `configs/parallelqa_react/tools.py` | Pass proxy to `ReActWikipedia` |

## Testing

1. Run without proxy — should work as before
2. Run with proxy — Wikipedia requests should go through proxy
3. Verify model API calls still work directly (no proxy)
