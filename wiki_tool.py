"""MediaWiki retrieval backing the ``search_wikipedia`` tool.

Scope note: this is deliberately *not* a production search system. The brief asks
for focus on "prompt quality and eval design, not on building a production search
system", so retrieval here is simple, transparent, and honest about its limits
(top-N title search, plaintext extract, explicit truncation). The engineering
depth lives in ``evals/``.

Constraint compliance: no hosted search or RAG product is used anywhere. This
module talks to the public MediaWiki APIs over plain HTTP and nothing else.

Two endpoints, chosen deliberately:

* **Search** -- MediaWiki REST v1 ``/search/page``. Clean, modern, returns titles.
* **Content** -- Action API ``prop=extracts&explaintext``. Returns *clean prose*
  plus the provenance we need (``pageid``, ``lastrevid``, ``canonicalurl``).
  The REST v1 ``/page/{title}`` endpoint was rejected because it returns raw
  wikitext (templates, infoboxes, markup) that would need a lossy cleaner.

Verified behaviour worth knowing: the Action API caps *full-text* extracts at one
page per request. ``exlimit=max`` only batches when ``exintro`` is set, so asking
for three full articles in one call silently returns text for one and empty
strings for the rest. We therefore fetch one title per request.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

REST_SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
ACTION_API_URL = "https://en.wikipedia.org/w/api.php"

# Wikimedia's User-Agent policy requires a descriptive agent identifying the tool.
USER_AGENT = (
    "claude-wiki-eval-harness/1.0 "
    "(https://github.com/joelostlund/claude-wiki-eval-harness; take-home project)"
)

REQUEST_TIMEOUT = 10.0
MAX_RESULTS = 3
"""Articles returned per search. Fixed rather than model-controlled: the tool
signature in the brief is ``search_wikipedia(query: str)``, and letting the model
tune result count would dilute the thing we are actually grading (query
formulation)."""

CHAR_BUDGET = 4_000
"""Per-article character budget (~1k tokens). Real articles run far larger --
Nile is ~61k chars, Albert Einstein ~89k -- so truncation is mandatory. It is
announced in-band so the model re-queries instead of guessing across the gap."""

CACHE_DIR = Path(__file__).parent / ".wiki_cache"

_RETRY_STATUSES = {429, 500, 502, 503, 504}
_RETRY_DELAY = 1.0

_RATE_LIMIT = threading.Semaphore(3)
"""Politeness cap. The eval runner fans out across cases; without this we could
issue a burst of concurrent requests at Wikipedia from a single client."""


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class Document:
    """One retrieved article, with the provenance needed to verify a citation."""

    title: str
    url: str
    page_id: int
    revision_id: int
    text: str
    retrieved_at: str
    chars_total: int
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SearchResult:
    """Outcome of one ``search_wikipedia`` call. Never raises; failure is data."""

    query: str
    documents: list[Document] = field(default_factory=list)
    error: str | None = None
    latency_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.documents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "documents": [d.to_dict() for d in self.documents],
            "error": self.error,
            "latency_ms": self.latency_ms,
        }


InjectionMode = Literal["prepend", "append", "standalone_doc"]
INJECTION_MODES: frozenset[str] = frozenset({"prepend", "append", "standalone_doc"})
"""Runtime counterpart to ``InjectionMode``. ``Literal`` is erased at runtime,
so a mode read from JSON needs an explicit membership check or a typo silently
falls through to the append default."""


@dataclass
class Injection:
    """A simulated indirect prompt-injection payload.

    Wikipedia is world-editable, so text crafted to hijack a reader-agent is the
    *native* threat model for this system rather than a contrived add-on. Rather
    than vandalise a live article to test it, the eval harness splices an
    attacker-controlled payload into retrieved content.

    The payload lands **inside** ``<untrusted_content>``, which is what makes the
    attack genuinely *indirect*: it arrives through the retrieval channel, not
    from the user turn. Putting it in the user's message would be a *direct*
    injection and would test nothing about the RAG path.

    This parameter is harness-only. It is bound at eval time via
    ``functools.partial`` and is invisible to the model -- the tool schema the
    model sees exposes ``query`` and nothing else.
    """

    payload: str
    mode: InjectionMode = "append"
    doc_index: int = 0
    doc_title: str = "Wikipedia:Sandbox"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Fence hardening
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(r"<\s*/?\s*untrusted_content\b[^>]*>", re.IGNORECASE)
"""Matches the fence tag in any form an attacker might write it.

Deliberately permissive: an earlier pattern (``</?\\s*untrusted_content\\s*>``)
put the whitespace class *after* the optional slash, so ``< /untrusted_content>``
slipped through un-neutralised, as did any tag carrying attributes
(``</untrusted_content x>``). HTML parsers tolerate both, and so would a model
reading the block -- which is enough to forge the boundary from inside a
document. ``\\b[^>]*`` closes both gaps."""


def neutralize_fence(text: str) -> str:
    """Defang any literal ``<untrusted_content>`` tag appearing in article text.

    Without this, an attacker who can edit a Wikipedia page could write a literal
    ``</untrusted_content>`` into it and break out of the quarantine fence,
    making the rest of their payload look like trusted harness framing rather
    than retrieved data. Neutralising the delimiter keeps the boundary
    unforgeable from inside the document.
    """
    return _FENCE_RE.sub(lambda m: m.group(0).replace("<", "&lt;"), text)


def _attr(value: Any) -> str:
    """Escape a value for use in an XML attribute."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #


def _cache_path(key: str) -> Path:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    return CACHE_DIR / f"{digest}.json"


def _cache_get(key: str) -> dict[str, Any] | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cache_put(key: str, data: dict[str, Any]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "key": key,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data": data,
        }
        _cache_path(key).write_text(json.dumps(record), encoding="utf-8")
    except OSError:
        pass  # Cache is an optimisation; never fail a lookup because of it.


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def _http_get_json(url: str, params: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """GET JSON, returning ``(data, error)``. Never raises."""
    last_error: str | None = None

    for attempt in range(2):
        try:
            with _RATE_LIMIT:
                response = requests.get(
                    url,
                    params=params,
                    headers={"User-Agent": USER_AGENT},
                    timeout=REQUEST_TIMEOUT,
                )
        except requests.Timeout:
            last_error = f"Wikipedia request timed out after {REQUEST_TIMEOUT:.0f}s."
        except requests.RequestException as exc:
            last_error = f"Could not reach Wikipedia: {type(exc).__name__}."
        else:
            status = getattr(response, "status_code", None)
            if status == 200:
                try:
                    payload = response.json()
                except (ValueError, json.JSONDecodeError):
                    return None, "Wikipedia returned a malformed (non-JSON) response."
                if not isinstance(payload, dict):
                    return None, "Wikipedia returned an unexpected response shape."
                return payload, None
            if status in _RETRY_STATUSES:
                last_error = f"Wikipedia returned HTTP {status}."
            else:
                return None, f"Wikipedia returned HTTP {status}."

        if attempt == 0:
            time.sleep(_RETRY_DELAY)

    return None, last_error


# --------------------------------------------------------------------------- #
# Retrieval steps
# --------------------------------------------------------------------------- #


def _search_titles(
    query: str, limit: int, refresh_cache: bool
) -> tuple[list[str], str | None]:
    cache_key = f"search:{limit}:{query.lower()}"
    record = None if refresh_cache else _cache_get(cache_key)

    if record is None:
        data, error = _http_get_json(REST_SEARCH_URL, {"q": query, "limit": limit})
        if error:
            return [], error
        if not isinstance(data.get("pages"), list):
            return [], "Unexpected response shape from the Wikipedia search API."
        _cache_put(cache_key, data)
        payload = data
    else:
        payload = record.get("data", {})

    pages = payload.get("pages")
    if not isinstance(pages, list):
        return [], "Unexpected response shape from the Wikipedia search API."

    titles = [p["title"] for p in pages if isinstance(p, dict) and p.get("title")]
    return titles, None


def _fetch_page(title: str, refresh_cache: bool) -> tuple[Document | None, str | None]:
    cache_key = f"page:{title}"
    record = None if refresh_cache else _cache_get(cache_key)

    fresh = record is None
    if fresh:
        data, error = _http_get_json(
            ACTION_API_URL,
            {
                "action": "query",
                "format": "json",
                "formatversion": "2",
                "prop": "extracts|info",
                "explaintext": "1",
                "inprop": "url",
                "redirects": "1",
                "titles": title,
            },
        )
        if error:
            return None, error
        payload = data
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    else:
        payload = record.get("data", {})
        fetched_at = record.get("fetched_at", "")

    # Validate BEFORE caching. Caching first meant one bad 200 -- an empty
    # envelope during a replication blip, a rate-limit stub, a transient
    # `missing` -- was written to disk and then replayed forever, because the
    # cache has no TTL and every production call site leaves refresh_cache False.
    # The article would stay broken until .wiki_cache was deleted by hand.
    pages = (payload.get("query") or {}).get("pages")
    if not isinstance(pages, list) or not pages:
        return None, f"No article content returned for {title!r}."

    page = pages[0]
    if not isinstance(page, dict) or page.get("missing"):
        return None, f"Wikipedia has no article titled {title!r}."

    extract = page.get("extract") or ""
    if not extract.strip():
        return None, f"Article {title!r} returned no readable text."

    if fresh:
        _cache_put(cache_key, payload)

    text, truncated = _truncate(extract, CHAR_BUDGET)

    return (
        Document(
            title=page.get("title", title),
            url=page.get("canonicalurl") or page.get("fullurl") or "",
            page_id=int(page.get("pageid") or 0),
            revision_id=int(page.get("lastrevid") or 0),
            text=text,
            retrieved_at=fetched_at,
            chars_total=len(extract),
            truncated=truncated,
        ),
        None,
    )


def _truncate(text: str, budget: int) -> tuple[str, bool]:
    """Cut to ``budget`` chars, backing off to a paragraph or sentence boundary."""
    if len(text) <= budget:
        return text, False

    window = text[:budget]
    for boundary in ("\n\n", "\n", ". "):
        cut = window.rfind(boundary)
        if cut > budget * 0.6:  # only honour a boundary that isn't wastefully early
            return window[: cut + len(boundary)].rstrip(), True
    return window.rstrip(), True


def apply_injection(documents: list[Document], injection: Injection) -> list[Document]:
    """Splice a simulated attacker payload into retrieved content."""
    docs = list(documents)

    if injection.mode == "standalone_doc":
        slug = injection.doc_title.replace(" ", "_")
        poisoned = Document(
            title=injection.doc_title,
            url=f"https://en.wikipedia.org/wiki/{slug}",
            page_id=0,
            revision_id=0,
            text=injection.payload,
            retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            chars_total=len(injection.payload),
            truncated=False,
        )
        docs.insert(min(injection.doc_index, len(docs)), poisoned)
        return docs

    if not docs:
        return docs

    index = min(injection.doc_index, len(docs) - 1)
    target = docs[index]
    if injection.mode == "prepend":
        merged = f"{injection.payload}\n\n{target.text}"
    else:
        merged = f"{target.text}\n\n{injection.payload}"

    # chars_total must grow with the text, or format_for_model's
    # "omitted = chars_total - len(text)" goes wrong -- and on an already-truncated
    # article it goes *negative*, handing the model a nonsensical
    # "[TRUNCATED - -102 of 4,500 characters omitted]" right after the payload.
    # Growing it by exactly the inserted length keeps the reported omission equal
    # to the genuine one from the original article.
    docs[index] = Document(
        **{
            **target.to_dict(),
            "text": merged,
            "chars_total": target.chars_total + (len(merged) - len(target.text)),
        }
    )
    return docs


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def search_wikipedia(
    query: str,
    *,
    injection: Injection | None = None,
    max_results: int = MAX_RESULTS,
    refresh_cache: bool = False,
) -> SearchResult:
    """Search Wikipedia and retrieve article text. Never raises.

    Every failure mode -- timeout, rate limit, malformed payload, missing page --
    comes back as a ``SearchResult`` carrying an ``error`` string that is safe to
    hand straight to the model, so a transient network problem degrades into a
    recoverable tool result rather than a crashed agent loop.
    """
    started = time.monotonic()

    def elapsed() -> int:
        return int((time.monotonic() - started) * 1000)

    cleaned = (query or "").strip()
    if not cleaned:
        return SearchResult(
            query=cleaned,
            error="Empty query. Provide search terms.",
            latency_ms=elapsed(),
        )

    titles, error = _search_titles(cleaned, max_results, refresh_cache)
    if error:
        return SearchResult(query=cleaned, error=error, latency_ms=elapsed())

    documents: list[Document] = []
    failures: list[str] = []
    for title in titles[:max_results]:
        document, fetch_error = _fetch_page(title, refresh_cache)
        if document is not None:
            documents.append(document)
        elif fetch_error:
            failures.append(fetch_error)

    if not documents and failures:
        return SearchResult(query=cleaned, error=failures[0], latency_ms=elapsed())

    if injection is not None:
        documents = apply_injection(documents, injection)

    return SearchResult(query=cleaned, documents=documents, latency_ms=elapsed())


def format_for_model(result: SearchResult) -> str:
    """Render a ``SearchResult`` as the XML string handed back to the model.

    Article text is wrapped in ``<untrusted_content>`` so the trust boundary is
    unmistakable, and any literal fence tag inside the article is neutralised
    first so the boundary cannot be forged from within a document.
    """
    query_attr = _attr(result.query)

    if result.error:
        return (
            f'<search_results query="{query_attr}">\n'
            f"  <error>{_attr(result.error)}</error>\n"
            "  <guidance>The search failed. This is a retrieval problem, not a "
            "statement about whether the information exists. Retry with different "
            "terms, or tell the user the lookup failed - do not answer from memory."
            "</guidance>\n"
            "</search_results>"
        )

    if not result.documents:
        return (
            f'<search_results query="{query_attr}">\n'
            "  <no_results>No Wikipedia articles matched this query.</no_results>\n"
            "  <guidance>Try a distinctive noun phrase likely to be an article "
            "title. If a reformulated search also returns nothing, say the "
            "information could not be found on Wikipedia rather than answering "
            "from memory.</guidance>\n"
            "</search_results>"
        )

    blocks = [f'<search_results query="{query_attr}">']
    for index, doc in enumerate(result.documents, start=1):
        body = neutralize_fence(doc.text)
        if doc.truncated:
            omitted = doc.chars_total - len(doc.text)
            body += (
                f"\n\n[TRUNCATED - {omitted:,} of {doc.chars_total:,} characters "
                "omitted. If the information you need was cut off, search again "
                "with more specific terms.]"
            )
        blocks.append(
            f'  <document index="{index}" title="{_attr(doc.title)}" '
            f'url="{_attr(doc.url)}" page_id="{doc.page_id}" '
            f'revision_id="{doc.revision_id}" retrieved="{_attr(doc.retrieved_at)}">\n'
            "    <untrusted_content>\n"
            f"{body}\n"
            "    </untrusted_content>\n"
            "  </document>"
        )
    blocks.append("</search_results>")
    return "\n".join(blocks)


if __name__ == "__main__":  # Manual smoke check: python wiki_tool.py "Mount Fuji"
    import sys

    probe = sys.argv[1] if len(sys.argv) > 1 else "Mount Fuji"
    print(format_for_model(search_wikipedia(probe))[:2000])
