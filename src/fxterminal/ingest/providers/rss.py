"""Generic RSS/Atom document provider (news + central-bank feeds).

Used for general FX news AND as the central-bank document ingester: CB press
feeds (Fed monetary-press RSS, BoJ whatsnew) are classified by title keyword
into statement/minutes/speech. feedparser is a synchronous library; parsing is
CPU-bound and fast enough for our volumes — fetch is async via FetchClient.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

import feedparser

from fxterminal.core.logging import get_logger
from fxterminal.ingest.base import FetchClient, ProviderError
from fxterminal.ingest.types import DocType, RawDoc

logger = get_logger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(html: str) -> str:
    """Cheap HTML -> text for feed summaries (keeps raw payload deterministic)."""
    return _WS_RE.sub(" ", _TAG_RE.sub(" ", html or "")).strip()


def classify_cb_doc(title: str) -> DocType:
    """Keyword classifier for central-bank press items.

    Conservative: only obvious statement/minutes/speech titles get typed;
    everything else stays 'news' so downstream steps never mis-file a release.
    """
    t = title.lower()
    if "minutes" in t:
        return "minutes"
    if "statement" in t or "monetary policy decision" in t or "policy decision" in t:
        return "statement"
    if "speech" in t or "remarks" in t or "testimony" in t or "lecture" in t:
        return "speech"
    return "news"


# Ordered most-specific first — 'australian dollar' must beat 'dollar'.
_CURRENCY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("AUD", ("australian dollar", "aussie", "audusd", "aud")),
    ("NZD", ("new zealand dollar", "kiwi", "nzdusd", "nzd")),
    ("CHF", ("swiss franc", "swiss national bank", "usdchf", "chf")),
    ("EUR", ("euro", "european central bank", "eurusd", "ecb", "eur")),
    ("JPY", ("bank of japan", "yen", "usdjpy", "boj", "jpy")),
    ("GBP", ("bank of england", "sterling", "pound", "gbpusd", "cable", "boe", "gbp")),
    ("USD", ("u.s. dollar", "us dollar", "federal reserve", "fomc", "usd", "dollar")),
)


def _tag_currency(title: str) -> str | None:
    """Best-effort currency tag for general-news items (no spec.currency).

    Returns the first currency whose keyword appears in the title. Short
    tokens match on word boundaries only ('gbp' won't fire inside 'giblet').
    """
    t = title.lower()
    for ccy, keywords in _CURRENCY_KEYWORDS:
        for kw in keywords:
            if len(kw) <= 4:
                if re.search(rf"\b{re.escape(kw)}\b", t):
                    return ccy
            elif kw in t:
                return ccy
    return None


def _entry_time(entry: feedparser.FeedParserDict) -> datetime | None:
    for attr in ("published_parsed", "updated_parsed"):
        st = getattr(entry, attr, None)
        if st:
            return datetime(*st[:6], tzinfo=UTC)
    return None


@dataclass(frozen=True)
class FeedSpec:
    """One configured feed. `doc_type=None` means classify by title keywords."""

    url: str
    source: str
    doc_type: DocType | None = None
    central_bank: str | None = None
    currency: str | None = None


# Default feeds. Verified reachable 200/XML 2026-10-01:
#   federalreserve.gov/feeds/press_monetary.xml, boj.or.jp/en/rss/whatsnew.xml,
#   ecb.europa.eu/rss/press.xml, bankofengland.co.uk/rss/news,
#   snb.ch/public/rss/en/news, investing.com feeds (x2).
# Confirmed UNREACHABLE (HTTP 403 to any agent incl. browser UA, 2026-10-01):
#   rba.gov.au/rss/rss-cb-media-releases.xml, rbnz.govt.nz — bot filtering;
#   no public machine-readable source. AUD/NZD docs must come via extra_feeds.
# Licensing: headlines/summaries fine for personal research; full-text
# redistribution in a product needs permission from each publisher.
DEFAULT_FEEDS: tuple[FeedSpec, ...] = (
    FeedSpec(
        url="https://www.federalreserve.gov/feeds/press_monetary.xml",
        source="fed_rss",
        doc_type=None,  # Fed mixes statements/minutes/speeches in this feed
        central_bank="FED",
        currency="USD",
    ),
    FeedSpec(
        url="https://www.boj.or.jp/en/rss/whatsnew.xml",
        source="boj_rss",
        doc_type=None,
        central_bank="BOJ",
        currency="JPY",
    ),
    FeedSpec(
        url="https://www.ecb.europa.eu/rss/press.xml",
        source="ecb_rss",
        doc_type=None,
        central_bank="ECB",
        currency="EUR",
    ),
    FeedSpec(
        url="https://www.bankofengland.co.uk/rss/news",
        source="boe_rss",
        doc_type=None,
        central_bank="BOE",
        currency="GBP",
    ),
    FeedSpec(
        # verified 2026-10-01 (was /public/en/rss/news -> 301 here)
        url="https://www.snb.ch/public/rss/en/news",
        source="snb_rss",
        doc_type=None,
        central_bank="SNB",
        currency="CHF",
    ),
    # RBA + RBNZ: no accessible feed — both sites return HTTP 403 to non-browser
    # agents (bot filtering). Their docs can still arrive via extra_feeds once a
    # reachable source is found; the pipeline just logs a provider error, never
    # blocks the cycle.
    #
    # General market news (no fixed currency/bank — tagged per-item by title
    # keywords via _tag_currency so theme extraction can use them):
    FeedSpec(
        url="https://www.investing.com/rss/news_25.rss",  # verified 2026-10-01
        source="investing_com_markets",
        doc_type="news",
    ),
    FeedSpec(
        url="https://www.investing.com/rss/news_285.rss",  # verified 2026-10-01
        source="investing_com_top",
        doc_type="news",
    ),
)


class RssDocumentProvider:
    """Fetches one RSS/Atom feed and emits RawDocs."""

    def __init__(self, client: FetchClient, spec: FeedSpec, max_items: int = 50) -> None:
        self._client = client
        self._spec = spec
        self._max_items = max_items

    @property
    def name(self) -> str:
        return self._spec.source

    async def fetch_documents(self) -> list[RawDoc]:
        spec = self._spec
        text = await self._client.get_text(spec.url)
        feed = feedparser.parse(text)
        if feed.bozo and not feed.entries:
            raise ProviderError(f"{spec.source}: unparseable feed ({feed.bozo_exception})")

        docs: list[RawDoc] = []
        for entry in feed.entries[: self._max_items]:
            published = _entry_time(entry)
            if published is None:
                logger.warning("rss_no_date", source=spec.source, title=entry.get("title", ""))
                continue
            title = strip_html(entry.get("title", ""))
            body = strip_html(
                entry.get("content", [{}])[0].get("value", "")
                if entry.get("content")
                else entry.get("summary", "")
            )
            doc_type = spec.doc_type or classify_cb_doc(title)
            docs.append(
                RawDoc(
                    source=spec.source,
                    doc_type=doc_type,
                    title=title,
                    published_at=published,
                    raw_text=body or title,
                    url=entry.get("link"),
                    central_bank=spec.central_bank,
                    # fixed-tag feeds keep theirs; general feeds get per-item tagging
                    currency=spec.currency or _tag_currency(title),
                    meta={"feed_url": spec.url, "entry_id": entry.get("id")},
                )
            )
        return docs
