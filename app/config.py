"""Application configuration.

Settings are read from environment variables at process start and cached. The
layer stays deliberately small and dependency-free (no settings library) so the
values and their defaults are easy to audit.

OCR settings (RAG-005) are centralised here so thresholds and the language/DPI
configuration are never scattered through parser code.
"""

from __future__ import annotations

import os
from functools import lru_cache

from pydantic import BaseModel, ConfigDict

DEFAULT_APP_NAME = "PhD Academic RAG"
DEFAULT_APP_VERSION = "0.1.0"
DEFAULT_ENVIRONMENT = "development"
DEFAULT_API_PREFIX = "/api"

# --- OCR defaults (RAG-005) ---
DEFAULT_OCR_ENABLED = True
DEFAULT_OCR_LANGUAGE = "eng"
DEFAULT_OCR_DPI = 300
DEFAULT_OCR_MIN_NATIVE_CHARS = 50

# --- Metadata enrichment defaults (RAG-006) ---
DEFAULT_METADATA_ENRICH_EXTERNAL = True
DEFAULT_METADATA_CACHE_PATH = ".metadata_cache/cache.json"
DEFAULT_METADATA_REQUEST_TIMEOUT = 10.0
DEFAULT_METADATA_MAX_RETRIES = 2
DEFAULT_METADATA_MIN_REQUEST_INTERVAL = 1.0

# --- Embedding defaults (RAG-008) ---
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
#: 16 is the measured throughput optimum for BGE-M3 on this corpus: identical
#: speed to 32/64/128 (132.5 embeddings/s steady state) at ~4 GiB reserved VRAM,
#: so it leaves ~28 GiB of headroom on a 32 GiB RTX 5090 (RAG-008 GPU benchmark).
DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_EMBEDDING_CACHE_PATH = ".embedding_cache/embeddings.npz"
DEFAULT_EMBEDDING_LOCAL_FILES_ONLY = False

# --- Qdrant / indexing defaults (RAG-008) ---
DEFAULT_QDRANT_COLLECTION = "phd_rag_chunks_v1"
DEFAULT_QDRANT_PATH = ".qdrant_data"
DEFAULT_QDRANT_UPLOAD_BATCH_SIZE = 128
DEFAULT_INDEX_MANIFEST_DIR = ".index_cache/manifests"

# --- Reranking defaults (RAG-009) ---
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
#: 16 is the measured optimum for a 25-candidate query (86 pairs/s vs 78 at batch 8
#: and 66 at batch 32) at ~4.9 GiB reserved VRAM, so it leaves ~27 GiB of headroom
#: on a 32 GiB RTX 5090 (RAG-009 reranker benchmark).
DEFAULT_RERANKER_BATCH_SIZE = 16

# --- Retrieval defaults (RAG-009) ---
DEFAULT_RETRIEVAL_CANDIDATE_K = 25
DEFAULT_RETRIEVAL_FINAL_K = 8
DEFAULT_RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT = 3
DEFAULT_RETRIEVAL_DUPLICATE_OVERLAP = 0.85
DEFAULT_RETRIEVAL_MAX_ROUNDS = 3
DEFAULT_RETRIEVAL_MODE = "evidence"
#: Withhold chunks that are reference lists but were not classified as REFERENCES
#: upstream (RAG-009 bibliography guard).
DEFAULT_RETRIEVAL_EXCLUDE_BIBLIOGRAPHY = True

# --- Discovery defaults (RAG-010) ---
#: External academic discovery is opt-in per request and never required for local RAG.
DEFAULT_DISCOVERY_ENABLED = True
DEFAULT_DISCOVERY_CACHE_PATH = ".discovery_cache/cache.json"
#: Searches move slowly; a day of caching avoids hammering the provider budgets.
DEFAULT_DISCOVERY_CACHE_TTL = 86400.0
DEFAULT_DISCOVERY_REQUEST_TIMEOUT = 15.0
DEFAULT_DISCOVERY_MAX_RETRIES = 2
DEFAULT_DISCOVERY_BACKOFF_BASE = 1.5
#: Politely spaced requests; keeps the shared anonymous provider budgets alive.
DEFAULT_DISCOVERY_MIN_REQUEST_INTERVAL = 1.0
DEFAULT_DISCOVERY_PROVIDERS = ("crossref", "openalex")
DEFAULT_DISCOVERY_MAX_RESULTS = 20
#: Upper bound on a downloaded open-access PDF used for identity verification.
DEFAULT_DISCOVERY_FULLTEXT_MAX_BYTES = 25 * 1024 * 1024
DEFAULT_DISCOVERY_FULLTEXT_TIMEOUT = 30.0

# --- Generation defaults (RAG-011) ---
#: Local generation model served by Ollama (34.7B MoE, Q4_K_M).
DEFAULT_GENERATION_MODEL = "qwen3.5:35b"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
#: Low temperature keeps grounded answers close to the evidence and repeatable.
DEFAULT_GENERATION_TEMPERATURE = 0.1
DEFAULT_GENERATION_TOP_P = 0.9
#: Context window requested from Ollama. The measured load at 8192 used 23.1 GiB of a
#: 32 GiB RTX 5090, so there is room for the KV cache without evicting the model.
DEFAULT_GENERATION_NUM_CTX = 8192
#: A long academic answer; the evaluation measures concise/normal/detailed explicitly.
DEFAULT_GENERATION_MAX_OUTPUT_TOKENS = 900
#: Wall-clock ceiling for one generation. A 35B model on this GPU produces ~150 tok/s,
#: so this is generous rather than tight.
DEFAULT_GENERATION_TIMEOUT = 300.0
#: Evidence context budget in estimated tokens. Kept below num_ctx minus the answer.
DEFAULT_GENERATION_CONTEXT_TOKENS = 6000
#: Tokens held back from the evidence when the caller asks for my_work/guidance context,
#: so an explicit request is not starved by a large evidence set.
DEFAULT_GENERATION_RESERVED_NON_EVIDENCE_TOKENS = 1500
#: Characters per token used by the deterministic estimator (calibrated in the report).
DEFAULT_GENERATION_CHARS_PER_TOKEN = 4.0
#: Reasoning output is off: the answer must be the answer, not a thought process.
DEFAULT_GENERATION_THINK = False
#: Fixed sampling seed, so the same prompt gives the same answer on the same build.
DEFAULT_GENERATION_SEED = 0
#: Drop evidence below this rerank score before generation; None keeps everything
#: (RAG-009 showed weak tail scores, so the recommendation is made from measurement).
DEFAULT_GENERATION_MIN_RERANK_SCORE: float | None = None
#: Ollama keeps the model resident between calls; this avoids a reload per question.
DEFAULT_GENERATION_KEEP_ALIVE = "30m"

# --- Simple chat defaults (RAG-015) ---
#: The simple-chat path is *not* the RAG-011 evidence path. Evidence generation asks for
#: 8192 tokens because a top-8 chunk package is small; a working conversation with attached
#: documents needs the model's real window. The workstation serves qwen3.5:35b with a
#: 262144-token context, so that is what a simple-chat request asks Ollama for.
DEFAULT_CHAT_NUM_CTX = 262144
#: Input budget for one simple-chat request. Deliberately below the window: the answer
#: (``DEFAULT_CHAT_MAX_OUTPUT_TOKENS``) and a safety margin must still fit, and a request
#: that tries to fill all 262144 tokens on every turn wastes prefill for no benefit.
DEFAULT_CHAT_CONTEXT_TOKENS = 236_544
#: Room kept for the answer inside the window. 4096 is a long answer for conversation (the
#: RAG-011 evidence path allows 900); asking for 8192 made the model run far past the point
#: where an answer had ended, which is where a degenerate continuation becomes more likely.
DEFAULT_CHAT_MAX_OUTPUT_TOKENS = 4096
#: Safety margin subtracted from the window when validating the input budget, so the
#: configured budgets can never add up to more than Ollama was asked to allocate.
DEFAULT_CHAT_CONTEXT_MARGIN_TOKENS = 4096
#: One automatic retry when Ollama aborts a prediction with its own repeat guard
#: ("prediction aborted, token repeat limit reached", RAG-015 FIX1). Ollama's runner can end
#: up in a state where it emits one token forever; a fresh runner is the only recovery, and
#: an ordinary conversation deserves one second chance before the error reaches the user.
DEFAULT_CHAT_RETRY_ON_REPEAT_ABORT = True
#: Shares of the input budget offered, in the documented priority order, to conversation
#: history, attached working documents and (only when requested) corpus material. They are
#: caps applied in order, not reservations: unclaimed budget stays available to the next
#: priority, and any remainder is handed back to truncated attachments.
DEFAULT_CHAT_HISTORY_SHARE = 0.25
DEFAULT_CHAT_ATTACHMENT_SHARE = 0.60
DEFAULT_CHAT_CORPUS_SHARE = 0.25
#: The largest single message the simple-chat path accepts. The RAG-014 mode API stops at
#: 20 000 characters because a mode query is a paragraph; a chat message is not (the user
#: may paste a chapter). This is a transport-sanity bound, not a context bound.
DEFAULT_CHAT_MAX_MESSAGE_CHARS = 2_000_000
#: Attachments. Extensions are fixed here so the UI, the API and the tests agree.
DEFAULT_CHAT_ALLOWED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")
DEFAULT_CHAT_MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024
DEFAULT_CHAT_MAX_ATTACHMENTS_PER_CHAT = 20
#: Local storage. Deliberately outside ``data/literature`` and ``data/my_work``: a chat
#: attachment is a temporary working document, never corpus material.
DEFAULT_CHAT_DB_PATH = "data/chat/chat.sqlite3"
DEFAULT_CHAT_ATTACHMENT_DIR = "data/chat/attachments"
#: Characters of the first user message used as an automatic chat title.
DEFAULT_CHAT_TITLE_CHARS = 60

# --- Cloud generation provider (RAG-016) ---
#: DeepSeek is a *manual* alternative for the simple chat only: it answers a turn when the
#: user selects it for that turn, and nothing ever falls back to it on its own. The chat
#: still lives in the local SQLite database and the local corpus is still the source of
#: truth; only the context selected for one turn leaves the machine.
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
#: The two DeepSeek models this build offers, in selector order: the default first. Model ids
#: are configuration, never application logic (RAG-016 §5): ``PHD_RAG_DEEPSEEK_MODELS`` adds
#: more. ``deepseek-flash`` is the everyday model; ``deepseek-v4-pro`` is selected by hand when
#: it is wanted (RAG-016 FIX1).
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
DEFAULT_DEEPSEEK_MODELS: tuple[str, ...] = ("deepseek-v4-pro",)
#: Wall-clock ceiling for one cloud answer.
DEFAULT_DEEPSEEK_TIMEOUT = 300.0
#: The provider's window for the current models (1M tokens). This is a *ceiling*, not a target:
#: the input budget below is what a turn actually sends, so removing the old 128K window value
#: does not make any request larger (RAG-016 FIX1).
DEFAULT_DEEPSEEK_NUM_CTX = 1_000_000
#: Input budget for one cloud turn. Deliberately unchanged by RAG-016 FIX1: a conversation does
#: not send a million tokens just because the window allows it.
DEFAULT_DEEPSEEK_CONTEXT_TOKENS = 96_000
DEFAULT_DEEPSEEK_MAX_OUTPUT_TOKENS = 4_096


class Settings(BaseModel):
    """Runtime configuration for the backend.

    Fields are sourced from ``PHD_RAG_*`` environment variables, falling back to
    safe development defaults.
    """

    model_config = ConfigDict(frozen=True)

    app_name: str = DEFAULT_APP_NAME
    app_version: str = DEFAULT_APP_VERSION
    environment: str = DEFAULT_ENVIRONMENT
    debug: bool = False
    api_prefix: str = DEFAULT_API_PREFIX

    # OCR (RAG-005)
    ocr_enabled: bool = DEFAULT_OCR_ENABLED
    ocr_language: str = DEFAULT_OCR_LANGUAGE
    ocr_dpi: int = DEFAULT_OCR_DPI
    ocr_min_native_chars: int = DEFAULT_OCR_MIN_NATIVE_CHARS

    # Metadata enrichment (RAG-006)
    metadata_enrich_external: bool = DEFAULT_METADATA_ENRICH_EXTERNAL
    metadata_cache_path: str = DEFAULT_METADATA_CACHE_PATH
    metadata_request_timeout: float = DEFAULT_METADATA_REQUEST_TIMEOUT
    metadata_max_retries: int = DEFAULT_METADATA_MAX_RETRIES
    metadata_min_request_interval: float = DEFAULT_METADATA_MIN_REQUEST_INTERVAL
    metadata_crossref_mailto: str | None = None

    # Embeddings (RAG-008)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_device: str | None = None
    embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE
    embedding_cache_path: str = DEFAULT_EMBEDDING_CACHE_PATH
    embedding_local_files_only: bool = DEFAULT_EMBEDDING_LOCAL_FILES_ONLY

    # Qdrant / indexing (RAG-008)
    qdrant_collection: str = DEFAULT_QDRANT_COLLECTION
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: str = DEFAULT_QDRANT_PATH
    qdrant_upload_batch_size: int = DEFAULT_QDRANT_UPLOAD_BATCH_SIZE
    index_manifest_dir: str = DEFAULT_INDEX_MANIFEST_DIR

    # Reranking (RAG-009)
    reranker_model: str = DEFAULT_RERANKER_MODEL
    reranker_device: str | None = None
    reranker_batch_size: int = DEFAULT_RERANKER_BATCH_SIZE

    # Retrieval (RAG-009)
    retrieval_mode: str = DEFAULT_RETRIEVAL_MODE
    retrieval_candidate_k: int = DEFAULT_RETRIEVAL_CANDIDATE_K
    retrieval_final_k: int = DEFAULT_RETRIEVAL_FINAL_K
    retrieval_max_chunks_per_document: int = DEFAULT_RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT
    retrieval_duplicate_overlap: float = DEFAULT_RETRIEVAL_DUPLICATE_OVERLAP
    retrieval_max_rounds: int = DEFAULT_RETRIEVAL_MAX_ROUNDS
    retrieval_exclude_bibliography: bool = DEFAULT_RETRIEVAL_EXCLUDE_BIBLIOGRAPHY

    # Discovery (RAG-010)
    discovery_enabled: bool = DEFAULT_DISCOVERY_ENABLED
    discovery_cache_path: str = DEFAULT_DISCOVERY_CACHE_PATH
    discovery_cache_ttl: float = DEFAULT_DISCOVERY_CACHE_TTL
    discovery_cache_enabled: bool = True
    discovery_request_timeout: float = DEFAULT_DISCOVERY_REQUEST_TIMEOUT
    discovery_max_retries: int = DEFAULT_DISCOVERY_MAX_RETRIES
    discovery_backoff_base: float = DEFAULT_DISCOVERY_BACKOFF_BASE
    discovery_min_request_interval: float = DEFAULT_DISCOVERY_MIN_REQUEST_INTERVAL
    discovery_providers: tuple[str, ...] = DEFAULT_DISCOVERY_PROVIDERS
    discovery_max_results: int = DEFAULT_DISCOVERY_MAX_RESULTS
    discovery_fulltext_max_bytes: int = DEFAULT_DISCOVERY_FULLTEXT_MAX_BYTES
    discovery_fulltext_timeout: float = DEFAULT_DISCOVERY_FULLTEXT_TIMEOUT
    #: Contact identities. Never hard-code an address: these are configured values,
    #: reused for OpenAlex ``mailto`` and the polite Crossref user agent (RAG-010 §30).
    openalex_email: str | None = None
    openalex_api_key: str | None = None

    # Generation (RAG-011)
    generation_model: str = DEFAULT_GENERATION_MODEL
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    generation_temperature: float = DEFAULT_GENERATION_TEMPERATURE
    generation_top_p: float = DEFAULT_GENERATION_TOP_P
    generation_num_ctx: int = DEFAULT_GENERATION_NUM_CTX
    generation_max_output_tokens: int = DEFAULT_GENERATION_MAX_OUTPUT_TOKENS
    generation_timeout: float = DEFAULT_GENERATION_TIMEOUT
    generation_context_tokens: int = DEFAULT_GENERATION_CONTEXT_TOKENS
    generation_reserved_non_evidence_tokens: int = DEFAULT_GENERATION_RESERVED_NON_EVIDENCE_TOKENS
    generation_chars_per_token: float = DEFAULT_GENERATION_CHARS_PER_TOKEN
    generation_think: bool = DEFAULT_GENERATION_THINK
    generation_seed: int = DEFAULT_GENERATION_SEED
    generation_min_rerank_score: float | None = DEFAULT_GENERATION_MIN_RERANK_SCORE
    generation_keep_alive: str = DEFAULT_GENERATION_KEEP_ALIVE

    # Simple chat (RAG-015)
    chat_num_ctx: int = DEFAULT_CHAT_NUM_CTX
    chat_context_tokens: int = DEFAULT_CHAT_CONTEXT_TOKENS
    chat_max_output_tokens: int = DEFAULT_CHAT_MAX_OUTPUT_TOKENS
    chat_context_margin_tokens: int = DEFAULT_CHAT_CONTEXT_MARGIN_TOKENS
    chat_retry_on_repeat_abort: bool = DEFAULT_CHAT_RETRY_ON_REPEAT_ABORT
    chat_history_share: float = DEFAULT_CHAT_HISTORY_SHARE
    chat_attachment_share: float = DEFAULT_CHAT_ATTACHMENT_SHARE
    chat_corpus_share: float = DEFAULT_CHAT_CORPUS_SHARE
    chat_max_message_chars: int = DEFAULT_CHAT_MAX_MESSAGE_CHARS
    chat_allowed_extensions: tuple[str, ...] = DEFAULT_CHAT_ALLOWED_EXTENSIONS
    chat_max_attachment_bytes: int = DEFAULT_CHAT_MAX_ATTACHMENT_BYTES
    chat_max_attachments_per_chat: int = DEFAULT_CHAT_MAX_ATTACHMENTS_PER_CHAT
    chat_db_path: str = DEFAULT_CHAT_DB_PATH
    chat_attachment_dir: str = DEFAULT_CHAT_ATTACHMENT_DIR
    chat_title_chars: int = DEFAULT_CHAT_TITLE_CHARS

    # Cloud generation provider (RAG-016). The key is server-side only: it is never stored,
    # never serialised by the API and never written to a log or an error message.
    deepseek_api_key: str | None = None
    deepseek_base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    deepseek_model: str = DEFAULT_DEEPSEEK_MODEL
    deepseek_models: tuple[str, ...] = DEFAULT_DEEPSEEK_MODELS
    deepseek_timeout: float = DEFAULT_DEEPSEEK_TIMEOUT
    deepseek_num_ctx: int = DEFAULT_DEEPSEEK_NUM_CTX
    deepseek_context_tokens: int = DEFAULT_DEEPSEEK_CONTEXT_TOKENS
    deepseek_max_output_tokens: int = DEFAULT_DEEPSEEK_MAX_OUTPUT_TOKENS


def _parse_bool(value: str) -> bool:
    """Interpret common truthy strings from the environment."""
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv(value: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    """Parse a comma-separated environment value into a tuple."""
    if not value:
        return default
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or default


def _parse_optional_float(value: str | None) -> float | None:
    """Parse an optional float; an empty or unset value means "no threshold"."""
    if value is None or not value.strip():
        return None
    return float(value)


def load_settings() -> Settings:
    """Build a :class:`Settings` instance from the current environment."""
    return Settings(
        app_name=os.getenv("PHD_RAG_APP_NAME", DEFAULT_APP_NAME),
        app_version=os.getenv("PHD_RAG_APP_VERSION", DEFAULT_APP_VERSION),
        environment=os.getenv("PHD_RAG_ENVIRONMENT", DEFAULT_ENVIRONMENT),
        debug=_parse_bool(os.getenv("PHD_RAG_DEBUG", "false")),
        api_prefix=os.getenv("PHD_RAG_API_PREFIX", DEFAULT_API_PREFIX),
        ocr_enabled=_parse_bool(os.getenv("PHD_RAG_OCR_ENABLED", "true")),
        ocr_language=os.getenv("PHD_RAG_OCR_LANGUAGE", DEFAULT_OCR_LANGUAGE),
        ocr_dpi=int(os.getenv("PHD_RAG_OCR_DPI", str(DEFAULT_OCR_DPI))),
        ocr_min_native_chars=int(
            os.getenv("PHD_RAG_OCR_MIN_NATIVE_CHARS", str(DEFAULT_OCR_MIN_NATIVE_CHARS))
        ),
        metadata_enrich_external=_parse_bool(
            os.getenv("PHD_RAG_METADATA_ENRICH_EXTERNAL", "true")
        ),
        metadata_cache_path=os.getenv(
            "PHD_RAG_METADATA_CACHE_PATH", DEFAULT_METADATA_CACHE_PATH
        ),
        metadata_request_timeout=float(
            os.getenv("PHD_RAG_METADATA_REQUEST_TIMEOUT", str(DEFAULT_METADATA_REQUEST_TIMEOUT))
        ),
        metadata_max_retries=int(
            os.getenv("PHD_RAG_METADATA_MAX_RETRIES", str(DEFAULT_METADATA_MAX_RETRIES))
        ),
        metadata_min_request_interval=float(
            os.getenv(
                "PHD_RAG_METADATA_MIN_REQUEST_INTERVAL",
                str(DEFAULT_METADATA_MIN_REQUEST_INTERVAL),
            )
        ),
        metadata_crossref_mailto=(
            os.getenv("PHD_RAG_CROSSREF_MAILTO")
            or os.getenv("PHD_RAG_METADATA_CROSSREF_MAILTO")
            or None
        ),
        embedding_model=os.getenv("PHD_RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        embedding_device=os.getenv("PHD_RAG_EMBEDDING_DEVICE") or None,
        embedding_batch_size=int(
            os.getenv("PHD_RAG_EMBEDDING_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE))
        ),
        embedding_cache_path=os.getenv(
            "PHD_RAG_EMBEDDING_CACHE_PATH", DEFAULT_EMBEDDING_CACHE_PATH
        ),
        embedding_local_files_only=_parse_bool(
            os.getenv("PHD_RAG_EMBEDDING_LOCAL_FILES_ONLY", "false")
        ),
        qdrant_collection=os.getenv("PHD_RAG_QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION),
        qdrant_url=os.getenv("PHD_RAG_QDRANT_URL") or None,
        qdrant_api_key=os.getenv("PHD_RAG_QDRANT_API_KEY") or None,
        qdrant_path=os.getenv("PHD_RAG_QDRANT_PATH", DEFAULT_QDRANT_PATH),
        qdrant_upload_batch_size=int(
            os.getenv(
                "PHD_RAG_QDRANT_UPLOAD_BATCH_SIZE", str(DEFAULT_QDRANT_UPLOAD_BATCH_SIZE)
            )
        ),
        index_manifest_dir=os.getenv("PHD_RAG_INDEX_MANIFEST_DIR", DEFAULT_INDEX_MANIFEST_DIR),
        reranker_model=os.getenv("PHD_RAG_RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
        reranker_device=os.getenv("PHD_RAG_RERANKER_DEVICE") or None,
        reranker_batch_size=int(
            os.getenv("PHD_RAG_RERANKER_BATCH_SIZE", str(DEFAULT_RERANKER_BATCH_SIZE))
        ),
        retrieval_mode=os.getenv("PHD_RAG_RETRIEVAL_MODE", DEFAULT_RETRIEVAL_MODE),
        retrieval_candidate_k=int(
            os.getenv("PHD_RAG_RETRIEVAL_CANDIDATE_K", str(DEFAULT_RETRIEVAL_CANDIDATE_K))
        ),
        retrieval_final_k=int(
            os.getenv("PHD_RAG_RETRIEVAL_FINAL_K", str(DEFAULT_RETRIEVAL_FINAL_K))
        ),
        retrieval_max_chunks_per_document=int(
            os.getenv(
                "PHD_RAG_RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT",
                str(DEFAULT_RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT),
            )
        ),
        retrieval_duplicate_overlap=float(
            os.getenv(
                "PHD_RAG_RETRIEVAL_DUPLICATE_OVERLAP",
                str(DEFAULT_RETRIEVAL_DUPLICATE_OVERLAP),
            )
        ),
        retrieval_max_rounds=int(
            os.getenv("PHD_RAG_RETRIEVAL_MAX_ROUNDS", str(DEFAULT_RETRIEVAL_MAX_ROUNDS))
        ),
        retrieval_exclude_bibliography=_parse_bool(
            os.getenv(
                "PHD_RAG_RETRIEVAL_EXCLUDE_BIBLIOGRAPHY",
                "true" if DEFAULT_RETRIEVAL_EXCLUDE_BIBLIOGRAPHY else "false",
            )
        ),
        discovery_enabled=_parse_bool(os.getenv("PHD_RAG_DISCOVERY_ENABLED", "true")),
        discovery_cache_path=os.getenv(
            "PHD_RAG_DISCOVERY_CACHE_PATH", DEFAULT_DISCOVERY_CACHE_PATH
        ),
        discovery_cache_ttl=float(
            os.getenv("PHD_RAG_DISCOVERY_CACHE_TTL", str(DEFAULT_DISCOVERY_CACHE_TTL))
        ),
        discovery_cache_enabled=_parse_bool(
            os.getenv("PHD_RAG_DISCOVERY_CACHE_ENABLED", "true")
        ),
        discovery_request_timeout=float(
            os.getenv(
                "PHD_RAG_DISCOVERY_REQUEST_TIMEOUT", str(DEFAULT_DISCOVERY_REQUEST_TIMEOUT)
            )
        ),
        discovery_max_retries=int(
            os.getenv("PHD_RAG_DISCOVERY_MAX_RETRIES", str(DEFAULT_DISCOVERY_MAX_RETRIES))
        ),
        discovery_backoff_base=float(
            os.getenv("PHD_RAG_DISCOVERY_BACKOFF_BASE", str(DEFAULT_DISCOVERY_BACKOFF_BASE))
        ),
        discovery_min_request_interval=float(
            os.getenv(
                "PHD_RAG_DISCOVERY_MIN_REQUEST_INTERVAL",
                str(DEFAULT_DISCOVERY_MIN_REQUEST_INTERVAL),
            )
        ),
        discovery_providers=_parse_csv(
            os.getenv("PHD_RAG_DISCOVERY_PROVIDERS"), DEFAULT_DISCOVERY_PROVIDERS
        ),
        discovery_max_results=int(
            os.getenv("PHD_RAG_DISCOVERY_MAX_RESULTS", str(DEFAULT_DISCOVERY_MAX_RESULTS))
        ),
        discovery_fulltext_max_bytes=int(
            os.getenv(
                "PHD_RAG_DISCOVERY_FULLTEXT_MAX_BYTES",
                str(DEFAULT_DISCOVERY_FULLTEXT_MAX_BYTES),
            )
        ),
        discovery_fulltext_timeout=float(
            os.getenv(
                "PHD_RAG_DISCOVERY_FULLTEXT_TIMEOUT", str(DEFAULT_DISCOVERY_FULLTEXT_TIMEOUT)
            )
        ),
        openalex_email=(
            os.getenv("PHD_RAG_OPENALEX_EMAIL")
            or os.getenv("PHD_RAG_CROSSREF_MAILTO")
            or os.getenv("PHD_RAG_METADATA_CROSSREF_MAILTO")
            or None
        ),
        openalex_api_key=os.getenv("PHD_RAG_OPENALEX_API_KEY") or None,
        generation_model=os.getenv("PHD_RAG_GENERATION_MODEL", DEFAULT_GENERATION_MODEL),
        ollama_base_url=os.getenv("PHD_RAG_OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/"),
        generation_temperature=float(
            os.getenv("PHD_RAG_GENERATION_TEMPERATURE", str(DEFAULT_GENERATION_TEMPERATURE))
        ),
        generation_top_p=float(
            os.getenv("PHD_RAG_GENERATION_TOP_P", str(DEFAULT_GENERATION_TOP_P))
        ),
        generation_num_ctx=int(
            os.getenv("PHD_RAG_GENERATION_NUM_CTX", str(DEFAULT_GENERATION_NUM_CTX))
        ),
        generation_max_output_tokens=int(
            os.getenv(
                "PHD_RAG_GENERATION_MAX_OUTPUT_TOKENS",
                str(DEFAULT_GENERATION_MAX_OUTPUT_TOKENS),
            )
        ),
        generation_timeout=float(
            os.getenv("PHD_RAG_GENERATION_TIMEOUT", str(DEFAULT_GENERATION_TIMEOUT))
        ),
        generation_context_tokens=int(
            os.getenv(
                "PHD_RAG_GENERATION_CONTEXT_TOKENS", str(DEFAULT_GENERATION_CONTEXT_TOKENS)
            )
        ),
        generation_reserved_non_evidence_tokens=int(
            os.getenv(
                "PHD_RAG_GENERATION_RESERVED_NON_EVIDENCE_TOKENS",
                str(DEFAULT_GENERATION_RESERVED_NON_EVIDENCE_TOKENS),
            )
        ),
        generation_chars_per_token=float(
            os.getenv(
                "PHD_RAG_GENERATION_CHARS_PER_TOKEN", str(DEFAULT_GENERATION_CHARS_PER_TOKEN)
            )
        ),
        generation_think=_parse_bool(
            os.getenv("PHD_RAG_GENERATION_THINK", "true" if DEFAULT_GENERATION_THINK else "false")
        ),
        generation_seed=int(os.getenv("PHD_RAG_GENERATION_SEED", str(DEFAULT_GENERATION_SEED))),
        generation_min_rerank_score=_parse_optional_float(
            os.getenv("PHD_RAG_GENERATION_MIN_RERANK_SCORE")
        ),
        generation_keep_alive=os.getenv("PHD_RAG_GENERATION_KEEP_ALIVE", DEFAULT_GENERATION_KEEP_ALIVE),
        chat_num_ctx=int(os.getenv("PHD_RAG_CHAT_NUM_CTX", str(DEFAULT_CHAT_NUM_CTX))),
        chat_context_tokens=int(
            os.getenv("PHD_RAG_CHAT_CONTEXT_TOKENS", str(DEFAULT_CHAT_CONTEXT_TOKENS))
        ),
        chat_max_output_tokens=int(
            os.getenv("PHD_RAG_CHAT_MAX_OUTPUT_TOKENS", str(DEFAULT_CHAT_MAX_OUTPUT_TOKENS))
        ),
        chat_context_margin_tokens=int(
            os.getenv(
                "PHD_RAG_CHAT_CONTEXT_MARGIN_TOKENS", str(DEFAULT_CHAT_CONTEXT_MARGIN_TOKENS)
            )
        ),
        chat_retry_on_repeat_abort=_parse_bool(
            os.getenv(
                "PHD_RAG_CHAT_RETRY_ON_REPEAT_ABORT",
                "true" if DEFAULT_CHAT_RETRY_ON_REPEAT_ABORT else "false",
            )
        ),
        chat_history_share=float(
            os.getenv("PHD_RAG_CHAT_HISTORY_SHARE", str(DEFAULT_CHAT_HISTORY_SHARE))
        ),
        chat_attachment_share=float(
            os.getenv("PHD_RAG_CHAT_ATTACHMENT_SHARE", str(DEFAULT_CHAT_ATTACHMENT_SHARE))
        ),
        chat_corpus_share=float(
            os.getenv("PHD_RAG_CHAT_CORPUS_SHARE", str(DEFAULT_CHAT_CORPUS_SHARE))
        ),
        chat_max_message_chars=int(
            os.getenv("PHD_RAG_CHAT_MAX_MESSAGE_CHARS", str(DEFAULT_CHAT_MAX_MESSAGE_CHARS))
        ),
        chat_allowed_extensions=_parse_csv(
            os.getenv("PHD_RAG_CHAT_ALLOWED_EXTENSIONS"), DEFAULT_CHAT_ALLOWED_EXTENSIONS
        ),
        chat_max_attachment_bytes=int(
            os.getenv(
                "PHD_RAG_CHAT_MAX_ATTACHMENT_BYTES", str(DEFAULT_CHAT_MAX_ATTACHMENT_BYTES)
            )
        ),
        chat_max_attachments_per_chat=int(
            os.getenv(
                "PHD_RAG_CHAT_MAX_ATTACHMENTS_PER_CHAT",
                str(DEFAULT_CHAT_MAX_ATTACHMENTS_PER_CHAT),
            )
        ),
        chat_db_path=os.getenv("PHD_RAG_CHAT_DB_PATH", DEFAULT_CHAT_DB_PATH),
        chat_attachment_dir=os.getenv(
            "PHD_RAG_CHAT_ATTACHMENT_DIR", DEFAULT_CHAT_ATTACHMENT_DIR
        ),
        chat_title_chars=int(os.getenv("PHD_RAG_CHAT_TITLE_CHARS", str(DEFAULT_CHAT_TITLE_CHARS))),
        # The RAG-016 documentation uses the provider's own variable names. Both spellings are
        # read so the documented names work and the project's ``PHD_RAG_`` prefix still wins.
        deepseek_api_key=(
            os.getenv("PHD_RAG_DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or None
        ),
        deepseek_base_url=(
            os.getenv("PHD_RAG_DEEPSEEK_BASE_URL")
            or os.getenv("DEEPSEEK_BASE_URL")
            or DEFAULT_DEEPSEEK_BASE_URL
        ).rstrip("/"),
        deepseek_model=(
            os.getenv("PHD_RAG_DEEPSEEK_MODEL") or os.getenv("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
        ),
        deepseek_models=_parse_csv(
            os.getenv("PHD_RAG_DEEPSEEK_MODELS") or os.getenv("DEEPSEEK_MODELS"),
            DEFAULT_DEEPSEEK_MODELS,
        ),
        deepseek_timeout=float(
            os.getenv("PHD_RAG_DEEPSEEK_TIMEOUT", str(DEFAULT_DEEPSEEK_TIMEOUT))
        ),
        deepseek_num_ctx=int(
            os.getenv("PHD_RAG_DEEPSEEK_NUM_CTX", str(DEFAULT_DEEPSEEK_NUM_CTX))
        ),
        deepseek_context_tokens=int(
            os.getenv("PHD_RAG_DEEPSEEK_CONTEXT_TOKENS", str(DEFAULT_DEEPSEEK_CONTEXT_TOKENS))
        ),
        deepseek_max_output_tokens=int(
            os.getenv(
                "PHD_RAG_DEEPSEEK_MAX_OUTPUT_TOKENS", str(DEFAULT_DEEPSEEK_MAX_OUTPUT_TOKENS)
            )
        ),
    )


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached settings instance."""
    return load_settings()
