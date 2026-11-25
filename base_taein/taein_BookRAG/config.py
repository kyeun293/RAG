"""Paths and hyper-parameters for the BookRAG-style pipeline on ewha.pdf."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(HERE, "..", "..", "ewha.pdf")   # shared course file
INDEX_DIR = os.path.join(HERE, "index")

# --- models (assignment: Solar Mini 4 for every LLM call) -------------------
CHAT_MODEL = "solar-mini4-260922"
EMBED_PASSAGE_MODEL = "solar-embedding-2-passage"
EMBED_QUERY_MODEL = "solar-embedding-2-query"
DOC_PARSE_MODEL = "document-parse"          # replaces MinerU (layout parsing)
DOC_PARSE_MODE = "enhanced"                 # complex merged-cell tables (별표)
MAX_TOKENS = 4096                           # assignment upper bound
TEMPERATURE = 0
REASONING_EFFORT = "none"
MAX_LLM_CALLS_PER_QUESTION = 5              # assignment limit (enforced)

# --- document layout ---------------------------------------------------------
# Pages (1-based) that contain the appendix tables (별표). Only these pages are
# sent to Document Parse; body text is taken from the PDF text layer as-is.
TABLE_PAGES = range(39, 55)
# Pages where Document Parse (enhanced) reads the table better than the text
# layer: 별표 1 (merged-cell admission quota tables). 별표 2/3 are letter-spaced
# lists that the text layer keeps intact, so they use the PDF text.
DOCPARSE_PAGES = range(39, 51)

# --- graph construction --------------------------------------------------------
# KG variants (ablation):
#   full  : Solar extracts entities + type + description + relations (~45-55 calls)
#   names : Solar extracts entity names only; relations = co-occurrence (~6 calls)
#   rule  : no LLM; titles, definitions, tf-idf terms; co-occurrence (0 calls)
KG_MODES = ("full", "names", "rule")
KG_BATCH_CHARS = 1800        # full: node text per call (long JSON gets truncated)
KG_BATCH_CHARS_NAMES = 9000  # names: short output, so much larger batches
ER_TOP_K = 5                 # candidates for entity resolution
ER_MIN_SIM = 0.86            # absolute cosine floor for a merge
ER_GRADIENT = 0.6            # BookRAG gradient threshold g (relative drop)

# --- retrieval -------------------------------------------------------------------
ENTITY_MATCH_SIM = 0.80      # query entity -> KG entity
DENSE_POOL = 20              # always add the top-N dense nodes as candidates
PPR_ALPHA = 0.85             # personalized PageRank damping
MIN_CONTEXT_NODES = 4
MAX_CONTEXT_NODES = 12
MAX_CONTEXT_CHARS = 14000
