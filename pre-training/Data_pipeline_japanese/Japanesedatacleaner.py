"""
CSV-Aware Japanese Text Cleaning Pipeline with xxHash Deduplication

Features:
- Proper CSV parsing (handles id,text format)
- xxHash deduplication (ultra-fast, 100k-300k lines/sec)
- Enhanced quality filters for forum/web content
- Sentence completeness checking
- Cross-file deduplication with checkpointing
- Production-ready error handling and logging

Installation:
    pip install xxhash

Author: Santosh
Version: 2.0
"""

import re
import unicodedata
import pickle
import os
import csv
import sys
import logging
from pathlib import Path
from typing import Set, Optional, List, Dict, Tuple
from dataclasses import dataclass, field

# Hash library imports with fallback
try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    import hashlib
    logging.warning("⚠ xxhash not available, using hashlib fallback")
    logging.warning("  Install for 2-5x speedup: pip install xxhash")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PRE-COMPILED REGEX PATTERNS (for performance)
# ============================================================================

class RegexPatterns:
    """Pre-compiled regex patterns for better performance"""
    
    # Cleaning
    HTML_TAGS = re.compile(r'<[^>]+>')
    HTML_ENTITIES_NAMED = re.compile(r'&[a-zA-Z]+;')
    HTML_ENTITIES_DEC = re.compile(r'&#\d+;')
    HTML_ENTITIES_HEX = re.compile(r'&#x[0-9a-fA-F]+;')
    URLS_HTTP = re.compile(r'https?://[^\s　]+')
    URLS_FTP = re.compile(r'ftp://[^\s]+')
    URLS_WWW = re.compile(r'www\.[^\s]+')
    EMAILS = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    # Emojis
    EMOJIS = re.compile(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        r"\U0001F1E0-\U0001F1FF\U0001F900-\U0001F9FF]+",
        flags=re.UNICODE
    )
    
    # Special symbols
    EMOTICONS_JP = re.compile(r'\(´・ω・\)|\(゜ー゜\*\)|\(´・ω・`\)')
    SYMBOLS_COMMON = re.compile(r'[★☆♪♥♡←→↑↓⇒⇔∀∴※]')
    SYMBOLS_SHAPES = re.compile(r'[◆◇■□▲△▼▽○●◎◯⊿]')
    SYMBOLS_NUMBERS = re.compile(r'[①②③④⑤⑥⑦⑧⑨⑩]')
    REPEATED_EQUALS = re.compile(r'[=]{3,}')
    REPEATED_DASHES = re.compile(r'[-]{3,}')
    REPEATED_STARS = re.compile(r'[*]{3,}')
    
    # Quality filters
    KATAKANA = re.compile(r'[\u30A0-\u30FF]')
    HIRAGANA = re.compile(r'[\u3040-\u309F]')
    KANJI = re.compile(r'[\u4E00-\u9FFF]')
    JAPANESE_CHARS = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
    REPETITIVE_CHARS = re.compile(r'(.)\1{5,}')
    PUNCTUATION_JP = re.compile(r'[。、！？…]')
    NUMBERS = re.compile(r'[0-9]')
    URL_FRAGMENTS = re.compile(r'(http|www|\.com|\.jp){2,}', re.IGNORECASE)
    
    # Forum artifacts
    QUOTE_MARKERS = re.compile(r'^>+')
    INTERJECTIONS = re.compile(r'^(おいおい|まぁ|やっぱり|ぶっちゃけ|てか|つーか)')
    NAVIGATION = re.compile(r'(こちら|詳細|→|>>|Click|More|続きを読む|もっと見る)')
    BOILERPLATE = re.compile(
        r'(Copyright|©|®|™|All rights|利用規約|プライバシー|Cookie)', 
        re.IGNORECASE
    )
    
    # Sentence structure
    SENTENCE_ENDINGS = re.compile(r'[。！？」』)]')
    PARTICLES = re.compile(r'[はがをにへとでやのもより]')
    
    # Punctuation
    PERIODS = re.compile(r'[．.]')
    COMMAS = re.compile(r'[，,]')
    EXCLAMATIONS = re.compile(r'[!！]{2,}')
    QUESTIONS = re.compile(r'[?？]{2,}')
    ELLIPSIS_DOTS = re.compile(r'\.{3,}')
    ELLIPSIS_MARU = re.compile(r'。{3,}')
    QUOTES_OPEN = re.compile(r'["""„]')
    QUOTES_CLOSE = re.compile(r'["""‟]')
    PUNCT_SPACE = re.compile(r'([。！？])([A-Za-z0-9])')
    
    # Whitespace
    WHITESPACE_AROUND_PUNCT = re.compile(r'\s*([。、！？])\s*')
    WHITESPACE_AROUND_BRACKETS = re.compile(r'\s*([「」『』（）])\s*')
    WHITESPACE_NORMALIZE = re.compile(r'[ 　\t]+')
    WHITESPACE_REMOVAL = re.compile(r'\s+')


# ============================================================================
# HASH FUNCTIONS WITH IMPROVED FALLBACK
# ============================================================================

def get_text_hash(text: str, seed: int = 0) -> int:
    """
    Get deterministic hash with xxhash or compatible blake2b fallback.
    
    Args:
        text: Text to hash
        seed: Seed for deterministic hashing
        
    Returns:
        64-bit integer hash
    """
    # Normalize text (remove all whitespace for consistency)
    normalized = re.sub(r'\s+', '', text)
    text_bytes = normalized.encode('utf-8')
    
    if XXHASH_AVAILABLE:
        return xxhash.xxh64(text_bytes, seed=seed).intdigest()
    else:
        # Improved fallback with proper seed handling
        # Use seed as salt for blake2b
        seed_bytes = seed.to_bytes(16, 'big', signed=False)
        h = hashlib.blake2b(text_bytes, digest_size=8, salt=seed_bytes)
        return int.from_bytes(h.digest(), byteorder='big')


# ============================================================================
# STEP 1: NORMALIZATION (ORIGINAL - UNCHANGED)
# ============================================================================

def unicode_normalize(cls, s):
    """Original normalization function - unchanged"""
    pt = re.compile('([{}]+)'.format(cls))
    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c
    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    """Original extra spaces removal - unchanged"""
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF', '\u3040-\u309F', '\u30A0-\u30FF',
                      '\u3000-\u303F', '\uFF00-\uFFEF'))
    basic_latin = '\u0000-\u007F'
    
    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s
    
    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    """Original NEologd normalization - unchanged"""
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)
    
    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}
    
    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)
    s = re.sub('[~∼∾〜〰～]', '', s)
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s


# ============================================================================
# STEP 2: INITIAL CLEANING
# ============================================================================

def remove_html_markup(text: str) -> str:
    """Remove HTML tags and entities"""
    text = RegexPatterns.HTML_TAGS.sub('', text)
    text = RegexPatterns.HTML_ENTITIES_NAMED.sub('', text)
    text = RegexPatterns.HTML_ENTITIES_DEC.sub('', text)
    text = RegexPatterns.HTML_ENTITIES_HEX.sub('', text)
    return text


def remove_urls(text: str) -> str:
    """Remove URLs"""
    text = RegexPatterns.URLS_HTTP.sub('', text)
    text = RegexPatterns.URLS_FTP.sub('', text)
    text = RegexPatterns.URLS_WWW.sub('', text)
    return text


def remove_emails(text: str) -> str:
    """Remove email addresses"""
    text = RegexPatterns.EMAILS.sub('', text)
    return text


def remove_special_symbols(text: str) -> str:
    """Remove emojis and special symbols"""
    text = RegexPatterns.EMOJIS.sub('', text)
    text = RegexPatterns.EMOTICONS_JP.sub('', text)
    text = RegexPatterns.SYMBOLS_COMMON.sub('', text)
    text = RegexPatterns.SYMBOLS_SHAPES.sub('', text)
    text = RegexPatterns.SYMBOLS_NUMBERS.sub('', text)
    text = RegexPatterns.REPEATED_EQUALS.sub('', text)
    text = RegexPatterns.REPEATED_DASHES.sub('', text)
    text = RegexPatterns.REPEATED_STARS.sub('', text)
    return text


def initial_cleaning(text: str) -> str:
    """Apply all initial cleaning steps"""
    text = remove_html_markup(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_special_symbols(text)
    return text


# ============================================================================
# STEP 3: ENHANCED QUALITY FILTERS
# ============================================================================

@dataclass
class QualityThresholds:
    """Configurable quality filter thresholds"""
    max_katakana_ratio: float = 0.7
    max_punctuation_ratio: float = 0.3
    max_number_ratio: float = 0.5
    min_particles_per_20chars: int = 1
    max_parentheses: int = 3


def filter_low_quality_japanese(
    text: str, 
    thresholds: QualityThresholds
) -> Tuple[bool, Optional[str]]:
    """
    Check basic quality filters.
    
    Args:
        text: Text to check
        thresholds: Quality thresholds
        
    Returns:
        (is_valid, rejection_reason)
    """
    if not text or len(text) == 0:
        return False, "empty"
    
    text_len = len(text)
    
    # Too much katakana (likely non-Japanese or low quality)
    katakana = RegexPatterns.KATAKANA.findall(text)
    if text_len > 0 and len(katakana) / text_len > thresholds.max_katakana_ratio:
        return False, "excessive_katakana"
    
    # Repetitive characters
    if RegexPatterns.REPETITIVE_CHARS.search(text):
        return False, "repetitive_chars"
    
    # Too much punctuation
    punctuation = RegexPatterns.PUNCTUATION_JP.findall(text)
    if text_len > 0 and len(punctuation) / text_len > thresholds.max_punctuation_ratio:
        return False, "excessive_punctuation"
    
    # URL fragments still present
    if RegexPatterns.URL_FRAGMENTS.search(text.lower()):
        return False, "url_fragments"
    
    # Too many numbers
    numbers = RegexPatterns.NUMBERS.findall(text)
    if text_len > 0 and len(numbers) / text_len > thresholds.max_number_ratio:
        return False, "excessive_numbers"
    
    return True, None


def filter_forum_artifacts(
    text: str, 
    thresholds: QualityThresholds
) -> Tuple[bool, Optional[str]]:
    """
    Filter forum/discussion board specific content.
    
    Args:
        text: Text to check
        thresholds: Quality thresholds
        
    Returns:
        (is_valid, rejection_reason)
    """
    # Quote reply markers
    if RegexPatterns.QUOTE_MARKERS.match(text):
        return False, "quote_marker"
    
    # Casual interjections
    if RegexPatterns.INTERJECTIONS.search(text):
        return False, "interjection"
    
    # Navigation/meta text
    if RegexPatterns.NAVIGATION.search(text):
        return False, "navigation"
    
    # Legal boilerplate
    if RegexPatterns.BOILERPLATE.search(text):
        return False, "boilerplate"
    
    # Too many parentheses (likely commentary/asides)
    if text.count('(') > thresholds.max_parentheses or text.count('（') > thresholds.max_parentheses:
        return False, "excessive_parentheses"
    
    return True, None


def filter_sentence_completeness(
    text: str, 
    thresholds: QualityThresholds
) -> Tuple[bool, Optional[str]]:
    """
    Check if text is a complete sentence.
    
    Args:
        text: Text to check
        thresholds: Quality thresholds
        
    Returns:
        (is_valid, rejection_reason)
    """
    text_len = len(text)
    
    # Ends with comma (incomplete)
    if text.endswith('、') or text.endswith(','):
        return False, "incomplete_comma"
    
    # Very short without proper ending
    if text_len < 15 and not RegexPatterns.SENTENCE_ENDINGS.search(text):
        return False, "short_no_ending"
    
    # Should have at least one sentence ending for longer text
    if text_len > 30 and not RegexPatterns.SENTENCE_ENDINGS.search(text):
        return False, "long_no_ending"
    
    # Check for sentence structure (particles)
    particles = RegexPatterns.PARTICLES.findall(text)
    if text_len > 20 and len(particles) < thresholds.min_particles_per_20chars:
        return False, "missing_particles"
    
    return True, None


# ============================================================================
# STEP 4: WHITESPACE & PUNCTUATION
# ============================================================================

def clean_whitespace_japanese(text: str) -> str:
    """Clean and normalize whitespace for Japanese text"""
    text = text.strip()
    text = RegexPatterns.WHITESPACE_NORMALIZE.sub(' ', text)
    text = RegexPatterns.WHITESPACE_AROUND_PUNCT.sub(r'\1', text)
    text = RegexPatterns.WHITESPACE_AROUND_BRACKETS.sub(r'\1', text)
    return text


def standardize_punctuation(text: str) -> str:
    """Standardize punctuation marks"""
    text = RegexPatterns.PERIODS.sub('。', text)
    text = RegexPatterns.COMMAS.sub('、', text)
    text = RegexPatterns.EXCLAMATIONS.sub('！', text)
    text = RegexPatterns.QUESTIONS.sub('？', text)
    text = RegexPatterns.ELLIPSIS_DOTS.sub('…', text)
    text = RegexPatterns.ELLIPSIS_MARU.sub('…', text)
    text = RegexPatterns.QUOTES_OPEN.sub('「', text)
    text = RegexPatterns.QUOTES_CLOSE.sub('」', text)
    text = RegexPatterns.PUNCT_SPACE.sub(r'\1 \2', text)
    return text


# ============================================================================
# DEDUPLICATION WITH CHECKPOINT VERSIONING
# ============================================================================

@dataclass
class DeduplicationState:
    """Serializable deduplication state with versioning"""
    seen_hashes: Set[int]
    doc_count: int
    seed: int
    hash_function: str
    version: int = 1


class FastDeduplication:
    """
    Fast deduplication using xxhash or blake2b.
    Includes checkpoint versioning and compatibility checking.
    """
    
    CHECKPOINT_VERSION = 1
    
    def __init__(self, seed: int = 0):
        self.seen_hashes: Set[int] = set()
        self.doc_count = 0
        self.seed = seed
        self.hash_function = 'xxhash64' if XXHASH_AVAILABLE else 'blake2b'
    
    def get_hash(self, text: str) -> int:
        """Get hash for text"""
        return get_text_hash(text, self.seed)
    
    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate"""
        text_hash = self.get_hash(text)
        return text_hash in self.seen_hashes
    
    def add(self, text: str) -> int:
        """Add text to deduplication set"""
        text_hash = self.get_hash(text)
        self.seen_hashes.add(text_hash)
        self.doc_count += 1
        return text_hash
    
    def save_state(self, filepath: str):
        """Save deduplication state with versioning"""
        state = DeduplicationState(
            seen_hashes=self.seen_hashes,
            doc_count=self.doc_count,
            seed=self.seed,
            hash_function=self.hash_function,
            version=self.CHECKPOINT_VERSION
        )
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size_mb = os.path.getsize(filepath) / (1024 ** 2)
            logger.info(f"✓ Saved checkpoint: {filepath} ({file_size_mb:.2f} MB)")
            logger.info(f"  Unique documents: {self.doc_count:,}")
            logger.info(f"  Hash function: {self.hash_function}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_state(self, filepath: str):
        """Load deduplication state with compatibility checking"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                state: DeduplicationState = pickle.load(f)
            
            # Check version compatibility
            if state.version != self.CHECKPOINT_VERSION:
                logger.warning(
                    f"⚠ Checkpoint version mismatch: "
                    f"saved={state.version}, current={self.CHECKPOINT_VERSION}"
                )
            
            # Check hash function compatibility
            if state.hash_function != self.hash_function:
                logger.warning(f"⚠ Hash function mismatch!")
                logger.warning(f"  Checkpoint: {state.hash_function}")
                logger.warning(f"  Current: {self.hash_function}")
                logger.warning(f"  Hashes will NOT match - consider regenerating checkpoint")
                
                response = input("Continue anyway? (yes/no): ")
                if response.lower() != 'yes':
                    raise RuntimeError("Hash function mismatch - aborting")
            
            self.seen_hashes = state.seen_hashes
            self.doc_count = state.doc_count
            self.seed = state.seed
            
            logger.info(f"✓ Loaded checkpoint: {filepath}")
            logger.info(f"  Documents indexed: {self.doc_count:,}")
            logger.info(f"  Hash function: {state.hash_function}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def get_stats(self) -> Dict:
        """Get deduplication statistics"""
        return {
            'num_documents': self.doc_count,
            'num_unique_hashes': len(self.seen_hashes),
            'dedup_method': self.hash_function,
            'seed': self.seed,
            'memory_mb': sys.getsizeof(self.seen_hashes) / (1024 ** 2)
        }


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class CleaningConfig:
    """Configuration for text cleaning pipeline"""
    
    # Basic filters
    apply_normalization: bool = True
    apply_initial_cleaning: bool = True
    check_duplicates: bool = True
    clean_whitespace: bool = True
    standardize_punctuation: bool = True
    apply_quality_filter: bool = True
    
    # Enhanced filters
    filter_forum_content: bool = True
    check_sentence_completeness: bool = True
    remove_csv_artifacts: bool = True
    
    # Length thresholds
    min_line_length: int = 10
    min_japanese_ratio: float = 0.5
    max_length: Optional[int] = None
    
    # Quality thresholds
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    
    # CSV settings
    csv_text_column: int = 1  # Column index (0-based)
    csv_has_header: bool = False
    
    # Hash seed
    hash_seed: int = 42
    
    # Encoding
    file_encoding: str = 'utf-8'
    encoding_errors: str = 'ignore'  # How to handle encoding errors


@dataclass
class CleaningStats:
    """Statistics for cleaning pipeline"""
    
    total_processed: int = 0
    exact_duplicates: int = 0
    too_short: int = 0
    low_japanese_ratio: int = 0
    empty_after_cleaning: int = 0
    csv_parse_errors: int = 0
    encoding_errors: int = 0
    accepted: int = 0
    
    # Detailed quality filter stats
    quality_rejections: Dict[str, int] = field(default_factory=dict)
    
    def increment_quality_rejection(self, reason: str):
        """Increment rejection counter for specific reason"""
        self.quality_rejections[reason] = self.quality_rejections.get(reason, 0) + 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        acceptance_rate = (
            self.accepted / self.total_processed * 100 
            if self.total_processed > 0 else 0.0
        )
        
        result = {
            'total_processed': self.total_processed,
            'accepted': self.accepted,
            'acceptance_rate': f"{acceptance_rate:.2f}%",
            'exact_duplicates': self.exact_duplicates,
            'too_short': self.too_short,
            'low_japanese_ratio': self.low_japanese_ratio,
            'empty_after_cleaning': self.empty_after_cleaning,
            'csv_parse_errors': self.csv_parse_errors,
            'encoding_errors': self.encoding_errors,
        }
        
        # Add detailed quality rejections
        if self.quality_rejections:
            result['quality_rejections'] = self.quality_rejections
        
        return result


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class JapaneseTextCleaningPipeline:
    """
    Production-ready Japanese text cleaning pipeline.
    
    Features:
    - Proper CSV parsing
    - Enhanced quality filters
    - xxHash deduplication
    - Checkpoint versioning
    - Comprehensive error handling
    - Detailed logging
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self.dedup = FastDeduplication(seed=self.config.hash_seed)
        self.stats = CleaningStats()
    
    def clean_text(self, text: str, check_duplicate: bool = True) -> Optional[str]:
        """
        Clean a single line of Japanese text.
        
        Args:
            text: Text to clean
            check_duplicate: Whether to check for duplicates
            
        Returns:
            Cleaned text or None if rejected
        """
        self.stats.total_processed += 1
        
        if not text or not text.strip():
            return None
        
        try:
            # Remove CSV artifacts
            if self.config.remove_csv_artifacts:
                text = text.strip('「」『』"\'')
                text = text.strip()
            
            # Initial cleaning
            if self.config.apply_initial_cleaning:
                text = initial_cleaning(text)
            
            # Normalization
            if self.config.apply_normalization:
                text = normalize_neologd(text)
            
            if not text.strip():
                self.stats.empty_after_cleaning += 1
                return None
            
            # Duplicate check (early rejection for performance)
            if self.config.check_duplicates and check_duplicate:
                if self.dedup.is_duplicate(text):
                    self.stats.exact_duplicates += 1
                    return None
            
            # Whitespace cleaning
            if self.config.clean_whitespace:
                text = clean_whitespace_japanese(text)
            
            # Punctuation standardization
            if self.config.standardize_punctuation:
                text = standardize_punctuation(text)
            
            # Length filters
            text_len = len(text.strip())
            if text_len < self.config.min_line_length:
                self.stats.too_short += 1
                return None
            
            if self.config.max_length and text_len > self.config.max_length:
                return None
            
            # Japanese character ratio
            japanese_chars = RegexPatterns.JAPANESE_CHARS.findall(text)
            if text_len > 0:
                japanese_ratio = len(japanese_chars) / text_len
                if japanese_ratio < self.config.min_japanese_ratio:
                    self.stats.low_japanese_ratio += 1
                    return None
            
            # Quality filters
            if self.config.apply_quality_filter:
                is_valid, reason = filter_low_quality_japanese(
                    text, self.config.quality_thresholds
                )
                if not is_valid:
                    self.stats.increment_quality_rejection(reason)
                    return None
            
            # Forum artifacts filter
            if self.config.filter_forum_content:
                is_valid, reason = filter_forum_artifacts(
                    text, self.config.quality_thresholds
                )
                if not is_valid:
                    self.stats.increment_quality_rejection(reason)
                    return None
            
            # Sentence completeness check
            if self.config.check_sentence_completeness:
                is_valid, reason = filter_sentence_completeness(
                    text, self.config.quality_thresholds
                )
                if not is_valid:
                    self.stats.increment_quality_rejection(reason)
                    return None
            
            # Add to deduplication set
            if self.config.check_duplicates and check_duplicate:
                self.dedup.add(text)
            
            self.stats.accepted += 1
            return text.strip()
            
        except UnicodeDecodeError as e:
            self.stats.encoding_errors += 1
            logger.debug(f"Encoding error at line {self.stats.total_processed}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error at line {self.stats.total_processed}: {e}")
            return None
    
    def clean_csv_file(
        self,
        input_file: str,
        output_file: str,
        buffer_size: int = 10000,
        progress_interval: int = 50000
    ):
        """
        Process CSV file with proper parsing.
        
        Args:
            input_file: Input CSV file path
            output_file: Output text file path
            buffer_size: Number of lines to buffer before writing
            progress_interval: How often to print progress
        """
        buffer = []
        
        logger.info("=" * 70)
        logger.info("CSV Processing Started")
        logger.info("=" * 70)
        logger.info(f"Input:  {input_file}")
        logger.info(f"Output: {output_file}")
        logger.info(f"Text column index: {self.config.csv_text_column}")
        logger.info(f"Has header: {self.config.csv_has_header}")
        logger.info("")
        
        try:
            with open(
                input_file, 'r', 
                encoding=self.config.file_encoding,
                errors=self.config.encoding_errors
            ) as f_in, \
                 open(
                     output_file, 'w', 
                     encoding=self.config.file_encoding
                 ) as f_out:
                
                csv_reader = csv.reader(f_in)
                
                # Handle header
                if self.config.csv_has_header:
                    try:
                        header = next(csv_reader)
                        logger.info(f"CSV columns: {header}")
                        logger.info("")
                    except StopIteration:
                        logger.error("Empty CSV file")
                        return
                
                text_col_idx = self.config.csv_text_column
                
                # Process rows
                for row_num, row in enumerate(csv_reader, 1):
                    try:
                        # Validate row
                        if len(row) <= text_col_idx:
                            self.stats.csv_parse_errors += 1
                            logger.debug(f"Row {row_num}: Invalid column count")
                            continue
                        
                        # Extract text
                        text = row[text_col_idx].strip()
                        if not text:
                            continue
                        
                        # Clean the text
                        cleaned = self.clean_text(text)
                        
                        if cleaned:
                            buffer.append(cleaned)
                            
                            if len(buffer) >= buffer_size:
                                for line in buffer:
                                    f_out.write(line + '\n')
                                buffer = []
                        
                        # Progress reporting
                        if row_num % progress_interval == 0:
                            acceptance_rate = (
                                self.stats.accepted / 
                                self.stats.total_processed * 100
                            )
                            logger.info(
                                f"Processed {row_num:,} rows | "
                                f"Accepted: {self.stats.accepted:,} ({acceptance_rate:.1f}%) | "
                                f"Duplicates: {self.stats.exact_duplicates:,}"
                            )
                    
                    except csv.Error as e:
                        self.stats.csv_parse_errors += 1
                        logger.debug(f"CSV parse error at row {row_num}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Unexpected error at row {row_num}: {e}")
                        continue
                
                # Flush remaining buffer
                if buffer:
                    for line in buffer:
                        f_out.write(line + '\n')
        
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file}")
            raise
        except PermissionError:
            logger.error(f"Permission denied: {input_file}")
            raise
        except Exception as e:
            logger.error(f"Fatal error during processing: {e}")
            raise
    
    def clean_file_streaming(
        self,
        input_file: str,
        output_file: str,
        buffer_size: int = 10000,
        progress_interval: int = 50000
    ):
        """
        Process plain text file (non-CSV).
        
        Args:
            input_file: Input text file path
            output_file: Output text file path
            buffer_size: Number of lines to buffer before writing
            progress_interval: How often to print progress
        """
        buffer = []
        
        logger.info("=" * 70)
        logger.info("Text File Processing Started")
        logger.info("=" * 70)
        logger.info(f"Input:  {input_file}")
        logger.info(f"Output: {output_file}")
        logger.info("")
        
        try:
            with open(
                input_file, 'r',
                encoding=self.config.file_encoding,
                errors=self.config.encoding_errors
            ) as f_in, \
                 open(
                     output_file, 'w',
                     encoding=self.config.file_encoding
                 ) as f_out:
                
                for line_num, line in enumerate(f_in, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    cleaned = self.clean_text(line)
                    if cleaned:
                        buffer.append(cleaned)
                        
                        if len(buffer) >= buffer_size:
                            for text in buffer:
                                f_out.write(text + '\n')
                            buffer = []
                    
                    if line_num % progress_interval == 0:
                        acceptance_rate = (
                            self.stats.accepted / 
                            self.stats.total_processed * 100
                        )
                        logger.info(
                            f"Processed {line_num:,} lines | "
                            f"Accepted: {self.stats.accepted:,} ({acceptance_rate:.1f}%) | "
                            f"Duplicates: {self.stats.exact_duplicates:,}"
                        )
                
                # Flush remaining buffer
                if buffer:
                    for text in buffer:
                        f_out.write(text + '\n')
        
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file}")
            raise
        except Exception as e:
            logger.error(f"Fatal error during processing: {e}")
            raise
    
    def save_dedup_state(self, filepath: str):
        """Save deduplication state"""
        self.dedup.save_state(filepath)
    
    def load_dedup_state(self, filepath: str):
        """Load deduplication state"""
        self.dedup.load_state(filepath)
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            **self.stats.to_dict(),
            **self.dedup.get_stats()
        }
    
    def reset(self):
        """Reset pipeline state"""
        self.dedup = FastDeduplication(seed=self.config.hash_seed)
        self.stats = CleaningStats()
        logger.info("Pipeline reset")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """Example usage"""
    import time
    
    print("\n" + "=" * 70)
    print("Japanese Text Cleaning Pipeline v2.0")
    print("Enhanced quality filters + xxHash deduplication")
    print("=" * 70 + "\n")
    
    # Configuration
    config = CleaningConfig(
        min_line_length=10,
        min_japanese_ratio=0.5,
        filter_forum_content=True,
        check_sentence_completeness=True,
        remove_csv_artifacts=True,
        csv_text_column=1,  # Second column (0-indexed)
        csv_has_header=False,
        hash_seed=42,
        quality_thresholds=QualityThresholds(
            max_katakana_ratio=0.7,
            max_punctuation_ratio=0.3,
            max_number_ratio=0.5
        )
    )
    
    pipeline = JapaneseTextCleaningPipeline(config)
    
    # Example: Process CSV file
    input_csv = "cc100_ja_0_part0.csv"
    output_txt = "cc100_ja_0_part0_cleaned.txt"
    checkpoint_file = "dedup_checkpoint.pkl"
    
    # Check if we should load a checkpoint
    if os.path.exists(checkpoint_file):
        response = input(f"Found checkpoint: {checkpoint_file}. Load it? (yes/no): ")
        if response.lower() == 'yes':
            try:
                pipeline.load_dedup_state(checkpoint_file)
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                return
    
    if os.path.exists(input_csv):
        logger.info(f"Processing: {input_csv}\n")
        start_time = time.time()
        
        try:
            pipeline.clean_csv_file(input_csv, output_txt, progress_interval=50000)
            
            elapsed = time.time() - start_time
            
            logger.info("\n" + "=" * 70)
            logger.info("Processing Complete!")
            logger.info("=" * 70)
            logger.info(f"Time: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
            logger.info(
                f"Speed: {pipeline.stats.total_processed/elapsed:,.0f} lines/sec"
            )
            
            logger.info("\nStatistics:")
            for key, value in pipeline.get_stats().items():
                logger.info(f"  {key}: {value}")
            
            # Save checkpoint
            pipeline.save_dedup_state(checkpoint_file)
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
    else:
        logger.warning(f"Example file not found: {input_csv}")
        logger.info("\nUsage:")
        logger.info("  python japanese_text_cleaner.py")
        logger.info("\nOr import and use:")
        logger.info(
            "  from japanese_text_cleaner import "
            "JapaneseTextCleaningPipeline, CleaningConfig"
        )


if __name__ == "__main__":
    main()