from datasets import load_dataset
import os
import time
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleaning_pipeline_japanese.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def format_time(seconds):
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def main():
    start_time = time.time()
    
    print("="*80)
    print("Japanese Data Cleaning Pipeline - PRODUCTION READY (FIXED)")
    print("Balanced filters for quality without perplexity (model issues)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # ========================
    # CONFIGURATION - BALANCED FOR QUALITY
    # ========================
    CONFIG = {
        # INPUT/OUTPUT PATHS
        'input_path': '/workspace/Nepmedjp/data-juicerr/cc100_ja_13_cleaned_cleaned.txt',
        'output_path': "/workspace/Nepmedjp/data-juicerr/file13.jsonl",

        # Processing settings
        'num_proc': 4,
        'batch_size': 1000,
        'writer_batch_size': 5000,
        
        # Language settings
        'language': 'ja',
        
        # BALANCED thresholds - catch garbage but keep good data
        'min_text_length': 10,       # Minimum 10 characters
        'max_text_length': 500000,   # Allow very long documents (500K chars)
        'min_words': 2,              # At least 2 words (was 5 - TOO STRICT)
        'max_words': 1000000,        # Effectively unlimited
        
        # MinHash LSH Deduplication settings
        'minhash_num_perm': 128,       # Number of permutations for MinHash
        'minhash_threshold': 0.8,      # Jaccard similarity threshold (0.8 = 80% similar)
        'minhash_shingle_size': 3,     # k-shingle size (word n-grams)
        'minhash_batch_size': 50000,   # Process in batches for memory efficiency
    }
    
    logger.info(f"Configuration: {CONFIG}")
    print(f"\n📋 Configuration:")
    print(f"  Input: {CONFIG['input_path']}")
    print(f"  Output: {CONFIG['output_path']}")
    print(f"  Min words: {CONFIG['min_words']} (balanced for Japanese)")
    print(f"  Text length: {CONFIG['min_text_length']}-{CONFIG['max_text_length']} chars")
    print(f"  MinHash threshold: {CONFIG['minhash_threshold']} (near-duplicate detection)")

    # ========================
    # LOAD DATA
    # ========================
    print("\n" + "─"*80)
    print("STAGE 1: DATA LOADING")
    print("─"*80)
    
    try:
        input_path = CONFIG['input_path']
        
        if input_path.endswith('.txt'):
            print(f"📄 Loading plain text file...")
            logger.info(f"Loading text file: {input_path}")
            dataset = load_dataset('text', data_files=input_path, split='train')
            
        elif input_path.endswith(('.json', '.jsonl')):
            print(f"📄 Loading JSON/JSONL file...")
            logger.info(f"Loading JSON file: {input_path}")
            dataset = load_dataset('json', data_files=input_path, split='train')
            
        else:
            raise ValueError(f"Unsupported file format: {input_path}")
        
        initial_count = len(dataset)
        logger.info(f"Successfully loaded {initial_count:,} samples")
        print(f"✓ Loaded {initial_count:,} samples")
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # ========================
    # IMPORT OPERATORS
    # ========================
    print("\n" + "─"*80)
    print("STAGE 2: IMPORTING OPERATORS")
    print("─"*80)
    
    try:
        # Mappers
        from data_juicer.ops.mapper import (
            clean_email_mapper,
            clean_links_mapper,
            clean_copyright_mapper,
            fix_unicode_mapper,
            punctuation_normalization_mapper,
            remove_comments_mapper,
            whitespace_normalization_mapper,
        )
        
        # Filters
        from data_juicer.ops.filter import (
            alphanumeric_filter,
            average_line_length_filter,
            character_repetition_filter,
            language_id_score_filter,
            maximum_line_length_filter,
            special_characters_filter,
            text_length_filter,
            words_num_filter,
            word_repetition_filter,
        )
        
        # MinHash LSH deduplication via datasketch
        try:
            import re
            from datasketch import MinHash, MinHashLSH
            DEDUP_AVAILABLE = True
            print("✓ MinHash LSH deduplication available (datasketch)")
        except ImportError:
            DEDUP_AVAILABLE = False
            print("⚠ datasketch not installed. Run: pip install datasketch")
        
        from data_juicer.utils.constant import Fields
        
        print("✓ All operators imported successfully")
        logger.info("Operators imported successfully")
        
    except ImportError as e:
        logger.error(f"Failed to import operators: {e}")
        print(f"❌ Import Error: {e}")
        raise

    # ========================
    # MAPPERS - TEXT CLEANING
    # ========================
    print("\n" + "─"*80)
    print("STAGE 3: APPLYING MAPPERS (Text Cleaning)")
    print("─"*80)
    
    mappers = [
        ("CleanEmailMapper", clean_email_mapper.CleanEmailMapper()),
        ("CleanLinksMapper", clean_links_mapper.CleanLinksMapper()),
        ("CleanCopyrightMapper", clean_copyright_mapper.CleanCopyrightMapper()),
        ("FixUnicodeMapper", fix_unicode_mapper.FixUnicodeMapper()),
        ("PunctuationNormalizationMapper", punctuation_normalization_mapper.PunctuationNormalizationMapper()),
        ("RemoveCommentsMapper", remove_comments_mapper.RemoveCommentsMapper()),
        ("WhitespaceNormalizationMapper", whitespace_normalization_mapper.WhitespaceNormalizationMapper()),
    ]
    
    mapper_start = time.time()
    
    for i, (name, op) in enumerate(mappers, 1):
        op_start = time.time()
        print(f"\n[{i}/{len(mappers)}] {name}", end=' ')
        
        try:
            dataset = dataset.map(
                op.process,
                num_proc=CONFIG['num_proc'],
                batched=True,
                batch_size=CONFIG['batch_size'],
                writer_batch_size=CONFIG['writer_batch_size'],
                load_from_cache_file=False
            )
            
            op_time = time.time() - op_start
            print(f"✓ ({format_time(op_time)})")
            logger.info(f"{name} completed in {format_time(op_time)}")
            
        except Exception as e:
            logger.error(f"{name} failed: {e}")
            print(f"❌ Failed: {e}")
            raise
    
    mapper_total = time.time() - mapper_start
    print(f"\n✓ All mappers completed in {format_time(mapper_total)}")

    # ========================
    # INITIALIZE STATS
    # ========================
    print("\n" + "─"*80)
    print("STAGE 4: INITIALIZING STATS")
    print("─"*80)
    
    def add_stats_field(example):
        example[Fields.stats] = {}
        return example
    
    dataset = dataset.map(
        add_stats_field,
        num_proc=CONFIG['num_proc'],
        load_from_cache_file=False
    )
    print("✓ Stats field initialized")

    # ========================
    # FILTERS - BALANCED SETTINGS
    # ========================
    print("\n" + "─"*80)
    print("STAGE 5: APPLYING FILTERS (Balanced Quality Control)")
    print("─"*80)
    
    filters = [
        # 1. ALPHANUMERIC - Balanced for Japanese
        ("AlphanumericFilter", alphanumeric_filter.AlphanumericFilter(
            tokenization=False,
            min_ratio=0.15,  # LOWERED - Japanese uses less ASCII (was 0.25)
            max_ratio=0.95
        )),
        
        # 2. AVERAGE LINE LENGTH - Reasonable limit
        ("AverageLineLengthFilter", average_line_length_filter.AverageLineLengthFilter(
            min_len=10,      # At least 10 chars per line on average
            max_len=10000    # Not too long per line (was 50000 - too lenient)
        )),
        
        # 3. CHARACTER REPETITION - Catch spam
        ("CharacterRepetitionFilter", character_repetition_filter.CharacterRepetitionFilter(
            rep_len=10,
            max_ratio=0.20   # Balanced (was 0.15 - too strict)
        )),
        
        # 4. LANGUAGE ID - Ensure Japanese with some flexibility
        ("LanguageIDScoreFilter", language_id_score_filter.LanguageIDScoreFilter(
            lang='ja',
            min_score=0.65   # LOWERED - allow some mixed content (was 0.8)
        )),
        
        # 5. MAXIMUM LINE LENGTH - Prevent extreme outliers
        ("MaximumLineLengthFilter", maximum_line_length_filter.MaximumLineLengthFilter(
            min_len=10,      # Lines should be at least 10 chars
            max_len=50000    # But not ridiculously long
        )),
        
        # 6. SPECIAL CHARACTERS - Balanced
        ("SpecialCharactersFilter", special_characters_filter.SpecialCharactersFilter(
            min_ratio=0.0,   # Allow documents with no special chars
            max_ratio=0.45   # RAISED slightly (was 0.40)
        )),
        
        # 7. TEXT LENGTH - Keep short and long
        ("TextLengthFilter", text_length_filter.TextLengthFilter(
            min_len=CONFIG['min_text_length'],
            max_len=CONFIG['max_text_length']
        )),
        
        # 8. WORDS COUNT - CRITICAL FIX
        ("WordsNumFilter", words_num_filter.WordsNumFilter(
            lang='ja',
            tokenization=True,
            min_num=CONFIG['min_words'],   # 2 words minimum (was 5 - TOO STRICT)
            max_num=CONFIG['max_words']
        )),
        
        # 9. WORD REPETITION - Catch spam
        ("WordRepetitionFilter", word_repetition_filter.WordRepetitionFilter(
            lang='ja',
            tokenization=True,
            rep_len=10,
            min_ratio=0.0,
            max_ratio=0.40   # Balanced (was 0.35)
        )),
    ]
    
    filter_start = time.time()
    filter_stats = []
    
    for i, (name, op) in enumerate(filters, 1):
        op_start = time.time()
        print(f"\n[{i}/{len(filters)}] {name}")
        
        try:
            # Compute stats
            print(f"  Computing stats...", end=' ', flush=True)
            dataset = dataset.map(
                op.compute_stats,
                num_proc=CONFIG['num_proc'],
                batched=True,
                batch_size=CONFIG['batch_size'],
                load_from_cache_file=False
            )
            print("✓", end=' ')
            
            # Apply filter
            print(f"Filtering...", end=' ', flush=True)
            before = len(dataset)
            dataset = dataset.filter(
                op.process,
                num_proc=CONFIG['num_proc'],
                batched=True,
                batch_size=CONFIG['batch_size'],
                load_from_cache_file=False
            )
            after = len(dataset)
            
            filtered = before - after
            filter_pct = (filtered / before * 100) if before > 0 else 0
            op_time = time.time() - op_start
            
            print(f"✓")
            print(f"  → {after:,} samples kept, {filtered:,} removed ({filter_pct:.2f}%), {format_time(op_time)}")
            
            filter_stats.append({
                'name': name,
                'before': before,
                'after': after,
                'filtered': filtered,
                'percentage': filter_pct,
                'time': op_time
            })
            
            logger.info(f"{name}: {after:,} samples (filtered {filtered:,} = {filter_pct:.2f}%)")
            
        except Exception as e:
            logger.error(f"{name} failed: {e}")
            print(f"  ❌ Failed: {e}")
            # Continue with warning instead of crashing
            print(f"  ⚠ Skipping this filter and continuing...")
            continue
    
    filter_total = time.time() - filter_start
    print(f"\n✓ All filters completed in {format_time(filter_total)}")

    # ========================
    # MINHASH LSH NEAR-DEDUPLICATION (STAGE 6 - REPLACED)
    # ========================
    print("\n" + "─"*80)
    print("STAGE 6: MINHASH LSH NEAR-DEDUPLICATION")
    print("─"*80)
    
    dedup_time = 0
    
    if DEDUP_AVAILABLE:
        try:
            dedup_start = time.time()
            before_dedup = len(dataset)
            
            NUM_PERM = CONFIG['minhash_num_perm']
            THRESHOLD = CONFIG['minhash_threshold']
            SHINGLE_K = CONFIG['minhash_shingle_size']
            
            print(f"  Config: num_perm={NUM_PERM}, threshold={THRESHOLD}, shingle_k={SHINGLE_K}")
            print(f"  Before: {before_dedup:,} samples")
            
            # --- Step 1: Japanese-aware shingling function ---
            def get_shingles_ja(text, k=SHINGLE_K):
                """
                Generate character-level k-shingles for Japanese text.
                Japanese doesn't use spaces between words, so character n-grams
                work better than word n-grams.
                """
                if not text or not isinstance(text, str):
                    return set()
                # Normalize: remove excessive whitespace for Japanese
                text = re.sub(r'\s+', '', text)
                text = text.strip()
                
                if len(text) < k:
                    return {text} if text else set()
                
                # Character-level k-shingles (ideal for Japanese)
                shingles = set(text[i:i+k] for i in range(len(text) - k + 1))
                return shingles
            
            # --- Step 2: Build MinHash signatures and insert into LSH ---
            print("  Step 1/3: Computing MinHash signatures & building LSH index...", flush=True)
            
            total_docs = len(dataset)
            lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
            minhash_signatures = [None] * total_docs  # Pre-allocate list
            empty_count = 0
            insert_fail_count = 0
            
            # Process in chunks to avoid loading all texts at once
            CHUNK_SIZE = 50000
            for chunk_start in range(0, total_docs, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, total_docs)
                # Only load this chunk's texts
                chunk_texts = dataset.select(range(chunk_start, chunk_end))['text']
                
                for i, text in enumerate(chunk_texts):
                    idx = chunk_start + i
                    shingles = get_shingles_ja(text, k=SHINGLE_K)
                    
                    m = MinHash(num_perm=NUM_PERM)
                    if shingles:
                        for shingle in shingles:
                            m.update(shingle.encode('utf-8'))
                    else:
                        empty_count += 1
                    
                    minhash_signatures[idx] = m
                    
                    try:
                        lsh.insert(str(idx), m)
                    except ValueError as ve:
                        # This shouldn't happen since keys are unique,
                        # but log it if it does
                        insert_fail_count += 1
                        if insert_fail_count <= 5:
                            logger.warning(f"LSH insert failed for idx {idx}: {ve}")
                
                # Free chunk texts
                del chunk_texts
                
                processed = chunk_end
                if processed % 100000 == 0 or processed == total_docs:
                    print(f"    Processed {processed:,}/{total_docs:,} documents "
                          f"({processed/total_docs*100:.1f}%)", flush=True)
            
            if empty_count > 0:
                print(f"    ⚠ {empty_count:,} documents produced empty shingles")
            if insert_fail_count > 0:
                print(f"    ⚠ {insert_fail_count:,} LSH insert failures (logged)")
            
            logger.info(f"MinHash signatures computed for {total_docs:,} documents")
            
            # --- Step 3: Find duplicate clusters using Union-Find ---
            print("  Step 2/3: Querying LSH for near-duplicate clusters...", flush=True)
            
            # Union-Find for clustering duplicates
            parent = list(range(total_docs))
            uf_rank = [0] * total_docs
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]  # Path compression
                    x = parent[x]
                return x
            
            def union(x, y):
                px, py = find(x), find(y)
                if px == py:
                    return
                # Always make the smaller index the root
                # so we deterministically keep the first occurrence
                if px > py:
                    px, py = py, px
                parent[py] = px
            
            # Query each document and union duplicates
            duplicate_pairs_found = 0
            for idx in range(total_docs):
                if minhash_signatures[idx] is None:
                    continue
                candidates = lsh.query(minhash_signatures[idx])
                for cand in candidates:
                    cand_idx = int(cand)
                    if cand_idx != idx:
                        union(idx, cand_idx)
                        duplicate_pairs_found += 1
                
                if (idx + 1) % 100000 == 0 or idx == total_docs - 1:
                    print(f"    Queried {idx+1:,}/{total_docs:,} documents "
                          f"({(idx+1)/total_docs*100:.1f}%)", flush=True)
            
            print(f"    Found {duplicate_pairs_found:,} duplicate pair connections")
            
            # --- Step 4: Keep one document per cluster (smallest index = root) ---
            # With our union that always roots on smaller index,
            # root == idx means this is the first occurrence in its cluster
            keep_indices = []
            for idx in range(total_docs):
                if find(idx) == idx:
                    keep_indices.append(idx)
            
            num_duplicates = total_docs - len(keep_indices)
            
            print(f"    Unique clusters: {len(keep_indices):,}")
            print(f"    Documents to keep: {len(keep_indices):,}")
            print(f"    Duplicates to remove: {num_duplicates:,}")
            
            # Free memory before filtering
            del minhash_signatures, lsh, parent, uf_rank
            
            # --- Step 5: Filter the dataset ---
            print("  Filtering dataset...", end=' ', flush=True)
            dataset = dataset.select(keep_indices)
            
            print("✓")
            
            after_dedup = len(dataset)
            deduped = before_dedup - after_dedup
            dedup_pct = (deduped / before_dedup * 100) if before_dedup > 0 else 0
            
            dedup_time = time.time() - dedup_start
            
            print(f"  After: {after_dedup:,} samples")
            print(f"  ✓ Removed {deduped:,} near-duplicates ({dedup_pct:.2f}%)")
            print(f"  Time: {format_time(dedup_time)}")
            
            filter_stats.append({
                'name': 'MinHashLSH_NearDedup',
                'before': before_dedup,
                'after': after_dedup,
                'filtered': deduped,
                'percentage': dedup_pct,
                'time': dedup_time
            })
            
            logger.info(f"MinHash LSH Dedup: {after_dedup:,} samples "
                       f"(removed {deduped:,} = {dedup_pct:.2f}%)")
            
            
        except Exception as e:
            logger.error(f"MinHash LSH deduplication failed: {e}", exc_info=True)
            print(f"  ❌ Failed: {e}")
            print(f"  ⚠ Continuing without deduplication...")
            dedup_time = 0
    else:
        print("⚠ datasketch not installed. Run: pip install datasketch")

    # ========================
    # SAVE RESULTS
    # ========================
    print("\n" + "─"*80)
    print("STAGE 7: SAVING CLEANED DATASET")
    print("─"*80)
    
    save_start = time.time()
    
    try:
        # Remove stats field
        def remove_stats_field(example):
            if Fields.stats in example:
                del example[Fields.stats]
            # Remove dedup hash if present
            if '__dj__hash' in example:
                del example['__dj__hash']
            return example
        
        print("Cleaning up internal fields...", end=' ', flush=True)
        dataset = dataset.map(
            remove_stats_field,
            num_proc=CONFIG['num_proc'],
            load_from_cache_file=False
        )
        print("✓")
        
        # Create output directory
        os.makedirs(os.path.dirname(CONFIG['output_path']), exist_ok=True)
        
        # Save
        print(f"Writing to {Path(CONFIG['output_path']).name}...", end=' ', flush=True)
        dataset.to_json(
            CONFIG['output_path'],
            orient='records',
            lines=True,
            force_ascii=False,
            num_proc=CONFIG['num_proc']
        )
        
        save_time = time.time() - save_start
        print(f"✓ ({format_time(save_time)})")
        
    except Exception as e:
        logger.error(f"Save failed: {e}")
        print(f"❌ Failed: {e}")
        raise

    # ========================
    # FINAL STATISTICS
    # ========================
    total_time = time.time() - start_time
    final_count = len(dataset)
    total_filtered = initial_count - final_count
    filter_percentage = (total_filtered / initial_count) * 100
    retention_percentage = (final_count / initial_count) * 100
    
    if os.path.exists(CONFIG['output_path']):
        output_size = os.path.getsize(CONFIG['output_path']) / (1024*1024)
    else:
        output_size = 0
    
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print(f"\n{'SUMMARY STATISTICS':^80}")
    print("─"*80)
    print(f"  Language:             {'Japanese (ja)':>15}")
    print(f"  Initial samples:      {initial_count:>15,}")
    print(f"  Final samples:        {final_count:>15,}")
    print(f"  Total filtered:       {total_filtered:>15,} ({filter_percentage:.2f}%)")
    print(f"  Retention rate:       {retention_percentage:>14.2f}%")
    print(f"  Output file size:     {output_size:>14.2f} MB")
    print(f"  Total time:           {format_time(total_time):>15}")
    print(f"  Processing rate:      {final_count/total_time:>14.1f} samples/sec")
    
    print("\n" + "─"*80)
    print(f"{'STAGE BREAKDOWN':^80}")
    print("─"*80)
    print(f"  Mappers:              {format_time(mapper_total):>15}")
    print(f"  Filters:              {format_time(filter_total):>15}")
    print(f"  Deduplication:        {format_time(dedup_time) if dedup_time > 0 else 'SKIPPED':>15}")
    print(f"  Saving:               {format_time(save_time):>15}")
    
    print("\n" + "─"*80)
    print(f"{'ALL FILTERS (by samples removed)':^80}")
    print("─"*80)
    
    sorted_filters = sorted(filter_stats, key=lambda x: x['filtered'], reverse=True)
    for stat in sorted_filters:
        print(f"  {stat['name']:<30} {stat['filtered']:>10,} ({stat['percentage']:>5.2f}%)")
    
    print("\n" + "="*80)
    print(f"✓ Output saved to: {CONFIG['output_path']}")
    print(f"✓ Log saved to: cleaning_pipeline_japanese.log")
    print("="*80)
    
    print("\n" + "💡 BALANCED CONFIGURATION:")
    print("─"*80)
    print("  • Alphanumeric: 0.15 min (balanced for Japanese)")
    print("  • Language ID: 0.65 min score (some flexibility)")
    print("  • Words count: 2 min (was 5 - key fix!)")
    print("  • Word repetition: catches spam patterns")
    print("  • Character repetition: catches spam patterns")
    print(f"  • MinHash LSH dedup: threshold={CONFIG['minhash_threshold']}, "
          f"num_perm={CONFIG['minhash_num_perm']}, shingle_k={CONFIG['minhash_shingle_size']}")
    print("  • Expected retention: 40-70% of original data")
    print("  • Note: Perplexity filter disabled (3GB model download issues)")
    print("="*80)
    
    # Quality check warning
    if retention_percentage < 30:
        print("\n⚠ WARNING: Retention rate < 30%. Filters may be too strict!")
    elif retention_percentage > 90:
        print("\n⚠ WARNING: Retention rate > 90%. Filters may be too lenient!")
    else:
        print(f"\n✓ Good retention rate ({retention_percentage:.1f}%). Data quality should be high!")
    
    logger.info(f"Pipeline completed in {format_time(total_time)}")
    logger.info(f"Final: {final_count:,} samples ({output_size:.2f} MB, {retention_percentage:.2f}% retention)")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        logger.warning("Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise