#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Process ONE file at a time with xxHash Persistent Deduplication

ULTRA-FAST VERSION: 10-50x faster than MinHash!

USAGE:
1. Edit INPUT_FILE below to point to the file you want to process NOW
2. Run: python datacleaning_single_xxhash.py
3. Later, change INPUT_FILE to the next file and run again
4. The deduplication index is saved and loaded automatically!

The deduplication index will remember all previously processed files.
"""

from Japanesedatacleaner import (
    JapaneseTextCleaningPipeline, 
    CleaningConfig, 
    QualityThresholds
)
import pickle
import time
from pathlib import Path


# ============================================================================
# CONFIGURATION: Edit this for each run
# ============================================================================

# Change this to the file you want to process NOW
INPUT_FILE = "/Volumes/research112/pre-trainingdata/Japanese/CC-100-jp/cc100ja/cc100_ja_10.csv"

# Output directory
OUTPUT_DIR = "/Volumes/research112/finalcleaneddata1"

# Deduplication index file (DO NOT CHANGE - this persists across runs)
DEDUP_INDEX_FILE = "/Volumes/research112/xxhash_dedup_index.pkl"

# Pipeline configuration
config = CleaningConfig(
    # Length filters
    min_line_length=10,
    min_japanese_ratio=0.5,
    max_length=None,
    
    # Quality thresholds
    quality_thresholds=QualityThresholds(
        max_katakana_ratio=0.7,
        max_punctuation_ratio=0.3,
        max_number_ratio=0.5,
        min_particles_per_20chars=1,
        max_parentheses=3
    ),
    
    # Enhanced filters
    filter_forum_content=True,
    check_sentence_completeness=True,
    remove_csv_artifacts=True,
    
    # CSV settings
    csv_text_column=1,  # Column index (0-based) - second column
    csv_has_header=False,
    
    # Basic filters
    apply_normalization=True,
    apply_initial_cleaning=True,
    check_duplicates=True,
    clean_whitespace=True,
    standardize_punctuation=True,
    apply_quality_filter=True,
    
    # Hash seed for reproducibility
    hash_seed=42,
    
    # Encoding
    file_encoding='utf-8',
    encoding_errors='ignore'
)


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    print("="*80)
    print("Japanese Text Cleaning - ULTRA-FAST xxHash Deduplication")
    print("10-50x faster than MinHash!")
    print("="*80 + "\n")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Check if input file exists
    if not Path(INPUT_FILE).exists():
        print(f"✗ ERROR: File not found: {INPUT_FILE}")
        return
    
    # Determine if input is CSV or plain text
    is_csv = INPUT_FILE.lower().endswith('.csv')
    
    # Generate output filename
    input_filename = Path(INPUT_FILE).name
    output_file = Path(OUTPUT_DIR) / input_filename.replace('.csv', '_cleaned.txt').replace('.txt', '_cleaned.txt')
    
    print(f"Input:  {INPUT_FILE}")
    print(f"Type:   {'CSV' if is_csv else 'Plain Text'}")
    print(f"Output: {output_file}\n")
    
    # ========================================================================
    # STEP 1: Load existing deduplication index (if it exists)
    # ========================================================================
    
    pipeline = JapaneseTextCleaningPipeline(config)
    
    if Path(DEDUP_INDEX_FILE).exists():
        print("="*80)
        print("✓ Found existing deduplication index - loading...")
        print("="*80)
        
        try:
            with open(DEDUP_INDEX_FILE, 'rb') as f:
                dedup_state = pickle.load(f)
            
            # Check for version compatibility
            version = dedup_state.get('version', 1)
            hash_function = dedup_state.get('hash_function', 'unknown')
            
            print(f"  Checkpoint version: {version}")
            print(f"  Hash function: {hash_function}")
            
            # Check compatibility
            current_hash_func = pipeline.dedup.hash_function
            if hash_function != current_hash_func and hash_function != 'unknown':
                print(f"\n⚠ WARNING: Hash function mismatch!")
                print(f"  Checkpoint: {hash_function}")
                print(f"  Current: {current_hash_func}")
                response = input("Continue anyway? (yes/no): ")
                if response.lower() != 'yes':
                    print("Aborted.")
                    return
            
            # Restore the xxHash deduplication state
            pipeline.dedup.seen_hashes = dedup_state['seen_hashes']
            pipeline.dedup.doc_count = dedup_state['doc_count']
            pipeline.dedup.seed = dedup_state['seed']
            
            print(f"\n✓ Loaded deduplication index from: {DEDUP_INDEX_FILE}")
            print(f"  Previously processed unique documents: {dedup_state['doc_count']:,}")
            
            # Show previously processed files
            processed_files = dedup_state.get('processed_files', [])
            if processed_files:
                print(f"  Previously processed files: {len(processed_files)}")
                print(f"\n  Last 5 processed files:")
                for i, fname in enumerate(processed_files[-5:], 1):
                    print(f"    {i}. {Path(fname).name}")
                if len(processed_files) > 5:
                    print(f"    ... and {len(processed_files) - 5} more")
            
            print("\nThis file will be checked against all previously processed documents!\n")
            
        except Exception as e:
            print(f"✗ Warning: Could not load deduplication index: {e}")
            print("Starting with fresh index...\n")
            import traceback
            traceback.print_exc()
    else:
        print("="*80)
        print("No existing deduplication index found")
        print("="*80)
        print("This appears to be the FIRST file you're processing.")
        print("A new deduplication index will be created.\n")
    
    # Track stats before processing
    docs_before = pipeline.dedup.doc_count
    
    # ========================================================================
    # STEP 2: Process the file
    # ========================================================================
    
    print("="*80)
    print("Processing file...")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    try:
        if is_csv:
            # Process CSV file
            pipeline.clean_csv_file(
                input_file=INPUT_FILE,
                output_file=str(output_file),
                buffer_size=10000,
                progress_interval=50000
            )
        else:
            # Process plain text file
            pipeline.clean_file_streaming(
                input_file=INPUT_FILE,
                output_file=str(output_file),
                buffer_size=10000,
                progress_interval=50000
            )
        
        elapsed = time.time() - start_time
        
        # Calculate statistics for this file
        docs_after = pipeline.dedup.doc_count
        docs_added = docs_after - docs_before
        duplicates_found = pipeline.stats.exact_duplicates
        lines_per_sec = pipeline.stats.total_processed / elapsed if elapsed > 0 else 0
        
        print(f"\n{'='*80}")
        print("✓ Processing Complete!")
        print("="*80)
        print(f"Time taken: {elapsed/60:.1f} minutes ({elapsed:.1f} seconds)")
        print(f"Processing speed: {lines_per_sec:,.0f} lines/second")
        
        print(f"\nResults for this file:")
        print(f"  Total lines processed: {pipeline.stats.total_processed:,}")
        print(f"  Lines accepted: {pipeline.stats.accepted:,}")
        print(f"  Duplicates found: {duplicates_found:,}")
        print(f"  New unique documents added: {docs_added:,}")
        
        print(f"\nCumulative statistics (across ALL files processed):")
        print(f"  Total unique documents: {docs_after:,}")
        
        print(f"\nRejection breakdown:")
        print("-"*80)
        stats_dict = pipeline.stats.to_dict()
        print(f"  Exact duplicates: {stats_dict['exact_duplicates']}")
        print(f"  Too short: {stats_dict['too_short']}")
        print(f"  Low Japanese ratio: {stats_dict['low_japanese_ratio']}")
        print(f"  Empty after cleaning: {stats_dict['empty_after_cleaning']}")
        print(f"  CSV parse errors: {stats_dict.get('csv_parse_errors', 0)}")
        print(f"  Encoding errors: {stats_dict.get('encoding_errors', 0)}")
        
        # Show quality rejections if available
        if 'quality_rejections' in stats_dict:
            print(f"\n  Quality filter rejections:")
            for reason, count in stats_dict['quality_rejections'].items():
                print(f"    {reason}: {count:,}")
        
        print(f"\n  Acceptance rate: {stats_dict['acceptance_rate']}")
        print("-"*80 + "\n")
        
    except Exception as e:
        print(f"✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # STEP 3: Save the updated deduplication index
    # ========================================================================
    
    print("="*80)
    print("Saving deduplication index for future runs...")
    print("="*80)
    
    try:
        # Get list of previously processed files
        if Path(DEDUP_INDEX_FILE).exists():
            try:
                with open(DEDUP_INDEX_FILE, 'rb') as f:
                    old_state = pickle.load(f)
                    processed_files = old_state.get('processed_files', [])
            except:
                processed_files = []
        else:
            processed_files = []
        
        # Add current file to the list
        if INPUT_FILE not in processed_files:
            processed_files.append(INPUT_FILE)
        
        # Save the xxHash deduplication state with versioning
        dedup_state = {
            'seen_hashes': pipeline.dedup.seen_hashes,
            'doc_count': pipeline.dedup.doc_count,
            'seed': pipeline.dedup.seed,
            'hash_function': pipeline.dedup.hash_function,
            'version': 1,  # Checkpoint version
            'processed_files': processed_files,
        }
        
        with open(DEDUP_INDEX_FILE, 'wb') as f:
            pickle.dump(dedup_state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Calculate checkpoint file size
        checkpoint_size_mb = Path(DEDUP_INDEX_FILE).stat().st_size / (1024 ** 2)
        
        print(f"✓ Deduplication index saved to: {DEDUP_INDEX_FILE}")
        print(f"  Checkpoint size: {checkpoint_size_mb:.2f} MB")
        print(f"  Total unique documents: {dedup_state['doc_count']:,}")
        print(f"  Hash function: {dedup_state['hash_function']}")
        print(f"  Files processed so far: {len(processed_files)}")
        
        # Show all processed files
        if len(processed_files) <= 10:
            print(f"\n  All processed files:")
            for i, fname in enumerate(processed_files, 1):
                print(f"    {i}. {Path(fname).name}")
        else:
            print(f"\n  Last 10 processed files:")
            for i, fname in enumerate(processed_files[-10:], len(processed_files)-9):
                print(f"    {i}. {Path(fname).name}")
        
    except Exception as e:
        print(f"✗ Warning: Could not save deduplication index: {e}")
        print("You may lose deduplication context for the next file!")
        import traceback
        traceback.print_exc()
    
    # ========================================================================
    # Show preview
    # ========================================================================
    
    print("\n" + "="*80)
    print("Preview of cleaned output (first 5 lines):")
    print("="*80)
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(f"{i+1}. {line.strip()}")
    except Exception as e:
        print(f"Could not read preview: {e}")
    
    print("\n" + "="*80)
    print("✓ All Done!")
    print("="*80)
    print(f"\nCleaned file saved to: {output_file}")
    print(f"Deduplication index saved to: {DEDUP_INDEX_FILE}")
    print(f"\nPerformance summary:")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Speed: {lines_per_sec:,.0f} lines/sec")
    print(f"  Acceptance rate: {stats_dict['acceptance_rate']}")
    print(f"  New unique docs: {docs_added:,}")
    print(f"  Total unique docs: {docs_after:,}")
    
    print("\n" + "="*80)
    print("To process the next file:")
    print("="*80)
    print("1. Edit INPUT_FILE in this script")
    print("2. Run: python datacleaning_single_xxhash.py")
    print("3. The new file will be checked against all previous files!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()