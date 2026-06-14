#!/usr/bin/env python3
"""
Diagnostic script to check mT5 tokenizer loading
"""

import sys
print("Python version:", sys.version)
print("=" * 80)

print("\n1. Checking if transformers is installed...")
try:
    import transformers
    print(f"✓ transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"✗ transformers not found: {e}")
    sys.exit(1)

print("\n2. Checking if sentencepiece is installed...")
try:
    import sentencepiece
    print(f"✓ sentencepiece is installed")
except ImportError as e:
    print(f"✗ sentencepiece not found: {e}")
    sys.exit(1)

print("\n3. Checking cache directory...")
import os
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
print(f"Cache directory: {cache_dir}")
print(f"Cache exists: {os.path.exists(cache_dir)}")

if os.path.exists(cache_dir):
    models = [d for d in os.listdir(cache_dir) if 'mt5' in d.lower()]
    print(f"mT5 models in cache: {models}")

print("\n4. Attempting to load tokenizer with verbose output...")
print("This may take a while if downloading...")
print("-" * 80)

try:
    from transformers import AutoTokenizer
    print("Calling AutoTokenizer.from_pretrained()...")
    tokenizer = AutoTokenizer.from_pretrained(
        "google/mt5-base", 
        use_fast=False,
        verbose=True
    )
    print("✓ Tokenizer loaded successfully!")
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Vocab size: {len(tokenizer)}")
    
    # Quick test
    print("\n5. Quick tokenization test...")
    test_text = "नमस्ते"
    tokens = tokenizer.tokenize(test_text)
    print(f"Text: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"✓ Tokenization works!")
    
except KeyboardInterrupt:
    print("\n✗ Interrupted by user (Ctrl+C)")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("All checks passed! Your tokenizer is working correctly.")
print("=" * 80)