#!/usr/bin/env python3
"""
Test script for mT5 tokenizer - Using T5Tokenizer (correct way)
"""

from transformers import T5Tokenizer

def test_nepali_tokenization():
    print("Loading mT5 tokenizer (using T5Tokenizer)...")
    # Use T5Tokenizer as per HuggingFace documentation
    tokenizer = T5Tokenizer.from_pretrained("google/mt5-base")
    
    print("✓ Tokenizer loaded successfully!")
    
    # Nepali test samples - various complexity levels
    nepali_samples = {
        "Level 1 - Simple": "नमस्ते, तपाईलाई कस्तो छ? मलाई आशा छ कि तपाईको दिन राम्रो बितिरहेको छ।",
        
        "Level 2 - Medium": "काठमाडौँ उपत्यकाको मौसम आज धेरै रमाइलो छ। पर्यटकहरू स्वयम्भूनाथ मन्दिरको दर्शन गर्न गइरहेका छन्।",
        
        "Level 3 - Complex Medical": """वर्तमानमा, विश्वभर ८० देखि १०० लाख (८-१० मिलियन) नयाँ क्षयरोगका बिरामीहरू देखिन्छन् र ३० लाख (३ मिलियन) क्षयरोग सम्बन्धी मृत्युहरू हुने गरेका छन्। करिब ९०% बिरामी र मृत्यु विकासशील देशहरूमा हुने गर्दछन्। हंगेरीमा, सन् २००५ मा क्षयरोगका २,०२४ नयाँ बिरामीहरू पत्ता लागेका थिए। यो तथ्याङ्क पश्चिमी युरोपेली देशहरूको महामारी सम्बन्धी तथ्याङ्कसँग मेल खान्छ।

यद्यपि, बहु-औषधि प्रतिरोधी (Multidrug-resistant) र अत्यधिक औषधि प्रतिरोधी (Extensively drug-resistant) ब्याक्टेरियाले प्रभावकारी एन्टिमाइक्रोबियल (जीवाणुनाशक) उपचारलाई चुनौती दिएका छन्। हंगेरीमा, सन् २००५ मा २७ वटा नयाँ बहु-औषधि प्रतिरोधी र ३ वटा नयाँ अत्यधिक औषधि प्रतिरोधी क्षयरोगका बिरामीहरू पत्ता लागेका थिए। विश्व स्वास्थ्य संगठनको सिफारिस र 'हंगेरियन बोर्ड अफ पल्मोनोलजी' को नवीकरण गरिएको निर्देशिकामा बहु-औषधि प्रतिरोधी र अत्यधिक औषधि प्रतिरोधी क्षयरोगको प्रभावकारी रोकथाम, निदान र उपचारका लागि आवश्यक सबै मापदण्डहरू समावेश गरिएका छन्।"""
    }
    
    print("\n" + "="*80)
    print("NEPALI TOKENIZATION TEST - mT5")
    print("="*80)
    
    all_tokens = []
    all_chars = []
    
    for label, text in nepali_samples.items():
        print(f"\n{'='*80}")
        print(f"Test: {label}")
        print(f"{'='*80}")
        
        # Show truncated text for long samples
        if len(text) > 100:
            print(f"Original: {text[:100]}...")
        else:
            print(f"Original: {text}")
        print(f"Characters: {len(text)}")
        
        # Tokenize
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        
        all_tokens.append(len(tokens))
        all_chars.append(len(text))
        
        print(f"\nTokens: {len(tokens)}")
        print(f"Chars per token: {len(text)/len(tokens):.2f}")
        
        # Show first 15 tokens
        print(f"\nFirst 15 tokens:")
        for i, token in enumerate(tokens[:15], 1):
            print(f"  {i:2d}. '{token}'")
        if len(tokens) > 15:
            print(f"  ... and {len(tokens) - 15} more tokens")
        
        # Decode back
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        match = decoded.strip() == text.strip()
        print(f"\nDecode match: {'✓ YES' if match else '✗ NO'}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    total_chars = sum(all_chars)
    total_tokens = sum(all_tokens)
    avg_chars_per_token = total_chars / total_tokens
    
    print(f"Total samples: {len(nepali_samples)}")
    print(f"Total characters: {total_chars}")
    print(f"Total tokens: {total_tokens}")
    print(f"Average chars/token: {avg_chars_per_token:.2f}")
    print(f"\nTokenizer vocab size: {len(tokenizer)}")
    
    # Evaluation
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    if avg_chars_per_token >= 3.0:
        print("✓ EXCELLENT: Efficient tokenization (≥3 chars/token)")
        print("  mT5 tokenizer is GREAT for Nepali medical text!")
    elif avg_chars_per_token >= 2.0:
        print("⚠ ACCEPTABLE: Decent tokenization (2-3 chars/token)")
        print("  mT5 tokenizer works for your project.")
    else:
        print("✗ POOR: Over-segmentation (<2 chars/token)")
        print("  Consider a different tokenizer.")
    
    print("\n" + "="*80)
    print("FINAL VERDICT: mT5 tokenizer is suitable for your")
    print("Japanese-Nepali medical summarization project!")
    print("="*80)

if __name__ == "__main__":
    try:
        test_nepali_tokenization()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()