from __future__ import unicode_literals
import re
import unicodedata

def unicode_normalize(cls, s):
    """Normalize specific character classes to NFC form"""
    pt = re.compile('([{}]+)'.format(cls))
    
    def norm(c):
        return unicodedata.normalize('NFC', c) if pt.match(c) else c
    
    s = ''.join(norm(x) for x in re.split(pt, s))
    return s

def remove_extra_spaces(s):
    """Remove extra spaces, handling Devanagari script appropriately"""
    # Normalize all whitespace types to single space
    s = re.sub('[ \u00A0\u2000-\u200D\u202F\u205F\u3000\uFEFF]+', ' ', s)
    
    # Devanagari blocks
    devanagari_blocks = ''.join((
        '\u0900-\u097F',  # Devanagari
        '\u1CD0-\u1CFF',  # Vedic Extensions
        '\uA8E0-\uA8FF',  # Devanagari Extended
    ))
    
    basic_latin = '\u0000-\u007F'
    
    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s
    
    # Remove spaces between Devanagari characters
    s = remove_space_between(devanagari_blocks, devanagari_blocks, s)
    s = remove_space_between(devanagari_blocks, basic_latin, s)
    s = remove_space_between(basic_latin, devanagari_blocks, s)
    
    return s

def normalize_punctuation(s):
    """Normalize various punctuation marks"""
    # Normalize danda variations
    s = re.sub('[|｜]', '।', s)
    s = re.sub('।+', '।', s)  # Multiple dandas to single
    
    # Normalize ellipsis
    s = re.sub('\.{3,}|…+', '...', s)
    
    # Normalize hyphens and dashes
    s = re.sub('[˗֊‐‑‒–—⁃⁻₋−‾﹘﹣－]+', '-', s)
    
    # Normalize brackets
    s = re.sub('[（]', '(', s)
    s = re.sub('[）]', ')', s)
    s = re.sub('[［]', '[', s)
    s = re.sub('[］]', ']', s)
    s = re.sub('[｛]', '{', s)
    s = re.sub('[｝]', '}', s)
    
    # Normalize quotation marks
    s = re.sub("['‚‛']", "'", s)
    s = re.sub('[""„‟"]', '"', s)
    
    # Normalize exclamation and question marks
    s = re.sub('[！]', '!', s)
    s = re.sub('[？]', '?', s)
    
    return s

def fix_common_devanagari_issues(s):
    """Fix common Devanagari encoding/rendering issues"""
    # Remove invisible formatting characters
    s = re.sub('[\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]', '', s)
    
    # Remove replacement characters and invalid Unicode
    s = re.sub('[\uFFFD\uFFFE\uFFFF]', '', s)
    
    # Normalize multiple consecutive matras (usually errors)
    # Example: े े -> े
    matras = '\u093E-\u094F\u0955-\u0957\u0962-\u0963'
    s = re.sub('([{}])\\1+'.format(matras), r'\1', s)
    
    # Fix spacing around punctuation
    s = re.sub(r'\s+([।॥,.!?;:])', r'\1', s)
    s = re.sub(r'([।॥])\s*([।॥])', r'\1\2', s)
    
    # Remove spaces before Devanagari conjuncts (halant combinations)
    s = re.sub(r'\s+्', '्', s)
    
    return s

def normalize_numbers(s):
    """Normalize number representations"""
    # Optional: Convert Devanagari numerals to ASCII
    # Uncomment if you want this behavior
    # devanagari_to_ascii = str.maketrans('०१२३४५६७८९', '0123456789')
    # s = s.translate(devanagari_to_ascii)
    
    # Normalize fullwidth numbers
    fullwidth_to_ascii = str.maketrans('０１２３４５６７８９', '0123456789')
    s = s.translate(fullwidth_to_ascii)
    
    return s

def normalize_nepali(s, preserve_devanagari_numbers=True):
    """
    Normalize Nepali text with focus on Devanagari script.
    Uses NFC normalization to maintain proper combining character sequences.
    
    Args:
        s: Input string
        preserve_devanagari_numbers: If True, keep Devanagari numerals (०-९)
    """
    if not s or not isinstance(s, str):
        return s
    
    # Strip leading/trailing whitespace
    s = s.strip()
    
    if not s:
        return s
    
    # Early NFC normalization to fix encoding issues
    s = unicodedata.normalize('NFC', s)
    
    # Fix common Devanagari-specific issues
    s = fix_common_devanagari_issues(s)
    
    # Normalize fullwidth ASCII to halfwidth
    s = unicode_normalize('Ａ-Ｚａ-ｚ', s)
    
    # Normalize numbers
    if not preserve_devanagari_numbers:
        devanagari_to_ascii = str.maketrans('०१२३४५६७८९', '0123456789')
        s = s.translate(devanagari_to_ascii)
    
    # Normalize fullwidth numbers regardless
    fullwidth_to_ascii = str.maketrans('０１२３４５６７８９', '0123456789')
    s = s.translate(fullwidth_to_ascii)
    
    # Normalize punctuation
    s = normalize_punctuation(s)
    
    # Remove extra spaces
    s = remove_extra_spaces(s)
    
    # Remove multiple consecutive spaces
    s = re.sub(' {2,}', ' ', s)
    
    # Final NFC normalization
    s = unicodedata.normalize('NFC', s)
    
    # Final trim
    s = s.strip()
    
    return s

if __name__ == "__main__":
    # Original test cases
    assert "0123456789" == normalize_nepali("０１२３४५６７８९")
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" == normalize_nepali("ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ")
    
    # Devanagari text
    assert "नमस्ते" == normalize_nepali("नमस्ते")
    assert "नेपाल" == normalize_nepali("नेपाल")
    
    # Extra spaces
    assert "नमस्ते संसार" == normalize_nepali("नमस्ते　　　संसार")
    assert "नेपाल" == normalize_nepali("  नेपाल  ")
    assert "नेपाल सरकार" == normalize_nepali("नेपाल     सरकार")
    
    # Mixed content
    assert "नेपाल 2024" == normalize_nepali("नेपाल　２０２４")
    assert "काठमाडौं-पोखरा" == normalize_nepali("काठमाडौं－पोखरा")
    
    # Quotation marks
    assert "नेपाल 'देश'" == normalize_nepali("नेपाल 'देश'")
    assert 'नेपाल "देश"' == normalize_nepali('नेपाल "देश"')
    
    # Zero-width characters
    assert "नमस्ते" == normalize_nepali("नमस्‍ते")
    
    # New test cases
    # Punctuation normalization
    assert "नेपाल।" == normalize_nepali("नेपाल |")
    assert "नेपाल।" == normalize_nepali("नेपाल।।।")
    assert "के हो?" == normalize_nepali("के हो？")
    assert "वाह!" == normalize_nepali("वाह！")
    
    # Brackets
    assert "नेपाल (देश)" == normalize_nepali("नेपाल（देश）")
    
    # Multiple spaces
    assert "नेपाल सरकार" == normalize_nepali("नेपाल        सरकार")
    
    # Ellipsis
    assert "नेपाल..." == normalize_nepali("नेपाल…")
    
    print("All tests passed!")