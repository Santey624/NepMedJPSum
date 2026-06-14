"""
Nepali Text Data Cleaner
A comprehensive cleaning pipeline for Nepali (Devanagari) text data
"""

import re
import unicodedata
import pandas as pd
from pathlib import Path
from typing import List, Set, Optional
import emoji


class NepaliTextCleaner:
    """Advanced Nepali text cleaning with Devanagari script support"""
    
    # Devanagari Unicode ranges
    DEVANAGARI_RANGE = r'[\u0900-\u097F\u1CD0-\u1CFF\u200C\u200D]'
    
    # Common Nepali punctuation to keep
    NEPALI_PUNCTUATION = '।॥'
    
    # English and common punctuation to keep
    KEEP_PUNCTUATION = '.,!?;:\'"()-'
    
    def __init__(self, keep_english: bool = True, keep_numbers: bool = True, 
                 remove_urls: bool = True, remove_emails: bool = True, 
                 remove_phones: bool = False):
        """
        Initialize the cleaner
        
        Args:
            keep_english: Whether to keep English characters
            keep_numbers: Whether to keep numeric digits
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            remove_phones: Whether to remove phone numbers
        """
        self.keep_english = keep_english
        self.keep_numbers = keep_numbers
        self.remove_urls_flag = remove_urls
        self.remove_emails_flag = remove_emails
        self.remove_phones_flag = remove_phones
        self.seen_texts: Set[str] = set()
        
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode using NFD then NFC for Devanagari consistency
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # NFD normalization (decompose combined characters)
        text = unicodedata.normalize('NFD', text)
        # NFC normalization (compose characters back)
        text = unicodedata.normalize('NFC', text)
        
        return text
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text
        
        Args:
            text: Input text
            
        Returns:
            Text without URLs
        """
        if not text:
            return ""
        
        # Pattern for URLs (http, https, www, ftp)
        url_pattern = re.compile(
            r'(?:(?:https?|ftp):\/\/)?'  # Optional protocol
            r'(?:www\.)?'  # Optional www
            r'(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}'  # Domain
            r'(?::[0-9]{1,5})?'  # Optional port
            r'(?:\/[^\s]*)?',  # Optional path
            flags=re.IGNORECASE
        )
        text = url_pattern.sub('', text)
        
        return text
    
    def remove_emails_func(self, text: str) -> str:
        """
        Remove email addresses from text
        
        Args:
            text: Input text
            
        Returns:
            Text without email addresses
        """
        if not text:
            return ""
        
        # Pattern for email addresses
        email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        text = email_pattern.sub('', text)
        
        return text
    
    def remove_phone_numbers(self, text: str) -> str:
        """
        Remove phone numbers (Nepali format)
        
        Args:
            text: Input text
            
        Returns:
            Text without phone numbers
        """
        if not text:
            return ""
        
        # Nepali phone patterns: 
        # - Mobile: 98########, 97########, etc.
        # - Landline: 01-#######
        phone_pattern = re.compile(
            r'\b(?:'
            r'(?:98|97|96|95|94|93|92|91|90)\d{8}|'  # Mobile numbers
            r'(?:01|02|03|04|05|06|07|08|09)-?\d{7}|'  # Landline with area code
            r'\+977[-\s]?\d{10}|'  # International format
            r'977[-\s]?\d{10}'  # Without plus
            r')\b'
        )
        text = phone_pattern.sub('', text)
        
        return text
    
    def remove_emojis(self, text: str) -> str:
        """
        Remove all emojis from text
        
        Args:
            text: Input text
            
        Returns:
            Text without emojis
        """
        if not text:
            return ""
        
        # Remove emojis using emoji library
        text = emoji.replace_emoji(text, replace='')
        
        # Additional emoji patterns (for fallback)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", 
            flags=re.UNICODE
        )
        text = emoji_pattern.sub(r'', text)
        
        return text
    
    def clean_devanagari_chars(self, text: str) -> str:
        """
        Clean and standardize Devanagari characters
        
        Args:
            text: Input text
            
        Returns:
            Cleaned Devanagari text
        """
        if not text:
            return ""
        
        # Remove zero-width characters except ZWNJ and ZWJ (important for Devanagari)
        text = re.sub(r'[\u200B\u200E\u200F\uFEFF]', '', text)
        
        # Fix common Devanagari character issues
        replacements = {
            '\u0950': 'ॐ',  # OM symbol normalization
            '\u093D': 'ऽ',  # Avagraha normalization
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def remove_unwanted_symbols(self, text: str) -> str:
        """
        Remove unwanted symbols while keeping valid punctuation
        
        Args:
            text: Input text
            
        Returns:
            Text with unwanted symbols removed
        """
        if not text:
            return ""
        
        # Build pattern for allowed characters
        allowed_chars = self.DEVANAGARI_RANGE
        
        if self.keep_english:
            allowed_chars += r'a-zA-Z'
        
        if self.keep_numbers:
            allowed_chars += r'0-9\u0966-\u096F'  # Include Devanagari digits
        
        # Add punctuation
        allowed_chars += re.escape(self.KEEP_PUNCTUATION + self.NEPALI_PUNCTUATION)
        
        # Add whitespace
        allowed_chars += r'\s'
        
        # Remove everything else
        pattern = f'[^{allowed_chars}]'
        text = re.sub(pattern, ' ', text)
        
        return text
    
    def normalize_spaces(self, text: str) -> str:
        """
        Remove extra spaces and normalize whitespace
        
        Args:
            text: Input text
            
        Returns:
            Text with normalized spacing
        """
        if not text:
            return ""
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing spaces
        text = text.strip()
        
        # Fix spacing around punctuation
        text = re.sub(r'\s*([।॥.,!?;:])\s*', r'\1 ', text)
        text = re.sub(r'\s+$', '', text)  # Remove trailing space after punctuation cleanup
        
        return text
    
    def is_incomplete_sentence(self, text: str, min_words: int = 3) -> bool:
        """
        Check if sentence is incomplete
        
        Args:
            text: Input text
            min_words: Minimum number of words for a complete sentence
            
        Returns:
            True if sentence is incomplete
        """
        if not text or not text.strip():
            return True
        
        # Count words (split by spaces)
        words = text.strip().split()
        
        if len(words) < min_words:
            return True
        
        # Check if ends with proper punctuation
        nepali_end_punct = ['।', '॥', '.', '!', '?']
        has_end_punct = any(text.strip().endswith(p) for p in nepali_end_punct)
        
        # If very short and no ending punctuation, consider incomplete
        if len(words) < 5 and not has_end_punct:
            return True
        
        return False
    
    def clean_text(self, text: str, remove_incomplete: bool = True) -> str:
        """
        Apply all cleaning steps to text
        
        Args:
            text: Input text
            remove_incomplete: Whether to remove incomplete sentences
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str) or text.strip() == '':
            return ""
        
        # Step 1: Unicode normalization
        text = self.normalize_unicode(text)
        
        # Step 2: Remove URLs (before emoji removal to catch emoji in URLs)
        if self.remove_urls_flag:
            text = self.remove_urls(text)
        
        # Step 3: Remove emails
        if self.remove_emails_flag:
            text = self.remove_emails_func(text)
        
        # Step 4: Remove phone numbers
        if self.remove_phones_flag:
            text = self.remove_phone_numbers(text)
        
        # Step 5: Remove emojis
        text = self.remove_emojis(text)
        
        # Step 6: Clean Devanagari characters
        text = self.clean_devanagari_chars(text)
        
        # Step 7: Remove unwanted symbols
        text = self.remove_unwanted_symbols(text)
        
        # Step 8: Normalize spaces
        text = self.normalize_spaces(text)
        
        # Step 9: Check for incomplete sentences
        if remove_incomplete and self.is_incomplete_sentence(text):
            return ""
        
        return text
    
    def is_duplicate(self, text: str) -> bool:
        """
        Check if text is a duplicate
        
        Args:
            text: Input text
            
        Returns:
            True if duplicate
        """
        if not text:
            return True
        
        text_normalized = text.lower().strip()
        
        if text_normalized in self.seen_texts:
            return True
        
        self.seen_texts.add(text_normalized)
        return False
    
    def clean_dataframe(
        self, 
        df: pd.DataFrame, 
        text_column: str = 'text',
        remove_duplicates: bool = True,
        remove_incomplete: bool = True,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Clean a pandas DataFrame containing Nepali text
        
        Args:
            df: Input DataFrame
            text_column: Name of the column containing text
            remove_duplicates: Whether to remove duplicate rows
            remove_incomplete: Whether to remove incomplete sentences
            inplace: Whether to modify DataFrame in place
            
        Returns:
            Cleaned DataFrame
        """
        if not inplace:
            df = df.copy()
        
        # Reset seen texts for this DataFrame
        self.seen_texts.clear()
        
        # Handle missing data - fill with empty string
        df[text_column] = df[text_column].fillna('')
        
        # Apply cleaning
        print(f"Cleaning {len(df)} rows...")
        df[text_column] = df[text_column].apply(
            lambda x: self.clean_text(x, remove_incomplete=remove_incomplete)
        )
        
        # Remove rows with empty text after cleaning
        initial_count = len(df)
        df = df[df[text_column].str.strip() != '']
        removed_empty = initial_count - len(df)
        print(f"Removed {removed_empty} empty/incomplete rows")
        
        # Remove duplicates
        if remove_duplicates:
            initial_count = len(df)
            # Mark duplicates
            df['is_duplicate'] = df[text_column].apply(self.is_duplicate)
            df = df[~df['is_duplicate']]
            df = df.drop(columns=['is_duplicate'])
            removed_dupes = initial_count - len(df)
            print(f"Removed {removed_dupes} duplicate rows")
        
        # Reset index
        df = df.reset_index(drop=True)
        
        print(f"Final dataset: {len(df)} rows")
        
        return df
    
    def clean_csv(
        self,
        input_path: str,
        output_path: str,
        text_column: str = 'text',
        remove_duplicates: bool = True,
        remove_incomplete: bool = True
    ) -> None:
        """
        Clean a CSV file containing Nepali text
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save cleaned CSV file
            text_column: Name of the column containing text
            remove_duplicates: Whether to remove duplicates
            remove_incomplete: Whether to remove incomplete sentences
        """
        print(f"Reading CSV from: {input_path}")
        
        # Read CSV
        df = pd.read_csv(input_path)
        
        print(f"Original dataset: {len(df)} rows")
        
        # Clean DataFrame
        df_cleaned = self.clean_dataframe(
            df,
            text_column=text_column,
            remove_duplicates=remove_duplicates,
            remove_incomplete=remove_incomplete,
            inplace=False
        )
        
        # Save cleaned CSV
        df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Cleaned CSV saved to: {output_path}")
        
        # Print statistics
        print("\n" + "="*50)
        print("CLEANING STATISTICS")
        print("="*50)
        print(f"Original rows: {len(df)}")
        print(f"Cleaned rows: {len(df_cleaned)}")
        print(f"Removed rows: {len(df) - len(df_cleaned)}")
        print(f"Retention rate: {len(df_cleaned)/len(df)*100:.2f}%")
        print("="*50)


def main():
    """Example usage"""
    
    # Initialize cleaner
    cleaner = NepaliTextCleaner(
        keep_english=True,  # Keep English characters
        keep_numbers=True,   # Keep numbers
        remove_urls=True,    # Remove URLs
        remove_emails=True,  # Remove emails
        remove_phones=False  # Keep phone numbers
    )
    
    # Example: Clean single text
    print("EXAMPLE 1: Single Text Cleaning")
    print("-" * 50)
    
    sample_text = "  नेपाल  🇳🇵  एक   सुन्दर देश हो  ।।   This is Nepal!!!  https://example.com test@email.com 9812345678  "
    print(f"Original: {repr(sample_text)}")
    
    cleaned = cleaner.clean_text(sample_text)
    print(f"Cleaned: {repr(cleaned)}")
    print()
    
    # Example: Clean CSV file
    print("EXAMPLE 2: CSV File Cleaning")
    print("-" * 50)
    
    # Check if a CSV file exists
    import sys
    
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2] if len(sys.argv) > 2 else "cleaned_output.csv"
        text_col = sys.argv[3] if len(sys.argv) > 3 else "text"
        
        cleaner.clean_csv(
            input_path=input_csv,
            output_path=output_csv,
            text_column=text_col,
            remove_duplicates=True,
            remove_incomplete=True
        )
    else:
        print("Usage: python nepali_text_cleaner.py <input.csv> [output.csv] [text_column]")
        print("\nExample:")
        print("  python nepali_text_cleaner.py data.csv cleaned_data.csv text")


if __name__ == "__main__":
    main()