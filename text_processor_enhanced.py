"""
Hybrid LLM Text Assistant - Enhanced Version
--------------------------------------------
A comprehensive NLP system combining:
- N-gram language modeling (statistical baseline)
- BERT for context-aware typo correction (MLM)
- DistilGPT2 for fluent text generation and completion
- Ensemble methods for improved accuracy

Authors: El Guelta Mohamad Saber, El Hadifi Soukaina
Enhanced with Transformer models - 2026
"""

import re
import numpy as np
import random
import multiprocessing
import tempfile
import os
import time
import torch
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from transformers import (
    BertTokenizer, BertForMaskedLM,
    GPT2LMHeadModel, GPT2Tokenizer,
    pipeline
)


class TransformerModels:
    """Manages BERT and DistilGPT2 models for text processing."""
    
    def __init__(self, device: str = None):
        """
        Initialize transformer models.
        
        Args:
            device: Device to run models on ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # Initialize BERT for masked language modeling (typo correction)
        print("📚 Loading BERT model for typo correction...")
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.bert_model.to(self.device)
        self.bert_model.eval()
        
        # Initialize DistilGPT2 for text generation
        print("🤖 Loading DistilGPT2 model for text generation...")
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('distilgpt2')
        self.gpt2_model.to(self.device)
        self.gpt2_model.eval()
        
        # Set padding token
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        
        print("✅ Transformer models loaded successfully!")
    
    def bert_predict_masked_word(self, text: str, mask_position: int = -1, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Use BERT to predict masked words for typo correction.
        
        Args:
            text: Input text with [MASK] token or text where last word should be masked
            mask_position: Position to mask (-1 for last word)
            top_k: Number of top predictions to return
            
        Returns:
            List of (word, probability) tuples
        """
        # Tokenize and prepare input
        words = text.split()
        if '[MASK]' not in text and mask_position == -1:
            words[-1] = '[MASK]'
            text = ' '.join(words)
        
        # Encode and predict
        inputs = self.bert_tokenizer(text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            predictions = outputs.logits
        
        # Get masked token position
        mask_token_index = torch.where(inputs['input_ids'] == self.bert_tokenizer.mask_token_id)[1]
        
        if len(mask_token_index) == 0:
            return []
        
        mask_token_logits = predictions[0, mask_token_index, :]
        
        # Get top k predictions
        top_tokens = torch.topk(mask_token_logits, top_k, dim=1)
        top_probs = torch.softmax(top_tokens.values, dim=1)
        
        results = []
        for token_id, prob in zip(top_tokens.indices[0], top_probs[0]):
            word = self.bert_tokenizer.decode([token_id]).strip()
            results.append((word, prob.item()))
        
        return results
    
    def bert_correct_word(self, word: str, context: str = "") -> str:
        """
        Correct a potentially misspelled word using BERT and context.
        
        Args:
            word: Word to correct
            context: Surrounding context (sentence)
            
        Returns:
            Corrected word
        """
        if context:
            # Replace the word with [MASK] in context
            masked_text = context.replace(word, '[MASK]')
        else:
            masked_text = f"The word is [MASK]."
        
        predictions = self.bert_predict_masked_word(masked_text, top_k=1)
        
        if predictions:
            return predictions[0][0]
        return word
    
    def gpt2_generate(self, prompt: str, max_length: int = 50, 
                     temperature: float = 0.7, top_p: float = 0.9,
                     num_return_sequences: int = 1) -> List[str]:
        """
        Generate text using DistilGPT2.
        
        Args:
            prompt: Starting text
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            num_return_sequences: Number of sequences to generate
            
        Returns:
            List of generated text sequences
        """
        inputs = self.gpt2_tokenizer(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.gpt2_model.generate(
                inputs['input_ids'],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=True,
                pad_token_id=self.gpt2_tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        generated_texts = [
            self.gpt2_tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return generated_texts
    
    def gpt2_complete(self, text: str, num_words: int = 5) -> str:
        """
        Auto-complete text using DistilGPT2.
        
        Args:
            text: Input text to complete
            num_words: Approximate number of words to add
            
        Returns:
            Completed text
        """
        # Estimate token length
        token_length = len(self.gpt2_tokenizer.encode(text)) + num_words * 2
        
        generated = self.gpt2_generate(
            text, 
            max_length=min(token_length, 100),
            temperature=0.7,
            num_return_sequences=1
        )
        
        return generated[0] if generated else text


class HybridTextProcessor:
    """
    Enhanced text processor combining n-gram models with BERT and DistilGPT2.
    """
    
    def __init__(self, corpus_path: str, keyboard_graph_path: str, 
                 ngram_size: int = 2, smoothing_k: float = 0.1, 
                 min_frequency: int = 2, use_transformers: bool = True):
        """
        Initialize the hybrid text processor.
        
        Args:
            corpus_path: Path to the text corpus
            keyboard_graph_path: Path to the keyboard layout graph
            ngram_size: Size of n-grams to use
            smoothing_k: Smoothing parameter for add-k smoothing
            min_frequency: Minimum word frequency for dictionary inclusion
            use_transformers: Whether to load transformer models
        """
        self.ngram_size = ngram_size
        self.smoothing_k = smoothing_k
        self.use_transformers = use_transformers
        
        print("📖 Loading data...")
        self.dictionary, self.word_frequencies = self._create_dictionary_from_corpus(
            corpus_path, min_frequency
        )
        self.keyboard_graph = self._load_keyboard_graph(keyboard_graph_path)
        
        print("🔧 Training n-gram model...")
        sample_file = self._extract_sample_for_training(corpus_path)
        self.ngram_counts = self._train(sample_file, self.dictionary, ngram_size, smoothing_k)
        os.unlink(sample_file)
        
        print(f"✅ N-gram model ready! {len(self.ngram_counts)} contexts.")
        
        # Initialize transformer models
        self.transformer_models = None
        if use_transformers:
            self.transformer_models = TransformerModels()
        
        # Performance tracking
        self.metrics = {
            'corrections': 0,
            'completions': 0,
            'generations': 0,
            'avg_correction_time': 0,
            'avg_completion_time': 0
        }
    
    # ========== DATA PREPARATION METHODS ==========
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Clean and tokenize text."""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.split()
    
    def _create_dictionary_from_corpus(self, corpus_file: str, 
                                      min_frequency: int = 2) -> Tuple[set, Counter]:
        """Create a dictionary from frequent words in the corpus."""
        word_counts = Counter()
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                words = self._preprocess_text(line)
                word_counts.update(words)
        
        dictionary = {word for word, count in word_counts.items() 
                     if count >= min_frequency}
        return dictionary, word_counts
    
    def _extract_sample_for_training(self, corpus_file: str, 
                                    sample_size: int = 100000) -> str:
        """Extract a sample from corpus for faster training."""
        with open(corpus_file, 'r', encoding='utf-8') as f:
            sample_text = f.read(sample_size)
        
        temp_sample = tempfile.NamedTemporaryFile(
            delete=False, mode='w', encoding='utf-8'
        )
        temp_sample.write(sample_text)
        temp_sample.close()
        return temp_sample.name
    
    def _prepare_data(self, infile: str, vocab: set) -> List[str]:
        """Read and prepare data for training."""
        with open(infile, 'r', encoding='utf-8') as f:
            tokens = self._preprocess_text(f.read())
        tokens = [w if w in vocab else '<UNK>' for w in tokens]
        return ['<s>'] + tokens + ['</s>']
    
    # ========== N-GRAM MODEL METHODS ==========
    
    def _train_worker(self, tokens: List[str], start: int, end: int, 
                     ngram_size: int) -> Tuple[defaultdict, defaultdict]:
        """Worker function for parallel n-gram counting."""
        local_ngram_counts = defaultdict(Counter)
        local_total_counts = defaultdict(int)
        
        for i in range(start, end - ngram_size + 1):
            context = tuple(tokens[i:i + ngram_size - 1])
            word = tokens[i + ngram_size - 1]
            local_ngram_counts[context][word] += 1
            local_total_counts[context] += 1
        
        return local_ngram_counts, local_total_counts
    
    def _train(self, infile: str, vocab: set, ngram_size: int = 2, 
              smoothing_k: float = 0.1) -> defaultdict:
        """Train the n-gram language model with parallel processing."""
        tokens = self._prepare_data(infile, vocab)
        n_workers = multiprocessing.cpu_count()
        chunk_size = len(tokens) // n_workers
        args = [
            (tokens, i, min(i + chunk_size, len(tokens)), ngram_size) 
            for i in range(0, len(tokens), chunk_size)
        ]
        
        with multiprocessing.Pool(n_workers) as pool:
            results = pool.starmap(self._train_worker, args)
        
        ngram_counts = defaultdict(Counter)
        total_counts = defaultdict(int)
        vocab_size = len(vocab) + 1
        
        for local_ngram_counts, local_total_counts in results:
            for context, words in local_ngram_counts.items():
                ngram_counts[context].update(words)
                total_counts[context] += local_total_counts[context]
        
        # Apply smoothing
        for context in ngram_counts:
            total = total_counts[context] + smoothing_k * vocab_size
            ngram_counts[context] = {
                word: (count + smoothing_k) / total
                for word, count in ngram_counts[context].items()
            }
        
        return ngram_counts
    
    # ========== KEYBOARD AND EDIT DISTANCE ==========
    
    def _load_keyboard_graph(self, file_path: str) -> Dict[str, set]:
        """Load keyboard layout graph for spelling correction."""
        adjacency = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                keys = line.strip().split()
                if keys:
                    adjacency[keys[0]] = set(keys[1:])
        return adjacency
    
    def _soundex(self, word: str) -> str:
        """Compute Soundex code for phonetic matching."""
        soundex_dict = {
            "bfpv": "1", "cgjkqsxz": "2", "dt": "3",
            "l": "4", "mn": "5", "r": "6"
        }
        
        word = word.lower()
        first_letter = word[0]
        encoded = first_letter.upper()
        
        letter_to_code = {}
        for chars, code in soundex_dict.items():
            for char in chars:
                letter_to_code[char] = code
        
        for char in word[1:]:
            if char in letter_to_code:
                code = letter_to_code[char]
                if encoded[-1] != code:
                    encoded += code
        
        encoded = encoded.ljust(4, "0")[:4]
        return encoded
    
    def _levenshtein_distance(self, s1: str, s2: str, 
                             keyboard_graph: Optional[Dict] = None) -> int:
        """Calculate Levenshtein edit distance between strings."""
        len_s1, len_s2 = len(s1), len(s2)
        dp = np.zeros((len_s1 + 1, len_s2 + 1))
        
        for i in range(len_s1 + 1):
            dp[i][0] = i
        for j in range(len_s2 + 1):
            dp[0][j] = j
        
        for i in range(1, len_s1 + 1):
            for j in range(1, len_s2 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                
                if (keyboard_graph and s1[i - 1] in keyboard_graph and 
                    s2[j - 1] in keyboard_graph.get(s1[i - 1], set())):
                    cost = 0.5
                
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + cost
                )
                
                if (i > 1 and j > 1 and s1[i - 1] == s2[j - 2] and 
                    s1[i - 2] == s2[j - 1]):
                    dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)
        
        return int(dp[len_s1, len_s2])
    
    # ========== SPELLING CORRECTION METHODS ==========
    
    def _get_ngram_corrections(self, word: str, k: int = 5) -> List[str]:
        """Get corrections using traditional methods."""
        filtered_dict = {
            w for w in self.dictionary 
            if abs(len(w) - len(word)) <= 2 and w[0] == word[0]
        }
        
        candidates = [
            (dict_word, self._levenshtein_distance(word, dict_word, self.keyboard_graph))
            for dict_word in filtered_dict
        ]
        candidates = sorted(candidates, key=lambda x: x[1])
        return [word for word, _ in candidates[:k]]
    
    def correct_word_hybrid(self, word: str, context: str = "", 
                           use_bert: bool = True) -> Tuple[str, str, float]:
        """
        Hybrid word correction using n-gram + BERT.
        
        Args:
            word: Word to correct
            context: Surrounding context
            use_bert: Whether to use BERT
            
        Returns:
            Tuple of (corrected_word, method_used, confidence)
        """
        start_time = time.time()
        
        # If word is in dictionary, return as is
        if word in self.dictionary:
            return word, 'dictionary', 1.0
        
        # Get n-gram suggestions
        ngram_suggestions = self._get_ngram_corrections(word, k=5)
        
        # Use BERT if available and requested
        if use_bert and self.use_transformers and self.transformer_models:
            if context:
                bert_correction = self.transformer_models.bert_correct_word(word, context)
                
                # If BERT suggestion is in dictionary, prefer it
                if bert_correction in self.dictionary:
                    self.metrics['corrections'] += 1
                    self.metrics['avg_correction_time'] = (
                        (self.metrics['avg_correction_time'] * (self.metrics['corrections'] - 1) +
                         (time.time() - start_time)) / self.metrics['corrections']
                    )
                    return bert_correction, 'bert', 0.9
        
        # Fallback to n-gram with frequency weighting
        if ngram_suggestions:
            suggestions_with_scores = [
                (s, self._levenshtein_distance(word, s, self.keyboard_graph) - 
                 0.1 * np.log(self.word_frequencies.get(s, 1)))
                for s in ngram_suggestions
            ]
            best_suggestion = min(suggestions_with_scores, key=lambda x: x[1])[0]
            
            self.metrics['corrections'] += 1
            self.metrics['avg_correction_time'] = (
                (self.metrics['avg_correction_time'] * (self.metrics['corrections'] - 1) +
                 (time.time() - start_time)) / self.metrics['corrections']
            )
            
            return best_suggestion, 'ngram', 0.7
        
        return word, 'none', 0.0
    
    def correct_text(self, text: str, use_bert: bool = True) -> Dict:
        """
        Correct spelling errors in text with detailed results.
        
        Args:
            text: Input text with potential errors
            use_bert: Whether to use BERT for correction
            
        Returns:
            Dictionary with corrected text and statistics
        """
        words = self._preprocess_text(text)
        corrected_words = []
        corrections_made = []
        
        for i, word in enumerate(words):
            # Get context (5 words before and after)
            context_start = max(0, i - 5)
            context_end = min(len(words), i + 6)
            context = ' '.join(words[context_start:i] + ['[MASK]'] + words[i+1:context_end])
            
            corrected, method, confidence = self.correct_word_hybrid(word, context, use_bert)
            corrected_words.append(corrected)
            
            if corrected != word:
                corrections_made.append({
                    'original': word,
                    'corrected': corrected,
                    'method': method,
                    'confidence': confidence
                })
        
        return {
            'original': text,
            'corrected': ' '.join(corrected_words),
            'corrections': corrections_made,
            'num_corrections': len(corrections_made)
        }
    
    # ========== TEXT COMPLETION METHODS ==========
    
    def autocomplete_ngram(self, text: str, num_words: int = 5) -> str:
        """Auto-complete using n-gram model."""
        current_text = text
        result = current_text
        
        for _ in range(num_words):
            tokens = self._preprocess_text(current_text)
            context = tuple(tokens[-(self.ngram_size - 1):])
            predictions = self.ngram_counts.get(context, {})
            
            if predictions:
                word = max(predictions.items(), key=lambda item: item[1])[0]
            else:
                word = random.choice(list(self.dictionary))
            
            result += " " + word
            current_text += " " + word
        
        return result
    
    def autocomplete_hybrid(self, text: str, num_words: int = 5, 
                           use_gpt2: bool = True) -> Dict:
        """
        Hybrid auto-completion using n-gram + DistilGPT2.
        
        Args:
            text: Input text to complete
            num_words: Number of words to add
            use_gpt2: Whether to use DistilGPT2
            
        Returns:
            Dictionary with completions and metadata
        """
        start_time = time.time()
        
        results = {
            'original': text,
            'ngram_completion': self.autocomplete_ngram(text, num_words)
        }
        
        if use_gpt2 and self.use_transformers and self.transformer_models:
            gpt2_completion = self.transformer_models.gpt2_complete(text, num_words)
            results['gpt2_completion'] = gpt2_completion
            results['recommended'] = gpt2_completion
            results['method'] = 'gpt2'
        else:
            results['recommended'] = results['ngram_completion']
            results['method'] = 'ngram'
        
        self.metrics['completions'] += 1
        self.metrics['avg_completion_time'] = (
            (self.metrics['avg_completion_time'] * (self.metrics['completions'] - 1) +
             (time.time() - start_time)) / self.metrics['completions']
        )
        
        return results
    
    # ========== TEXT GENERATION METHODS ==========
    
    def generate_text_ngram(self, prompt: str = "", max_length: int = 20) -> str:
        """Generate text using n-gram model."""
        sentence = ['<s>'] if not prompt else ['<s>'] + self._preprocess_text(prompt)
        
        while len(sentence) < max_length and sentence[-1] != '</s>':
            context = tuple(sentence[-(self.ngram_size - 1):])
            if context not in self.ngram_counts:
                break
            
            words = list(self.ngram_counts[context].keys())
            probs = list(self.ngram_counts[context].values())
            
            probs_sum = sum(probs)
            if probs_sum > 0:
                probs = [p/probs_sum for p in probs]
            
            sentence.append(np.random.choice(words, p=probs))
        
        return ' '.join(sentence[1:-1]) if sentence[-1] == '</s>' else ' '.join(sentence[1:])
    
    def generate_text_hybrid(self, prompt: str = "", max_length: int = 50, 
                            use_gpt2: bool = True, temperature: float = 0.7) -> Dict:
        """
        Hybrid text generation using n-gram + DistilGPT2.
        
        Args:
            prompt: Starting text
            max_length: Maximum length of generated text
            use_gpt2: Whether to use DistilGPT2
            temperature: Sampling temperature for GPT2
            
        Returns:
            Dictionary with generated texts and metadata
        """
        start_time = time.time()
        
        results = {
            'prompt': prompt,
            'ngram_generation': self.generate_text_ngram(prompt, max_length)
        }
        
        if use_gpt2 and self.use_transformers and self.transformer_models:
            gpt2_generations = self.transformer_models.gpt2_generate(
                prompt or "Once upon a time",
                max_length=max_length,
                temperature=temperature,
                num_return_sequences=1
            )
            results['gpt2_generation'] = gpt2_generations[0]
            results['recommended'] = gpt2_generations[0]
            results['method'] = 'gpt2'
        else:
            results['recommended'] = results['ngram_generation']
            results['method'] = 'ngram'
        
        self.metrics['generations'] += 1
        
        return results
    
    def get_metrics(self) -> Dict:
        """Get performance metrics."""
        return self.metrics.copy()
