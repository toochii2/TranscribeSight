import streamlit as st
import pandas as pd
import os
from assemblyai import Client
from openai import OpenAI
import requests
import soundfile as sf
import tempfile
import json
import time
from datetime import datetime
import difflib
from io import BytesIO
import google.generativeai as genai  
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go  
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
import torch
import re

# Cost per minute for each service (in USD)
SERVICE_COSTS = {
    'AssemblyAI': 0.00783,  
    'Whisper': 0.006,  
    'Deepgram Nova 3': 0.0043,  
    'GPT-4o': 0.006,
    'GPT-4o Mini': 0.003,  
    'Google Gemini Pro 2.5': 0.00240,  
    'Google Gemini 2.0 Flash': 0.001344,  
    'Google Gemini 2.5 Flash': 0.00192  
}

def save_dataframe_to_excel(df):
    """Save DataFrame to Excel and get bytes for download"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    excel_data = output.getvalue()
    return excel_data

@st.cache_data
def convert_df_to_excel(_df):
    """Convert DataFrame to Excel bytes (cached)"""
    return save_dataframe_to_excel(_df)

def save_api_keys():
    """Save API keys to both session state and local encrypted file"""
    api_keys = {
        'assemblyai_key': st.session_state.assemblyai_key,
        'openai_key': st.session_state.openai_key,
        'deepgram_key': st.session_state.deepgram_key,
        'google_gemini_key': st.session_state.google_gemini_key
    }
    
    # Save to session state
    st.session_state.api_keys = api_keys
    
    # Save to local file
    config_dir = os.path.join(os.path.expanduser("~"), ".stt_config")
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "api_keys.json")
    
    try:
        # Write to file
        with open(config_file, 'w') as f:
            json.dump(api_keys, f)
    except Exception as e:
        st.error(f"Error saving API keys: {str(e)}")

def load_api_keys():
    """Initialize session state variables for API keys and load from file if available"""
    # Define default empty values
    default_keys = {
        'assemblyai_key': '',
        'openai_key': '',
        'deepgram_key': '',
        'google_gemini_key': ''
    }
    
    # Try to load from file
    config_file = os.path.join(os.path.expanduser("~"), ".stt_config", "api_keys.json")
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                saved_keys = json.load(f)
                # Update default values with saved values
                default_keys.update(saved_keys)
        except Exception as e:
            st.warning(f"Error loading saved API keys: {str(e)}")
    
    # Initialize session state
    if 'api_keys' not in st.session_state:
        st.session_state.api_keys = {}
    
    # Set values in session state
    for key, value in default_keys.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
    # Initialize custom prompt with default value if not present
    if 'custom_prompt' not in st.session_state:
        st.session_state.custom_prompt = ("")
                                         
    # Initialize prompt usage setting
    if 'use_prompt' not in st.session_state:
        st.session_state.use_prompt = True

    # Initialize model selection flags (all enabled by default)
    default_models = {
        'use_assemblyai': True,
        'use_whisper': True,
        'use_gpt4o': True,
        'use_gpt4o_mini': True,
        'use_deepgram': True,
        'use_gemini_pro': True,
        'use_gemini_flash': True,
        'use_gemini_flash_25': True
    }
    
    # Set model selection values in session state
    for key, value in default_models.items():
        if key not in st.session_state:
            st.session_state[key] = value
            
    # Initialize semantic analysis options (all disabled by default)
    semantic_options = {
        'use_semascore': False,
        'use_sts': False, 
        'use_bertscore': False,
        'use_llm': False,
        'use_swwer': False
    }
    
    # Set semantic analysis options in session state
    for key, value in semantic_options.items():
        if key not in st.session_state:
            st.session_state[key] = value

def api_keys_sidebar():
    """Create sidebar for API key input"""
    with st.sidebar:
        st.header("API Configuration")
        
        # Single dropdown containing all API keys
        with st.expander("API Keys"):
            # AssemblyAI
            st.subheader("AssemblyAI")
            st.text_input("AssemblyAI API Key", 
                         key="assemblyai_key",
                         type="password")
            
            # OpenAI
            st.subheader("OpenAI")
            st.text_input("OpenAI API Key", 
                         key="openai_key",
                         type="password")
            
            # Deepgram
            st.subheader("Deepgram")
            st.text_input("Deepgram API Key", 
                         key="deepgram_key",
                         type="password")
            
            # Google Gemini
            st.subheader("Google Gemini")
            st.text_input("Google Gemini API Key",
                         key="google_gemini_key",
                         type="password")
        
        # Add custom prompt input - remove redundant value parameter
        with st.expander("Transcription Prompt"):
            # Add checkbox to enable/disable prompting
            st.checkbox("Use custom prompt", key="use_prompt", 
                       help="Uncheck to disable custom prompting for all models")
            
            st.text_area(
                "Custom Prompt (for GPT and Gemini models)",
                key="custom_prompt",
                height=150,
                placeholder=(
               "IMPORTANT: Return ONLY the raw transcript text. Do not include ANY preambles like 'Transcript:' "
               "or phrases like 'Here is the transcription'. Do not use ANY markdown formatting. "
               "Provide only the transcribed text with nothing else."),
                help="This prompt will be used by all GPT-4o and Google Gemini models"
            )
        
        # Add model selection expander
        with st.expander("Models Selection"):
            st.info("Select which models to include in the analysis")
            
            # AssemblyAI
            st.checkbox("AssemblyAI", key="use_assemblyai")
            
            # OpenAI models
            st.checkbox("OpenAI Whisper", key="use_whisper")
            st.checkbox("OpenAI GPT-4o", key="use_gpt4o")
            st.checkbox("OpenAI GPT-4o Mini", key="use_gpt4o_mini")
            
            # Deepgram
            st.checkbox("Deepgram Nova 3", key="use_deepgram")
            
            # Google Gemini models
            st.checkbox("Google Gemini Pro 2.5", key="use_gemini_pro")
            st.checkbox("Google Gemini 2.0 Flash", key="use_gemini_flash")
            st.checkbox("Google Gemini 2.5 Flash", key="use_gemini_flash_25")
            
            # Add validation message
            if not any([
                st.session_state.use_assemblyai,
                st.session_state.use_whisper,
                st.session_state.use_gpt4o,
                st.session_state.use_gpt4o_mini,
                st.session_state.use_deepgram,
                st.session_state.use_gemini_pro,
                st.session_state.use_gemini_flash,
                st.session_state.use_gemini_flash_25
            ]):
                st.error("Please select at least one model")
        
        # Add semantic analysis options
        with st.expander("Semantic Analysis Options"):
            st.info("Select which semantic analysis metrics to calculate")
            st.checkbox("SeMaScore", key="use_semascore")
            st.checkbox("STS (Semantic Textual Similarity)", key="use_sts")
            st.checkbox("BERTScore", key="use_bertscore")
            st.checkbox("LLM-based Meaning Preservation", key="use_llm")
            st.checkbox("SWWER (Semantic-Weighted Word Error Rate)", key="use_swwer")
            
        # Add analysis model selection for Overall Analysis tab
        with st.expander("Overall Analysis Options"):
            st.info("Configure options for the Overall Analysis tab")
            analysis_models = ["GPT-4o", "GPT-4o mini", "Google Gemini Pro 2.5"]
            st.selectbox("Analysis Model", analysis_models, key="analysis_model", 
                       help="Select which model to use for analyzing the transcription results")
        
        # Save button remains outside the expander
        if st.button("Save API Keys"):
            save_api_keys()
            st.success("API keys saved!")

def calculate_accuracy_metrics(predicted_text, reference_text):
    """Calculate accuracy metrics between predicted and reference text"""
    if not predicted_text or not reference_text:
        return {
            'word_error_rate': 100,
            'character_error_rate': 100,
            'token_error_rate': 100,  # Added TER with default value
            'accuracy_percentage': 0,
            'word_diff': len(reference_text.split()),
            'char_diff': len(reference_text),
            'match_error_rate': 100,
            'word_information_lost': 100,
            'word_information_preserved': 0
        }
    
    # Normalize texts
    predicted = predicted_text.lower().strip()
    reference = reference_text.lower().strip()
    
    # Word-level metrics
    pred_words = predicted.split()
    ref_words = reference.split()
    
    # Character-level metrics
    char_matcher = difflib.SequenceMatcher(None, predicted, reference)
    char_accuracy = char_matcher.ratio()
    
    # Word-level accuracy using difflib
    word_matcher = difflib.SequenceMatcher(None, pred_words, ref_words)
    word_accuracy = word_matcher.ratio()
    
    # NEW: Token-level metrics (tokenize using NLTK for simplicity)
    try:
        # Ensure NLTK punkt is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        # Tokenize texts using NLTK's word_tokenize which better handles punctuation and contractions
        pred_tokens = word_tokenize(predicted)
        ref_tokens = word_tokenize(reference)
        
        # Calculate TER (Token Error Rate) based on Levenshtein distance
        # TER = (S + D + I) / N, where:
        # S = substitutions, D = deletions, I = insertions, N = number of tokens in reference
        
        # Create a matrix to compute the minimum edit distance
        matrix = [[0 for _ in range(len(pred_tokens) + 1)] for _ in range(len(ref_tokens) + 1)]
        
        # Initialize first row and column
        for i in range(len(ref_tokens) + 1):
            matrix[i][0] = i  # Deletion cost
        for j in range(len(pred_tokens) + 1):
            matrix[0][j] = j  # Insertion cost
            
        # Fill the matrix
        for i in range(1, len(ref_tokens) + 1):
            for j in range(1, len(pred_tokens) + 1):
                if ref_tokens[i-1] == pred_tokens[j-1]:
                    matrix[i][j] = matrix[i-1][j-1]  # No operation needed
                else:
                    # Find minimum of deletion, insertion, substitution
                    matrix[i][j] = min(
                        matrix[i-1][j] + 1,     # Deletion
                        matrix[i][j-1] + 1,     # Insertion
                        matrix[i-1][j-1] + 1    # Substitution
                    )
        
        # The final cell contains the minimum edit distance
        min_edit_distance = matrix[len(ref_tokens)][len(pred_tokens)]
        
        # Calculate TER
        token_error_rate = (min_edit_distance / max(1, len(ref_tokens))) * 100  # Convert to percentage
        
        # Also calculate token-level accuracy using sequence matcher (for compatibility)
        token_matcher = difflib.SequenceMatcher(None, pred_tokens, ref_tokens)
        token_accuracy = token_matcher.ratio()
    except Exception as e:
        print(f"Error calculating token error rate: {str(e)}")
        token_error_rate = 100  # Default to 100% error on failure
        token_accuracy = 0      # Default to 0% accuracy on failure
    
    # Calculate differences
    word_diff = abs(len(pred_words) - len(ref_words))
    char_diff = abs(len(predicted) - len(reference))
    
    # Match Error Rate (MER) calculation - Standard formula
    # MER = (S + D + I) / (H + S + D + I), where:
    # S = substitutions, D = deletions, I = insertions, H = hits (correct matches)
    # Calculate edit operations
    opcodes = word_matcher.get_opcodes()
    substitutions = deletions = insertions = hits = 0
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            # Count hits (correct matches)
            hits += (i2 - i1)
        elif tag == 'replace':
            # Count substitutions (min of the two lengths)
            min_length = min(i2 - i1, j2 - j1)
            substitutions += min_length
            # Count additional insertions or deletions
            if i2 - i1 > j2 - j1:
                # More elements in first sequence, count as deletions
                deletions += (i2 - i1) - (j2 - j1)
            else:
                # More elements in second sequence, count as insertions
                insertions += (j2 - j1) - (i2 - i1)
        elif tag == 'delete':
            # Count deletions
            deletions += (i2 - i1)
        elif tag == 'insert':
            # Count insertions
            insertions += (j2 - j1)
    
    # Calculate WER using standard formula
    total_errors = substitutions + deletions + insertions
    ref_len = len(ref_words)
    wer = total_errors / max(1, ref_len)  # Avoid division by zero
    
    # Calculate MER using correct standard formula
    total_alignment_ops = hits + substitutions + deletions + insertions
    mer = total_errors / max(1, total_alignment_ops)  # Avoid division by zero
    
    # Word Information Lost (WIL)
    # WIL = 1 - (1 - WER) * (1 - |ref_len - hyp_len| / ref_len)
    # This accounts for both error rate and length differences
    length_penalty = min(1.0, abs(len(ref_words) - len(pred_words)) / max(1, ref_len))
    wil = 1.0 - (1.0 - (1.0 - word_accuracy)) * (1.0 - length_penalty)
    
    # Word Information Preserved (WIP)
    # WIP = 1 - WIL
    wip = 1.0 - wil
    
    return {
        'word_error_rate': (1 - word_accuracy) * 100,
        'character_error_rate': (1 - char_accuracy) * 100,
        'token_error_rate': token_error_rate,  # Added TER
        'accuracy_percentage': char_accuracy * 100,
        'word_diff': word_diff,
        'char_diff': char_diff,
        'match_error_rate': mer * 100,  # Convert to percentage
        'word_information_lost': wil * 100,
        'word_information_preserved': wip * 100
    }

# New functions for semantic analysis
def calculate_semantic_metrics(predicted_text, reference_text, semantic_options=None):
    """Calculate semantic similarity metrics between predicted and reference text."""
    if semantic_options is None:
        # Default to all metrics disabled
        semantic_options = {
            'use_semascore': False,
            'use_sts': False,
            'use_bertscore': False,
            'use_llm': False,
            'use_swwer': False
        }
        
    # Initialize empty result dictionary
    results = {}
    
    # Only compute metrics that are enabled
    if not predicted_text or not reference_text:
        return {
            'semascore': 0.0 if semantic_options.get('use_semascore', False) else None,
            'sts_similarity': 0.0 if semantic_options.get('use_sts', False) else None,
            'bertscore': 0.0 if semantic_options.get('use_bertscore', False) else None,
            'llm_meaning_preservation': 0.0 if semantic_options.get('use_llm', False) else None,
            'swwer': 100.0 if semantic_options.get('use_swwer', False) else None
        }
    
    # Normalize text
    pred = predicted_text.lower().strip()
    ref = reference_text.lower().strip()
    
    # 1. Calculate SeMaScore if enabled
    if semantic_options.get('use_semascore', False):
        results['semascore'] = calculate_semascore(pred, ref)
    else:
        results['semascore'] = None
    
    # 2. Calculate STS Cosine Similarity if enabled
    if semantic_options.get('use_sts', False):
        results['sts_similarity'] = calculate_sts_similarity(pred, ref)
    else:
        results['sts_similarity'] = None
    
    # 3. Calculate BERTScore if enabled
    if semantic_options.get('use_bertscore', False):
        results['bertscore'] = calculate_bertscore(pred, ref)
    else:
        results['bertscore'] = None
    
    # 4. Calculate LLM-based Meaning Preservation if enabled
    if semantic_options.get('use_llm', False):
        results['llm_meaning_preservation'] = calculate_llm_meaning_preservation(pred, ref)
    else:
        results['llm_meaning_preservation'] = None
    
    # 5. Calculate SWWER if enabled
    if semantic_options.get('use_swwer', False):
        results['swwer'] = calculate_swwer(pred, ref)
    else:
        results['swwer'] = None
    
    return results

# SeMaScore implementation - Semantic similarity score based on embeddings
def calculate_semascore(predicted, reference):
    """Calculate SeMaScore between predicted and reference text.
    SeMaScore uses sentence embeddings to calculate semantic similarity."""
    try:
        # Load the model once and cache it
        model_name = 'all-MiniLM-L6-v2'  # Smaller, faster model
        if 'sentence_transformer_model' not in st.session_state:
            with st.spinner('Loading semantic model for SeMaScore...'):
                st.session_state.sentence_transformer_model = SentenceTransformer(model_name)
        
        model = st.session_state.sentence_transformer_model
        
        # Generate embeddings
        pred_embedding = model.encode([predicted], convert_to_tensor=True)
        ref_embedding = model.encode([reference], convert_to_tensor=True)
        
        # Calculate cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(pred_embedding, ref_embedding).item()
        
        return cos_sim * 100  # Convert to percentage
    except Exception as e:
        st.warning(f"Error calculating SeMaScore: {str(e)}")
        return 0.0

# STS (Semantic Textual Similarity) using cosine similarity
def calculate_sts_similarity(predicted, reference):
    """Calculate STS Cosine Similarity between predicted and reference text."""
    try:
        # Initialize NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        # Load the model once and cache it
        model_name = 'sentence-transformers/paraphrase-MiniLM-L3-v2'  # Smaller, efficient model
        if 'sts_model' not in st.session_state:
            with st.spinner('Loading model for STS similarity...'):
                st.session_state.sts_model = SentenceTransformer(model_name)
        
        model = st.session_state.sts_model
        
        # Tokenize both texts into sentences
        pred_sentences = nltk.sent_tokenize(predicted)
        ref_sentences = nltk.sent_tokenize(reference)
        
        # Handle empty sentences
        if not pred_sentences or not ref_sentences:
            return 0.0
        
        # Encode sentences
        pred_embeddings = model.encode(pred_sentences)
        ref_embeddings = model.encode(ref_sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(pred_embeddings, ref_embeddings)
        
        # Get max similarity for each predicted sentence and average them
        max_similarities = np.max(similarity_matrix, axis=1)
        avg_similarity = np.mean(max_similarities)
        
        return avg_similarity * 100  # Convert to percentage
    except Exception as e:
        st.warning(f"Error calculating STS similarity: {str(e)}")
        return 0.0

# BERTScore implementation
def calculate_bertscore(predicted, reference):
    """Calculate BERTScore between predicted and reference text."""
    try:
        # Check if the required packages are installed
        import bert_score
        
        # Normalize text for BERTScore
        p_tokens = predicted.split()
        r_tokens = reference.split()
        
        # For empty strings, return 0
        if len(p_tokens) == 0 or len(r_tokens) == 0:
            return 0.0
        
        # Calculate BERTScore (just F1 for simplicity)
        P, R, F1 = bert_score.score([predicted], [reference], lang='en', rescale_with_baseline=True)
        
        return F1.mean().item() * 100  # Convert to percentage
    except ImportError:
        st.warning("bert-score package not installed. Please install it using 'pip install bert-score'.")
        return 0.0
    except Exception as e:
        st.warning(f"Error calculating BERTScore: {str(e)}")
        return 0.0

# LLM-based Meaning Preservation using GPT-4o mini
def calculate_llm_meaning_preservation(predicted, reference):
    """Use GPT-4o mini to evaluate meaning preservation between predicted and reference text."""
    try:
        if not st.session_state.openai_key:
            st.warning("OpenAI API key not provided. Can't calculate LLM Meaning Preservation.")
            return 0.0
        
        client = OpenAI(api_key=st.session_state.openai_key)
        
        # Create prompt for assessing meaning preservation
        prompt = f"""
        Analyze the semantic similarity between these two texts on a scale of 0 to 100, 
        where 100 means identical meaning and 0 means completely different:
        
        Reference text: "{reference}"
        
        Predicted text: "{predicted}"
        
        Consider only the meaning, not grammatical errors, punctuation, or phrasing differences.
        Output just a single number between 0-100 representing the meaning preservation score.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant that evaluates meaning preservation."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0
        )
        
        # Extract the score from the response
        score_text = response.choices[0].message.content.strip()
        
        # Use regex to extract the number
        import re
        match = re.search(r'\b(\d+(?:\.\d+)?)\b', score_text)
        
        if match:
            score = float(match.group(1))
            # Ensure score is within 0-100 range
            score = max(0, min(100, score))
            return score
        else:
            st.warning(f"Could not extract a numerical score from LLM response: {score_text}")
            return 50.0  # Default to middle value if parsing fails
            
    except Exception as e:
        st.warning(f"Error calculating LLM meaning preservation: {str(e)}")
        return 0.0

# SWWER (Semantic-Weighted Word Error Rate)
def calculate_swwer(predicted, reference):
    """Calculate Semantic-Weighted Word Error Rate - weighs errors by semantic importance."""
    try:
        # Tokenize words
        ref_words = reference.split()
        pred_words = predicted.split()
        
        if not ref_words:
            return 100.0
            
        # Initialize NLTK resources if needed
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        
        # Compute edit distance operations
        d = [[0 for _ in range(len(pred_words) + 1)] for _ in range(len(ref_words) + 1)]
        
        # Initialize first row and column
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(pred_words) + 1):
            d[0][j] = j
            
        # Compute edit distance with backtracking
        operations = []
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(pred_words) + 1):
                if ref_words[i-1] == pred_words[j-1]:
                    d[i][j] = d[i-1][j-1]  # No operation needed
                else:
                    # Consider all possibilities
                    deletion = d[i-1][j] + 1
                    insertion = d[i][j-1] + 1
                    substitution = d[i-1][j-1] + 1
                    
                    # Choose the best option
                    d[i][j] = min(deletion, insertion, substitution)
                    
        # Semantic weighting factors
        semantic_weights = {
            'content_word': 2.0,  # Higher weight for content words
            'stop_word': 0.5,     # Lower weight for stop words
            'default': 1.0        # Default weight
        }
        
        # Count errors with semantic weights
        total_errors = 0
        total_weight = 0
        
        # Calculate weighted sum of errors from the edit distance matrix
        i, j = len(ref_words), len(pred_words)
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_words[i-1] == pred_words[j-1]:
                # Match - no error
                i -= 1
                j -= 1
                # Add weight for match (will be used in denominator)
                word = ref_words[i]
                weight = semantic_weights['stop_word'] if word.lower() in stop_words else semantic_weights['content_word']
                total_weight += weight
            elif j > 0 and (i == 0 or d[i][j-1] + 1 == d[i][j]):
                # Insertion
                j -= 1
                word = pred_words[j]
                weight = semantic_weights['stop_word'] if word.lower() in stop_words else semantic_weights['content_word']
                total_errors += weight
                total_weight += weight
            elif i > 0 and (j == 0 or d[i-1][j] + 1 == d[i][j]):
                # Deletion
                i -= 1
                word = ref_words[i]
                weight = semantic_weights['stop_word'] if word.lower() in stop_words else semantic_weights['content_word']
                total_errors += weight
                total_weight += weight
            else:
                # Substitution
                i -= 1
                j -= 1
                ref_word = ref_words[i]
                pred_word = pred_words[j]
                weight = semantic_weights['stop_word'] if ref_word.lower() in stop_words else semantic_weights['content_word']
                total_errors += weight
                total_weight += weight
                
        # Calculate weighted WER
        if total_weight > 0:
            swwer = (total_errors / total_weight) * 100
        else:
            swwer = 0.0
            
        return swwer
    except Exception as e:
        st.warning(f"Error calculating SWWER: {str(e)}")
        return 100.0  # Return worst score on error

def get_audio_duration(audio_path):
    """Get duration of audio file in minutes"""
    with sf.SoundFile(audio_path) as audio_file:
        duration = audio_file.frames / audio_file.samplerate
    return duration / 60  # Convert to minutes

def calculate_cost(duration, service):
    """Calculate cost for transcription"""
    return duration * SERVICE_COSTS[service]

def transcribe_with_metrics(func, audio_path, client, service_name):
    """Wrapper to measure transcription performance - only records actual transcription time"""
    start_time = time.time()
    try:
        # Our updated transcription functions now return both the text and the measured transcription time
        result = func(audio_path, client)
        
        # Check if the function returns the transcription time (new format)
        if isinstance(result, tuple) and len(result) == 2:
            text, transcription_time = result
            success = True
        else:
            # Fallback for any functions that haven't been updated yet
            text = result
            transcription_time = time.time() - start_time
            success = True
    except Exception as e:
        text = f"Error: {str(e)}"
        transcription_time = time.time() - start_time
        success = False
    
    end_time = time.time()
    duration = get_audio_duration(audio_path)
    
    metrics = {
        'duration_minutes': duration,
        'processing_time_seconds': transcription_time,  # Use the measured transcription time only
        'cost': calculate_cost(duration, service_name) if success else 0,
        'success': success,
        'words': len(text.split()) if success else 0,
        'characters': len(text) if success else 0
    }
    
    return text, metrics

# Initialize API clients and configurations
def init_assemblyai_client():
    if not st.session_state.assemblyai_key:
        raise ValueError("AssemblyAI API key not provided")
    
    # Set the API key in the package configuration
    import assemblyai as aai
    aai.settings.api_key = st.session_state.assemblyai_key
    return aai

def init_openai_client():
    if not st.session_state.openai_key:
        raise ValueError("OpenAI API key not provided")
    
    return OpenAI(api_key=st.session_state.openai_key)

def init_deepgram_client():
    """Return Deepgram API key for REST transcription."""
    if not st.session_state.deepgram_key:
        raise ValueError("Deepgram API key not provided")
    return st.session_state.deepgram_key

# Add init for Google Gemini
def init_gemini_client():
    if not st.session_state.google_gemini_key:
        raise ValueError("Google Gemini API key not provided")
    genai.configure(api_key=st.session_state.google_gemini_key)
    return genai

# Transcription functions
def transcribe_assemblyai(audio_path, aai):
    """
    Transcribe audio using AssemblyAI with more accurate processing time measurement.
    Returns the transcript text and tracks actual transcription time separately from upload/download time.
    """
    try:
        # Create the transcriber object
        transcriber = aai.Transcriber()
        
        # First, just upload the audio file without transcribing
        # We'll use the file path directly as AssemblyAI handles the file loading internally
        
        # Start timing only for the actual transcription process
        transcription_start = time.time()
        
        # Perform the transcription
        transcript = transcriber.transcribe(audio_path)
            
        # At this point, the transcription is complete
        transcription_complete = time.time()
        
        # Calculate the actual transcription time
        transcription_time = transcription_complete - transcription_start
        
        # Return the transcript text
        result_text = transcript.text if transcript.text else "No transcription available"
        
        # Log the actual processing times for debugging
        print(f"AssemblyAI actual transcription time: {transcription_time:.2f}s")
        
        return result_text, transcription_time
    except Exception as e:
        print(f"Error in AssemblyAI transcription: {str(e)}")
        raise e

def transcribe_whisper(audio_path, client):
    """
    Transcribe audio using OpenAI Whisper with accurate processing time measurement.
    Returns the transcript text and tracks only the actual transcription time.
    """
    try:
        # Load the audio file first
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
        
        # Start timing only for the actual transcription process
        transcription_start = time.time()
        
        # Perform the transcription
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file,
                response_format="text"
            )
            
        # At this point, the transcription is complete
        transcription_complete = time.time()
        
        # Calculate the actual transcription time
        transcription_time = transcription_complete - transcription_start
        
        # Return the transcript text
        result_text = transcript if transcript else "No transcription available"
        
        # Log the actual processing times for debugging
        print(f"Whisper actual transcription time: {transcription_time:.2f}s")
        
        return result_text, transcription_time
    except Exception as e:
        print(f"Error in Whisper transcription: {str(e)}")
        raise e

def transcribe_deepgram(audio_path, api_key):
    """
    Transcribe audio using Deepgram REST API (nova-3 model) with accurate processing time measurement.
    Returns the transcript text and tracks only the actual transcription time.
    """
    try:
        # Load the audio file first
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        # Prepare the API request
        url = "https://api.deepgram.com/v1/listen?model=nova-3&smart_format=true"
        headers = {"Authorization": f"Token {api_key}"}
        
        # Start timing only for the actual transcription process
        transcription_start = time.time()
        
        # Send the API request
        response = requests.post(url, headers=headers, data=audio_data)
        
        # At this point, the transcription is complete
        transcription_complete = time.time()
        
        # Calculate the actual transcription time
        transcription_time = transcription_complete - transcription_start
        
        # Process the response
        if response.status_code == 200:
            payload = response.json()
            try:
                transcript = payload['results']['channels'][0]['alternatives'][0]['transcript']
                result_text = transcript
            except (KeyError, IndexError):
                result_text = "No transcription available"
        else:
            result_text = f"Error: HTTP {response.status_code}"
        
        # Log the actual processing times for debugging
        print(f"Deepgram Nova 3 actual transcription time: {transcription_time:.2f}s")
        
        return result_text, transcription_time
    except Exception as e:
        print(f"Error in Deepgram transcription: {str(e)}")
        raise e

def transcribe_gpt4o(audio_path, client):
    """
    Transcribe audio using GPT-4o with accurate processing time measurement.
    Returns the transcript text and tracks only the actual transcription time.
    """
    try:
        # Load the audio file first
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
        
        # Prepare the transcription parameters
        kwargs = {
            "model": "gpt-4o-transcribe",
            "response_format": "text"
        }
        
        # Add prompt only if enabled
        if st.session_state.use_prompt:
            kwargs["prompt"] = st.session_state.custom_prompt
        
        # Start timing only for the actual transcription process
        transcription_start = time.time()
        
        # Perform the transcription
        with open(audio_path, "rb") as audio_file:
            kwargs["file"] = audio_file
            transcription = client.audio.transcriptions.create(**kwargs)
            
        # At this point, the transcription is complete
        transcription_complete = time.time()
        
        # Calculate the actual transcription time
        transcription_time = transcription_complete - transcription_start
        
        # Extract text result
        result_text = getattr(transcription, 'text', transcription)
        
        # Log the actual processing times for debugging
        print(f"GPT-4o actual transcription time: {transcription_time:.2f}s")
        
        return result_text, transcription_time
    except Exception as e:
        print(f"Error in GPT-4o transcription: {str(e)}")
        raise e

def transcribe_gpt4o_mini(audio_path, client):
    """
    Transcribe audio using GPT-4o Mini with accurate processing time measurement.
    Returns the transcript text and tracks only the actual transcription time.
    """
    try:
        # Load the audio file first
        with open(audio_path, "rb") as audio_file:
            audio_data = audio_file.read()
        
        # Prepare the transcription parameters
        kwargs = {
            "model": "gpt-4o-mini-transcribe",
            "response_format": "text"
        }
        
        # Add prompt only if enabled
        if st.session_state.use_prompt:
            kwargs["prompt"] = st.session_state.custom_prompt
        
        # Start timing only for the actual transcription process
        transcription_start = time.time()
        
        # Perform the transcription
        with open(audio_path, "rb") as audio_file:
            kwargs["file"] = audio_file
            transcription = client.audio.transcriptions.create(**kwargs)
            
        # At this point, the transcription is complete
        transcription_complete = time.time()
        
        # Calculate the actual transcription time
        transcription_time = transcription_complete - transcription_start
        
        # Extract text result
        result_text = getattr(transcription, 'text', transcription)
        
        # Log the actual processing times for debugging
        print(f"GPT-4o Mini actual transcription time: {transcription_time:.2f}s")
        
        return result_text, transcription_time
    except Exception as e:
        print(f"Error in GPT-4o Mini transcription: {str(e)}")
        raise e

# Google Gemini transcription
def transcribe_gemini_pro_2_5(audio_path, client):
    """
    Transcribe audio using Google Gemini Pro 2.5 with accurate processing time measurement.
    Returns the transcript text and tracks only the actual transcription time.
    """
    try:
        # Upload the audio file first
        audio_file = client.upload_file(path=audio_path)
        
        # Choose the model for transcription
        model = client.GenerativeModel(model_name="gemini-2.5-pro-preview-03-25")
        
        # Set prompt based on whether prompting is enabled
        if st.session_state.use_prompt:
            prompt = f'Generate a transcript of the speech. {st.session_state.custom_prompt} Provide only the raw transcript text with no extra formatting, headings or explanations.'
        else:
            prompt = 'Generate a transcript of the speech, Provide only the raw transcript text with no extra formatting, headings or explanations.'
        
        # Start timing only for the actual transcription process
        transcription_start = time.time()
        
        # Generate transcript
        response = model.generate_content([prompt, audio_file])
        
        # At this point, the transcription is complete
        transcription_complete = time.time()
        
        # Calculate the actual transcription time
        transcription_time = transcription_complete - transcription_start
        
        # Extract result text
        result_text = getattr(response, 'text', None) or 'No transcription available'
        
        # Log the actual processing time for debugging
        print(f"Google Gemini Pro 2.5 actual transcription time: {transcription_time:.2f}s")
        
        return result_text, transcription_time
    except Exception as e:
        print(f"Error in Google Gemini Pro 2.5 transcription: {str(e)}")
        raise e

# Google Gemini 2.0 Flash transcription
def transcribe_gemini_2flash(audio_path, client):
    """
    Transcribe audio using Google Gemini 2.0 Flash with accurate processing time measurement.
    Returns the transcript text and tracks only the actual transcription time.
    """
    try:
        # Upload the audio file first
        audio_file = client.upload_file(path=audio_path)
        
        # Choose the flash model for transcription
        model = client.GenerativeModel(model_name="gemini-2.0-flash")
        
        # Set prompt based on whether prompting is enabled
        if st.session_state.use_prompt:
            prompt = f'Generate a transcript of the speech. {st.session_state.custom_prompt} Provide only the raw transcript text with no extra formatting, headings or explanations.'
        else:
            prompt = 'Generate a transcript of the speech, Provide only the raw transcript text with no extra formatting, headings or explanations.'
        
        # Start timing only for the actual transcription process
        transcription_start = time.time()
        
        # Generate transcript
        response = model.generate_content([prompt, audio_file])
        
        # At this point, the transcription is complete
        transcription_complete = time.time()
        
        # Calculate the actual transcription time
        transcription_time = transcription_complete - transcription_start
        
        # Extract result text
        result_text = getattr(response, 'text', None) or 'No transcription available'
        
        # Log the actual processing time for debugging
        print(f"Google Gemini 2.0 Flash actual transcription time: {transcription_time:.2f}s")
        
        return result_text, transcription_time
    except Exception as e:
        print(f"Error in Google Gemini 2.0 Flash transcription: {str(e)}")
        raise e

# Google Gemini 2.5 Flash transcription
def transcribe_gemini_2_5_flash(audio_path, client):
    """
    Transcribe audio using Google Gemini 2.5 Flash with accurate processing time measurement.
    Returns the transcript text and tracks only the actual transcription time.
    """
    try:
        # Upload the audio file first
        audio_file = client.upload_file(path=audio_path)
        
        # Choose the flash model for transcription
        model = client.GenerativeModel(model_name="gemini-2.5-flash")
        
        # Set prompt based on whether prompting is enabled
        if st.session_state.use_prompt:
            prompt = f'Generate a transcript of the speech. {st.session_state.custom_prompt} Provide only the raw transcript text with no extra formatting, headings or explanations.'
        else:
            prompt = 'Generate a transcript of the speech, Provide only the raw transcript text with no extra formatting, headings or explanations.'
        
        # Start timing only for the actual transcription process
        transcription_start = time.time()
        
        # Generate transcript
        response = model.generate_content([prompt, audio_file])
        
        # At this point, the transcription is complete
        transcription_complete = time.time()
        
        # Calculate the actual transcription time
        transcription_time = transcription_complete - transcription_start
        
        # Extract result text
        result_text = getattr(response, 'text', None) or 'No transcription available'
        
        # Log the actual processing time for debugging
        print(f"Google Gemini 2.5 Flash actual transcription time: {transcription_time:.2f}s")
        
        return result_text, transcription_time
    except Exception as e:
        print(f"Error in Google Gemini 2.5 Flash transcription: {str(e)}")
        raise e

# New functions for radar charts and analysis
def prepare_metrics_for_radar(df_metrics):
    """Prepare metrics data for radar charts by normalizing metrics and inverting where needed."""
    # Get list of services used
    service_names = set()
    for col in df_metrics.columns:
        for service in ['AssemblyAI', 'Whisper', 'GPT-4o', 'GPT-4o Mini', 'Deepgram Nova 3', 
                      'Google Gemini Pro 2.5', 'Google Gemini 2.0 Flash', 'Google Gemini 2.5 Flash']:
            if service in col:
                service_names.add(service)
    
    service_names = sorted(service_names)
    
    # Define metric categories to include in radar chart (prefer metrics where higher is better)
    radar_categories = [
        #{'name': 'Accuracy', 'pattern': 'Accuracy (%)', 'invert': False},
        {'name': 'WER', 'pattern': 'Word Error Rate (%)', 'invert': True},  # Lower is better, so invert
        {'name': 'CER', 'pattern': 'Char Error Rate (%)', 'invert': True},  # Lower is better, so invert
        {'name': 'TER', 'pattern': 'Token Error Rate (%)', 'invert': True},  # Lower is better, so invert
        {'name': 'WIP', 'pattern': 'Word Information Preserved (%)', 'invert': False},
        {'name': 'Processing Speed', 'pattern': 'Processing Time (s)', 'invert': True},  # Lower is better, so invert
        {'name': 'Cost Efficiency', 'pattern': 'Cost ($)', 'invert': True}  # Lower is better, so invert
    ]
    
    # Add semantic metrics ONLY if they're enabled in session state AND present in the data
    if st.session_state.get('use_semascore', False) and any(col for col in df_metrics.columns if 'semascore' in col.lower()):
        radar_categories.append({'name': 'SeMaScore', 'pattern': 'SeMaScore', 'invert': False})
        
    if st.session_state.get('use_sts', False) and any(col for col in df_metrics.columns if 'sts similarity' in col.lower() or 'sts_similarity' in col.lower()):
        radar_categories.append({'name': 'STS Similarity', 'pattern': 'STS Similarity', 'invert': False})
        
    if st.session_state.get('use_bertscore', False) and any(col for col in df_metrics.columns if 'bertscore' in col.lower()):
        radar_categories.append({'name': 'BERTScore', 'pattern': 'BERTScore', 'invert': False})
        
    if st.session_state.get('use_llm', False) and any(col for col in df_metrics.columns if 'llm meaning' in col.lower() or 'meaning preservation' in col.lower()):
        radar_categories.append({'name': 'LLM Meaning', 'pattern': 'LLM Meaning Preservation', 'invert': False})
        
    if st.session_state.get('use_swwer', False) and any(col for col in df_metrics.columns if 'swwer' in col.lower()):
        radar_categories.append({'name': 'SWWER', 'pattern': 'SWWER', 'invert': True})  # Lower is better, so invert
        
    # Create normalized data structure
    radar_data = {
        'categories': [cat['name'] for cat in radar_categories],
        'services': {}
    }
    
    # For each metric category, find the corresponding columns and calculate service averages
    for service in service_names:
        service_metrics = []
        
        for category in radar_categories:
            metric_pattern = category['pattern']
            matching_cols = [col for col in df_metrics.columns if service in col and metric_pattern in col]
            
            if matching_cols:
                # Calculate average value across files
                avg_value = df_metrics[matching_cols].mean().mean()
                
                # For metrics where lower is better, invert the score to make higher = better
                if category['invert']:
                    # Find the max value across all services for this metric to normalize
                    all_matching_cols = [col for col in df_metrics.columns if metric_pattern in col]
                    max_value = df_metrics[all_matching_cols].mean().max() * 1.1  # Add 10% buffer
                    
                    # Invert the score: max_value - actual_value
                    # This way, lower original values become higher inverted values
                    if max_value > 0 and not pd.isna(avg_value):
                        avg_value = max_value - avg_value
                    else:
                        avg_value = 0
                
                service_metrics.append(avg_value)
            else:
                service_metrics.append(0)  # Default value for missing metrics
        
        radar_data['services'][service] = service_metrics
    
    # Normalize metrics to 0-100 scale for each category
    for i, _ in enumerate(radar_data['categories']):
        # Find the max value across all services for this category
        max_val = max([service_metrics[i] for service_metrics in radar_data['services'].values()])
        
        # Normalize each service's value
        if max_val > 0:
            for service in service_names:
                radar_data['services'][service][i] = (radar_data['services'][service][i] / max_val) * 100
    
    return radar_data

def create_radar_chart_matplotlib(radar_metrics, services_to_include, title):
    """Create a radar chart visualization for the specified services using Matplotlib."""
    categories = radar_metrics['categories']
    num_vars = len(categories)
    
    # Calculate the angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # Close the polygon by repeating the first angle
    angles += angles[:1]
    
    # Create figure and polar axes
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Y-axis ticks (0-100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'])
    ax.set_ylim(0, 100)
    
    # Add title
    ax.set_title(title, size=15, y=1.1)
    
    # Use consistent colors from the session_state service_colors dictionary
    for i, service in enumerate(services_to_include):
        if service in radar_metrics['services']:
            # Get color from the service_colors dictionary if available
            if hasattr(st.session_state, 'service_colors') and service in st.session_state.service_colors:
                color = st.session_state.service_colors[service]
            else:
                # Fallback to rainbow colormap
                colors = plt.cm.rainbow(np.linspace(0, 1, len(services_to_include)))
                color = colors[i]
            
            # Get the values for this service
            values = radar_metrics['services'][service].copy()
            
            # Ensure we have the right number of values
            if len(values) != num_vars:
                st.warning(f"Dimension mismatch for {service}: expected {num_vars} metrics but got {len(values)}")
                # Adjust values to match the expected dimensions
                if len(values) < num_vars:
                    # Pad with zeros if we have too few
                    values = values + [0] * (num_vars - len(values))
                else:
                    # Trim if we have too many
                    values = values[:num_vars]
            
            # Close the polygon by repeating the first value
            values += values[:1]
            
            # Make sure angles and values have the same length
            if len(angles) != len(values):
                st.warning(f"Length mismatch for {service}: angles={len(angles)}, values={len(values)}")
                # Fix the length - this should usually not happen if the above code works correctly
                min_len = min(len(angles), len(values))
                angles_plot = angles[:min_len]
                values_plot = values[:min_len]
            else:
                angles_plot = angles
                values_plot = values
            
            # Plot data and fill area
            ax.plot(angles_plot, values_plot, linewidth=2, linestyle='solid', label=service, color=color)
            ax.fill(angles_plot, values_plot, alpha=0.1, color=color)
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    return fig

def create_radar_chart_plotly(radar_metrics, services_to_include, title):
    """Create a radar chart visualization for the specified services using Plotly."""
    categories = radar_metrics['categories']
    
    # Create figure
    fig = go.Figure()
    
    # Define a more vibrant color palette
    vibrant_colors = [
        '#FF1493',  # Deep Pink
        '#00FFFF',  # Cyan
        '#FF4500',  # Orange Red
        '#32CD32',  # Lime Green
        '#8A2BE2',  # Blue Violet
        '#FFD700',  # Gold
        '#00CED1',  # Dark Turquoise
        '#FF6347',  # Tomato
        '#7B68EE',  # Medium Slate Blue
        '#3CB371',  # Medium Sea Green
        '#FF69B4',  # Hot Pink
        '#20B2AA',  # Light Sea Green
        '#FF8C00',  # Dark Orange
        '#4169E1',  # Royal Blue
        '#00FA9A',  # Medium Spring Green
    ]
    
    # Add each service as a trace
    for i, service in enumerate(services_to_include):
        if service in radar_metrics['services']:
            # Get color from the service_colors dictionary if available, or use vibrant colors
            if hasattr(st.session_state, 'service_colors') and service in st.session_state.service_colors:
                color = st.session_state.service_colors[service]
            else:
                # Use vibrant color palette
                color = vibrant_colors[i % len(vibrant_colors)]
                
            # Get values for this service
            values = radar_metrics['services'][service].copy()
            
            # Ensure we have the right number of values
            if len(values) != len(categories):
                if len(values) < len(categories):
                    # Pad with zeros if too few
                    values = values + [0] * (len(categories) - len(values))
                else:
                    # Trim if too many
                    values = values[:len(categories)]
            
            # Close the loop by adding the first value at the end
            values_plot = values + [values[0]]
            categories_plot = categories + [categories[0]]
            
            # Create line style with distinct borders - thicker for combined charts
            line_width = 3 if len(services_to_include) > 1 else 2
            line_color = 'black' if len(services_to_include) > 1 else color
            
            # Add trace for this service
            fig.add_trace(go.Scatterpolar(
                r=values_plot,
                theta=categories_plot,
                fill='toself',
                name=service,
                line=dict(
                    color=line_color,  # Black border for combined charts, matching color for single service
                    width=line_width,  # Thicker lines to create visible borders
                ),
                fillcolor=color,
                opacity=0.4 if len(services_to_include) > 1 else 0.6,  # More opaque for better visibility
            ))
    
    # Determine if this is an individual chart or a combined chart
    is_individual = len(services_to_include) == 1
    
    # Update layout for a cleaner look with larger size for individual charts
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showline=True,
                linewidth=1,
                gridcolor="rgba(200, 200, 200, 0.5)"  # Lighter grid lines
            ),
            angularaxis=dict(
                showline=True,
                linewidth=1,
                gridcolor="rgba(200, 200, 200, 0.5)"  # Lighter grid lines
            ),
            bgcolor="rgba(0, 0, 0, 0)"  # Transparent background
        ),
        title=dict(
            text=title,
            x=0.5,
            y=0.98,
            xanchor='center',
            yanchor='top',
            font=dict(size=16)
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=1.1,  # Position legend to the right of the chart
            bgcolor="rgba(255, 255, 255, 0.7)",  # More opaque background for better readability
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1
        ),
        # Adjust height for individual charts to make them larger
        height=650 if is_individual else 600,
        width=650 if is_individual else 700,
        margin=dict(l=50, r=150, t=80, b=50),  # Increased right margin for legend
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background for the entire figure
        font=dict(family="Arial, sans-serif")
    )
    
    return fig

def create_radar_chart(radar_metrics, services_to_include, title):
    """Create a radar chart visualization for the specified services using the selected library."""
    if st.session_state.get('chart_library', 'plotly') == 'plotly':
        return create_radar_chart_plotly(radar_metrics, services_to_include, title)
    else:
        return create_radar_chart_matplotlib(radar_metrics, services_to_include, title)

def get_combined_radar_png(radar_metrics, service_names):
    """Create combined radar chart and convert to PNG bytes."""
    try:
        # Use matplotlib for PNG generation regardless of chart library preference
        # This avoids the problematic Plotly to PNG conversion that causes infinite loops
        import matplotlib.pyplot as plt
        from io import BytesIO
        
        fig = create_radar_chart_matplotlib(radar_metrics, service_names, "All Services Comparison")
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)  # Close the figure to free memory
        return buf.getvalue()
    except Exception as e:
        # Generic error handling as last resort
        print(f"Error generating radar chart PNG: {str(e)}")
        # Create a very simple fallback image with text
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Error generating radar chart", ha='center', va='center', fontsize=20)
        ax.axis('off')
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        plt.close(fig)  # Close the figure to free memory
        return buf.getvalue()

def generate_llm_analysis(df_metrics, service_names, model_name):
    """Generate a comprehensive analysis of the transcription services using the specified LLM model."""
    if not service_names:
        return "No services to analyze. Please process some audio files first."
        
    # Create a summary of the metrics data to send to the LLM
    # We'll create a markdown table with the averages for each service
    metrics_summary = "# Transcription Services Performance Summary\n\n"
    
    # Get traditional accuracy metrics
    metrics_to_include = [
        ('Word Accuracy (%)', 'higher is better'),
        ('Word Error Rate (%)', 'lower is better'),
        ('Char Error Rate (%)', 'lower is better'),
        ('Token Error Rate (%)', 'lower is better'),
        ('Match Error Rate (%)', 'lower is better'),
        ('Word Information Preserved (%)', 'higher is better'),
        ('Word Information Lost (%)', 'lower is better'),
        ('Processing Time (s)', 'lower is better'),
        ('Cost ($)', 'lower is better')
    ]
    
    # Add semantic metrics if available
    if any(col for col in df_metrics.columns if 'semascore' in col.lower()):
        metrics_to_include.append(('SeMaScore', 'higher is better'))
    if any(col for col in df_metrics.columns if 'sts similarity' in col.lower() or 'sts_similarity' in col.lower()):
        metrics_to_include.append(('STS Similarity', 'higher is better'))
    if any(col for col in df_metrics.columns if 'bertscore' in col.lower()):
        metrics_to_include.append(('BERTScore', 'higher is better'))
    if any(col for col in df_metrics.columns if 'llm meaning' in col.lower()):
        metrics_to_include.append(('LLM Meaning Preservation', 'higher is better'))
    if any(col for col in df_metrics.columns if 'swwer' in col.lower()):
        metrics_to_include.append(('SWWER', 'lower is better'))
    
    # Create a table header
    metrics_summary += "| Metric | " + " | ".join(service_names) + " | Direction |\n"
    metrics_summary += "|--------|" + "----|".join(["" for _ in range(len(service_names) + 1)]) + "\n"
    
    # Add data rows
    for metric_name, direction in metrics_to_include:
        row = f"| {metric_name} | "
        for service in service_names:
            # Find columns for this service and metric
            cols = [col for col in df_metrics.columns if service in col and metric_name in col]
            if cols:
                avg_value = df_metrics[cols].mean().mean()
                # Use more decimal places for cost values
                if 'Cost' in metric_name:
                    row += f"{avg_value:.6f} | "  # Show 6 decimal places for costs
                else:
                    row += f"{avg_value:.2f} | "  # Keep 2 decimal places for other metrics
            else:
                row += "N/A | "
        
        row += f"{direction} |"
        metrics_summary += row + "\n"
    
    # Add performance efficiency metrics if available
    metrics_summary += "\n## Performance Efficiency\n\n"
    
    # Create efficiency metrics (similar to what's shown in Performance Metrics tab)
    efficiency_data = []
    for service in service_names:
        service_cost_cols = [col for col in df_metrics.columns if service in col and 'Cost' in col]
        service_time_cols = [col for col in df_metrics.columns if service in col and 'Processing Time' in col]
        
        if service_cost_cols and service_time_cols:
            service_total_cost = df_metrics[service_cost_cols].mean().sum()
            service_total_time = df_metrics[service_time_cols].mean().sum()
            cost_per_second = service_total_cost / max(service_total_time, 0.001)  # Avoid division by zero
            efficiency_data.append({
                'Service': service,
                'Avg Cost ($)': service_total_cost,
                'Avg Time (s)': service_total_time,
                'Cost per Second': cost_per_second,
                'Cost-Time Index': (service_total_cost * service_total_time) ** 0.5
            })
    
    if efficiency_data:
        efficiency_summary = "| Service | Avg Cost ($) | Avg Time (s) | Cost per Second | Cost-Time Index |\n"
        efficiency_summary += "|---------|----------|----------|----------------|----------------|\n"
        
        for data in efficiency_data:
            efficiency_summary += f"| {data['Service']} | {data['Avg Cost ($)']:.6f} | {data['Avg Time (s)']:.2f} | "
            efficiency_summary += f"{data['Cost per Second']:.6f} | {data['Cost-Time Index']:.4f} |\n"
        
        metrics_summary += efficiency_summary
        
    # Add transcription samples from results table if available
    transcription_examples = ""
    if hasattr(st.session_state, 'results') and st.session_state.results:
        # Get transcription results
        results_df = pd.DataFrame(st.session_state.results)
        
        # Only include files with ground truth
        examples_with_ground_truth = [r for r in st.session_state.results if 'Ground Truth' in r and r['Ground Truth'] != "No reference available"]
        
        # Pick up to 3 representative examples to include in the analysis
        num_examples = min(3, len(examples_with_ground_truth))
        
        if num_examples > 0:
            transcription_examples = "\n\n## Transcription Text Samples\n\n"
            
            for i in range(num_examples):
                example = examples_with_ground_truth[i]
                file_name = example.get('File Name', f"Example {i+1}")
                ground_truth = example.get('Ground Truth', "")
                
                # Limit text length for API constraints
                max_text_length = 300
                if len(ground_truth) > max_text_length:
                    ground_truth = ground_truth[:max_text_length] + "..."
                
                transcription_examples += f"### Sample {i+1}: {file_name}\n\n"
                transcription_examples += f"**Ground Truth:**\n\n{ground_truth}\n\n"
                transcription_examples += "**Service Transcriptions:**\n\n"
                
                # Add each service's transcription
                for service in service_names:
                    # For each service, look for the appropriate transcript column
                    # It could be either "{service}" or "{service} Transcript"
                    service_text = ""
                    if service in example:
                        service_text = example[service]
                    elif f"{service} Transcript" in example:
                        service_text = example[f"{service} Transcript"]
                    
                    # Only add the service if we have a transcript
                    if service_text:
                        # Limit the text length
                        if len(service_text) > max_text_length:
                            service_text = service_text[:max_text_length] + "..."
                        transcription_examples += f"**{service}:**\n\n{service_text}\n\n"
    
    # Add analysis prompt
    analysis_prompt = f"""
{metrics_summary}
{transcription_examples}

Please provide a comprehensive analysis of these speech-to-text transcription services based on the metrics and transcription samples provided. Your analysis should include:

1. Overall ranking of the services from best to worst based on a balanced consideration of accuracy metrics (Word Error Rate, Character Error Rate, Token Error Rate) and performance metrics (cost, processing time)

2. Strengths and weaknesses of each service presented in a table format like this:
| Service | Strengths | Weaknesses |
|---------|-----------|------------|
| Service 1 | - Strength 1\\n- Strength 2 | - Weakness 1\\n- Weakness 2 |
| Service 2 | - Strength 1\\n- Strength 2 | - Weakness 1\\n- Weakness 2 |

3. Specific recommendations for different use cases:
   - Best service for accuracy-critical applications
   - Best service for cost-sensitive applications
   - Best service for real-time/low-latency needs
   - Best all-around service (balance of accuracy, cost, and speed)

4. Any notable patterns or insights from the error metrics (Word Error Rate, Character Error Rate, Token Error Rate)

5. Interpretation of the semantic metrics (if available) and how they compare to traditional error metrics

6. Analysis of the transcription samples:
   - Notable differences between services in handling specific words or phrases
   - Which services maintain meaning best even if they don't get every word correct
   - Any patterns of errors that are specific to certain services

Format your analysis in markdown with clear sections, bullet points where appropriate, and bold text for important conclusions.
"""
    
    # Select the appropriate model based on user selection
    if model_name == "GPT-4o":
        return analyze_with_openai(analysis_prompt, "gpt-4o")
    elif model_name == "GPT-4o mini":
        return analyze_with_openai(analysis_prompt, "gpt-4o-mini")
    elif model_name == "Google Gemini Pro 2.5":
        return analyze_with_gemini(analysis_prompt, "gemini-2.5-pro-preview-03-25")
    else:
        # Default to GPT-4o mini as fallback
        return analyze_with_openai(analysis_prompt, "gpt-4o-mini")

def analyze_with_openai(analysis_prompt, model_name):
    """Generate analysis using OpenAI models."""
    if not st.session_state.openai_key:
        return "**Error:** OpenAI API key not provided. Please enter your API key in the sidebar."
    
    try:
        client = OpenAI(api_key=st.session_state.openai_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert in speech-to-text technology analysis. Provide clear, data-driven insights about different transcription services."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.1,  # Low temperature for more consistent results
            max_tokens=14000   # Increased token limit to accommodate the transcription analysis
        )
        
        # Clean up the table formatting to replace \n with HTML breaks instead
        result_text = response.choices[0].message.content
        # Replace \n inside table cells while preserving proper markdown table structure
        if "| Service | Strengths | Weaknesses |" in result_text:
            # Process the table to replace newlines
            # Find the table start
            table_start = result_text.find("| Service | Strengths | Weaknesses |")
            if table_start != -1:
                # Find the end of the table (next header or empty line)
                lines = result_text[table_start:].split("\n")
                table_end = 0
                for i, line in enumerate(lines):
                    if i > 2 and (not line.strip() or not line.startswith("|")):
                        table_end = i
                        break
                
                # Extract the table content
                table_content = lines[:table_end] if table_end > 0 else lines
                
                # Process the table rows to replace \n with <br> in cells
                for i in range(len(table_content)):
                    if table_content[i].startswith("|") and i >= 2:  # Skip header and separator rows
                        # Split the row into cells while preserving the | characters
                        cells = table_content[i].split("|")
                        for j in range(len(cells)):
                            # Replace \n in each cell with HTML break
                            cells[j] = cells[j].replace("\\n", " • ")
                        # Reconstruct the row
                        table_content[i] = "|".join(cells)
                
                # Reconstruct the text with the modified table
                modified_table = "\n".join(table_content)
                result_text = result_text[:table_start] + modified_table + result_text[table_start + len(modified_table):]
        
        return result_text
    except Exception as e:
        return f"**Error generating analysis:** {str(e)}"

def analyze_with_gemini(analysis_prompt, model_name):
    """Generate analysis using Google Gemini models."""
    if not st.session_state.google_gemini_key:
        return "**Error:** Google Gemini API key not provided. Please enter your API key in the sidebar."
    
    try:
        genai.configure(api_key=st.session_state.google_gemini_key)
        model = genai.GenerativeModel(model_name=model_name)
        
        # Gemini might have length limitations, so truncate the prompt if necessary
        max_prompt_length = 60000  # Approximate limit, adjust as needed
        if len(analysis_prompt) > max_prompt_length:
            # Truncate the transcription examples section to fit limits
            parts = analysis_prompt.split("## Transcription Text Samples")
            if len(parts) > 1:
                # Keep the metrics and append a shortened version of the samples
                main_part = parts[0]
                samples_part = "## Transcription Text Samples" + parts[1]
                remaining_length = max_prompt_length - len(main_part) - 100  # Leave some buffer
                
                if remaining_length > 500:  # Only include samples if we have reasonable space
                    truncated_samples = samples_part[:remaining_length] + "\n\n[Note: Some transcription samples were truncated due to length constraints]"
                    analysis_prompt = main_part + truncated_samples
                else:
                    # Not enough space for samples, use just the metrics
                    analysis_prompt = main_part + "\n\n[Note: Transcription samples were omitted due to length constraints]"
        
        response = model.generate_content(
            contents=[
                "You are an expert in speech-to-text technology analysis. Provide clear, data-driven insights about different transcription services.", 
                analysis_prompt
            ],
            generation_config=genai.types.GenerationConfig(temperature=0.1)
        )
        
        # Apply the same cleaning to Gemini responses as well
        result_text = response.text
        # Replace \n inside table cells while preserving proper markdown table structure
        if "| Service | Strengths | Weaknesses |" in result_text:
            # Process the table to replace newlines
            # Find the table start
            table_start = result_text.find("| Service | Strengths | Weaknesses |")
            if table_start != -1:
                # Find the end of the table (next header or empty line)
                lines = result_text[table_start:].split("\n")
                table_end = 0
                for i, line in enumerate(lines):
                    if i > 2 and (not line.strip() or not line.startswith("|")):
                        table_end = i
                        break
                
                # Extract the table content
                table_content = lines[:table_end] if table_end > 0 else lines
                
                # Process the table rows to replace \n with bullet points in cells
                for i in range(len(table_content)):
                    if table_content[i].startswith("|") and i >= 2:  # Skip header and separator rows
                        # Split the row into cells while preserving the | characters
                        cells = table_content[i].split("|")
                        for j in range(len(cells)):
                            # Replace \n in each cell with a bullet point
                            cells[j] = cells[j].replace("\\n", " • ")
                        # Reconstruct the row
                        table_content[i] = "|".join(cells)
                
                # Reconstruct the text with the modified table
                modified_table = "\n".join(table_content)
                result_text = result_text[:table_start] + modified_table + result_text[table_start + len(modified_table):]
        
        return result_text
    except Exception as e:
        return f"**Error generating analysis:** {str(e)}"

def main():
    st.title("TranscribeSight: Your comprehensive Transcription Evaluation Platform")
    
    # Initialize session state for API keys
    load_api_keys()
    
    # Show API key input sidebar
    api_keys_sidebar()
    
    # Initialize chart library choice if not set
    if 'chart_library' not in st.session_state:
        st.session_state.chart_library = 'plotly'  # Default to plotly
        
    # Define vibrant color palette for service charts
    if 'service_colors' not in st.session_state:
        st.session_state.service_colors = {
            'AssemblyAI': '#1f77b4',
            'Whisper': '#ff7f0e',
            'GPT-4o': '#2ca02c',
            'GPT-4o Mini': '#d62728',
            'Deepgram Nova 3': '#9467bd',
            'Google Gemini Pro 2.5': '#8c564b',
            'Google Gemini 2.0 Flash': '#e377c2',
            'Google Gemini 2.5 Flash': '#7f7f7f',
            # Additional colors if needed
            'Service 9': '#bcbd22',
            'Service 10': '#17becf',
            'Service 11': '#f0027f',
            'Service 12': '#bf5b17',
            'Service 13': '#7fc97f',
            'Service 14': '#386cb0',
            'Service 15': '#fdc086'
        }
    
    st.write("Upload audio files and their reference transcriptions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Audio Files")
        uploaded_files = st.file_uploader("Upload WAV files", type=['wav'], accept_multiple_files=True)
        
    with col2:
        st.subheader("Reference Transcriptions")
        st.write("Upload text files with same names as audio files (e.g., audio1.wav → audio1.txt)")
        reference_files = st.file_uploader("Upload reference transcriptions", type=['txt'], accept_multiple_files=True)
    
    if uploaded_files and 'api_keys' in st.session_state:
        # Create reference text dictionary
        reference_texts = {}
        if reference_files:
            for ref_file in reference_files:
                base_name = os.path.splitext(ref_file.name)[0]
                reference_texts[base_name] = ref_file.getvalue().decode('utf-8')
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Processing", "Transcription Results", "Performance Metrics", "Accuracy Analysis", "Overall Analysis"])
        
        with tab1:
            st.header("Processing Files")
            
            # Display list of files to be processed
            st.subheader("Files to Process:")
            for file in uploaded_files:
                st.text(f"• {file.name}")
            
            # Replace duplicate buttons with a single button that has a unique key
            if st.button("Start Processing", key="start_processing_button"):
                # Check if at least one model is selected
                if any([
                    st.session_state.use_assemblyai,
                    st.session_state.use_whisper,
                    st.session_state.use_gpt4o,
                    st.session_state.use_gpt4o_mini,
                    st.session_state.use_deepgram,
                    st.session_state.use_gemini_pro,
                    st.session_state.use_gemini_flash,
                    st.session_state.use_gemini_flash_25
                ]):
                    results = []
                    performance_metrics = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Start processing
                    try:
                        for i, uploaded_file in enumerate(uploaded_files):
                            status_text.text(f"Processing {uploaded_file.name}...")
                            
                            # Save temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                temp_path = tmp_file.name
                            
                            # Dictionary to store API results and metrics
                            file_results = {'File Name': uploaded_file.name}
                            file_metrics = {'File Name': uploaded_file.name}  # Initialize with file name
                            
                            # Get reference text if available
                            base_name = os.path.splitext(uploaded_file.name)[0]
                            reference_text = reference_texts.get(base_name, "")
                            
                            # Add reference text to results
                            if reference_text:
                                file_results['Ground Truth'] = reference_text
                            else:
                                file_results['Ground Truth'] = "No reference available"
                            
                            # Get current semantic analysis options
                            semantic_options = {
                                'use_semascore': st.session_state.use_semascore,
                                'use_sts': st.session_state.use_sts,
                                'use_bertscore': st.session_state.use_bertscore,
                                'use_llm': st.session_state.use_llm,
                                'use_swwer': st.session_state.use_swwer
                            }
                            
                            # Process with each API if credentials are provided AND model is selected
                            if st.session_state.assemblyai_key and st.session_state.use_assemblyai:
                                try:
                                    assemblyai_client = init_assemblyai_client()
                                    text, metrics = transcribe_with_metrics(
                                        transcribe_assemblyai, temp_path, assemblyai_client, 'AssemblyAI')
                                    file_results['AssemblyAI'] = text
                                    
                                    accuracy_metrics = calculate_accuracy_metrics(text, reference_text) if reference_text else {}
                                    
                                    # Calculate semantic metrics if reference text is available and options are enabled
                                    semantic_metrics = calculate_semantic_metrics(text, reference_text, semantic_options) if reference_text else {}
                                    
                                    file_metrics.update({  # Changed to update() instead of direct assignment
                                        'AssemblyAI Processing Time (s)': metrics['processing_time_seconds'],
                                        'AssemblyAI Cost ($)': metrics['cost'],
                                        'AssemblyAI Word Count': metrics['words'],
                                        'AssemblyAI Success': metrics['success'],
                                        'AssemblyAI Word Error Rate (%)': accuracy_metrics.get('word_error_rate', None),
                                        'AssemblyAI Char Error Rate (%)': accuracy_metrics.get('character_error_rate', None),
                                        'AssemblyAI Token Error Rate (%)': accuracy_metrics.get('token_error_rate', None),
                                        'AssemblyAI Accuracy (%)': accuracy_metrics.get('accuracy_percentage', None),
                                        'AssemblyAI Word Diff': accuracy_metrics.get('word_diff', None),
                                        'AssemblyAI Char Diff': accuracy_metrics.get('char_diff', None),
                                        'AssemblyAI Match Error Rate (%)': accuracy_metrics.get('match_error_rate', None),
                                        'AssemblyAI Word Information Lost (%)': accuracy_metrics.get('word_information_lost', None),
                                        'AssemblyAI Word Information Preserved (%)': accuracy_metrics.get('word_information_preserved', None),
                                        'AssemblyAI SeMaScore': semantic_metrics.get('semascore', None),
                                        'AssemblyAI STS Similarity': semantic_metrics.get('sts_similarity', None),
                                        'AssemblyAI BERTScore': semantic_metrics.get('bertscore', None),
                                        'AssemblyAI LLM Meaning Preservation': semantic_metrics.get('llm_meaning_preservation', None),
                                        'AssemblyAI SWWER': semantic_metrics.get('swwer', None)
                                    })
                                except Exception as e:
                                    file_results['AssemblyAI'] = f"Error: {str(e)}"
                            
                            if st.session_state.openai_key:
                                # Initialize OpenAI client once for all OpenAI models
                                try:
                                    openai_client = init_openai_client()
                                except Exception as e:
                                    st.error(f"Error initializing OpenAI client: {str(e)}")
                                    openai_client = None
                                
                                # Whisper transcription
                                if st.session_state.use_whisper and openai_client:
                                    try:
                                        text, metrics = transcribe_with_metrics(
                                            transcribe_whisper, temp_path, openai_client, 'Whisper')
                                        file_results['Whisper'] = text
                                        
                                        accuracy_metrics = calculate_accuracy_metrics(text, reference_text) if reference_text else {}
                                        
                                        # Calculate semantic metrics for Whisper
                                        semantic_metrics = calculate_semantic_metrics(text, reference_text, semantic_options) if reference_text else {}
                                        
                                        file_metrics.update({
                                            'Whisper Processing Time (s)': metrics['processing_time_seconds'],
                                            'Whisper Cost ($)': metrics['cost'],
                                            'Whisper Word Count': metrics['words'],
                                            'Whisper Success': metrics['success'],
                                            'Whisper Word Error Rate (%)': accuracy_metrics.get('word_error_rate', None),
                                            'Whisper Char Error Rate (%)': accuracy_metrics.get('character_error_rate', None),
                                            'Whisper Token Error Rate (%)': accuracy_metrics.get('token_error_rate', None),
                                            'Whisper Accuracy (%)': accuracy_metrics.get('accuracy_percentage', None),
                                            'Whisper Word Diff': accuracy_metrics.get('word_diff', None),
                                            'Whisper Char Diff': accuracy_metrics.get('char_diff', None),
                                            'Whisper Match Error Rate (%)': accuracy_metrics.get('match_error_rate', None),
                                            'Whisper Word Information Lost (%)': accuracy_metrics.get('word_information_lost', None),
                                            'Whisper Word Information Preserved (%)': accuracy_metrics.get('word_information_preserved', None),
                                            'Whisper SeMaScore': semantic_metrics.get('semascore', None),
                                            'Whisper STS Similarity': semantic_metrics.get('sts_similarity', None),
                                            'Whisper BERTScore': semantic_metrics.get('bertscore', None),
                                            'Whisper LLM Meaning Preservation': semantic_metrics.get('llm_meaning_preservation', None),
                                            'Whisper SWWER': semantic_metrics.get('swwer', None)
                                        })
                                    except Exception as e:
                                        file_results['Whisper'] = f"Error: {str(e)}"
                                    
                                # GPT-4o transcription
                                if st.session_state.use_gpt4o and openai_client:
                                    try:
                                        text2, metrics2 = transcribe_with_metrics(
                                            transcribe_gpt4o, temp_path, openai_client, 'GPT-4o')
                                        file_results['GPT-4o'] = text2
                                        acc2 = calculate_accuracy_metrics(text2, reference_text) if reference_text else {}
                                        
                                        # Calculate semantic metrics for GPT-4o
                                        semantic_metrics2 = calculate_semantic_metrics(text2, reference_text, semantic_options) if reference_text else {}
                                        
                                        file_metrics.update({
                                            'GPT-4o Processing Time (s)': metrics2['processing_time_seconds'],
                                            'GPT-4o Cost ($)': metrics2['cost'],
                                            'GPT-4o Word Count': metrics2['words'],
                                            'GPT-4o Success': metrics2['success'],
                                            'GPT-4o Word Error Rate (%)': acc2.get('word_error_rate', None),
                                            'GPT-4o Char Error Rate (%)': acc2.get('character_error_rate', None),
                                            'GPT-4o Token Error Rate (%)': acc2.get('token_error_rate', None),
                                            'GPT-4o Accuracy (%)': acc2.get('accuracy_percentage', None),
                                            'GPT-4o Word Diff': acc2.get('word_diff', None),
                                            'GPT-4o Char Diff': acc2.get('char_diff', None),
                                            'GPT-4o Match Error Rate (%)': acc2.get('match_error_rate', None),
                                            'GPT-4o Word Information Lost (%)': acc2.get('word_information_lost', None),
                                            'GPT-4o Word Information Preserved (%)': acc2.get('word_information_preserved', None),
                                            'GPT-4o SeMaScore': semantic_metrics2.get('semascore', None),
                                            'GPT-4o STS Similarity': semantic_metrics2.get('sts_similarity', None),
                                            'GPT-4o BERTScore': semantic_metrics2.get('bertscore', None),
                                            'GPT-4o LLM Meaning Preservation': semantic_metrics2.get('llm_meaning_preservation', None),
                                            'GPT-4o SWWER': semantic_metrics2.get('swwer', None)
                                        })
                                    except Exception as e:
                                        file_results['GPT-4o'] = f"Error: {str(e)}"
                                    
                                # GPT-4o Mini transcription
                                if st.session_state.use_gpt4o_mini and openai_client:
                                    try:
                                        text3, metrics3 = transcribe_with_metrics(
                                            transcribe_gpt4o_mini, temp_path, openai_client, 'GPT-4o Mini')
                                        file_results['GPT-4o Mini'] = text3
                                        acc3 = calculate_accuracy_metrics(text3, reference_text) if reference_text else {}
                                        
                                        # Calculate semantic metrics for GPT-4o Mini
                                        semantic_metrics3 = calculate_semantic_metrics(text3, reference_text, semantic_options) if reference_text else {}
                                        
                                        file_metrics.update({
                                            'GPT-4o Mini Processing Time (s)': metrics3['processing_time_seconds'],
                                            'GPT-4o Mini Cost ($)': metrics3['cost'],
                                            'GPT-4o Mini Word Count': metrics3['words'],
                                            'GPT-4o Mini Success': metrics3['success'],
                                            'GPT-4o Mini Word Error Rate (%)': acc3.get('word_error_rate', None),
                                            'GPT-4o Mini Char Error Rate (%)': acc3.get('character_error_rate', None),
                                            'GPT-4o Mini Token Error Rate (%)': acc3.get('token_error_rate', None),
                                            'GPT-4o Mini Accuracy (%)': acc3.get('accuracy_percentage', None),
                                            'GPT-4o Mini Word Diff': acc3.get('word_diff', None),
                                            'GPT-4o Mini Char Diff': acc3.get('char_diff', None),
                                            'GPT-4o Mini Match Error Rate (%)': acc3.get('match_error_rate', None),
                                            'GPT-4o Mini Word Information Lost (%)': acc3.get('word_information_lost', None),
                                            'GPT-4o Mini Word Information Preserved (%)': acc3.get('word_information_preserved', None),
                                            'GPT-4o Mini SeMaScore': semantic_metrics3.get('semascore', None),
                                            'GPT-4o Mini STS Similarity': semantic_metrics3.get('sts_similarity', None),
                                            'GPT-4o Mini BERTScore': semantic_metrics3.get('bertscore', None),
                                            'GPT-4o Mini LLM Meaning Preservation': semantic_metrics3.get('llm_meaning_preservation', None),
                                            'GPT-4o Mini SWWER': semantic_metrics3.get('swwer', None)
                                        })
                                    except Exception as e:
                                        file_results['GPT-4o Mini'] = f"Error: {str(e)}"
                            
                            if st.session_state.deepgram_key and st.session_state.use_deepgram:
                                try:
                                    deepgram_client = init_deepgram_client()
                                    text, metrics = transcribe_with_metrics(
                                        transcribe_deepgram, temp_path, deepgram_client, 'Deepgram Nova 3')
                                    file_results['Deepgram Nova 3'] = text
                                    
                                    accuracy_metrics = calculate_accuracy_metrics(text, reference_text) if reference_text else {}
                                    
                                    # Calculate semantic metrics for Deepgram
                                    semantic_metrics_dg = calculate_semantic_metrics(text, reference_text, semantic_options) if reference_text else {}
                                    
                                    file_metrics.update({
                                        'Deepgram Nova 3 Processing Time (s)': metrics['processing_time_seconds'],
                                        'Deepgram Nova 3 Cost ($)': metrics['cost'],
                                        'Deepgram Nova 3 Word Count': metrics['words'],
                                        'Deepgram Nova 3 Success': metrics['success'],
                                        'Deepgram Nova 3 Word Error Rate (%)': accuracy_metrics.get('word_error_rate', None),
                                        'Deepgram Nova 3 Char Error Rate (%)': accuracy_metrics.get('character_error_rate', None),
                                        'Deepgram Nova 3 Token Error Rate (%)': accuracy_metrics.get('token_error_rate', None),
                                        'Deepgram Nova 3 Accuracy (%)': accuracy_metrics.get('accuracy_percentage', None),
                                        'Deepgram Nova 3 Word Diff': accuracy_metrics.get('word_diff', None),
                                        'Deepgram Nova 3 Char Diff': accuracy_metrics.get('char_diff', None),
                                        'Deepgram Nova 3 Match Error Rate (%)': accuracy_metrics.get('match_error_rate', None),
                                        'Deepgram Nova 3 Word Information Lost (%)': accuracy_metrics.get('word_information_lost', None),
                                        'Deepgram Nova 3 Word Information Preserved (%)': accuracy_metrics.get('word_information_preserved', None),
                                        'Deepgram Nova 3 SeMaScore': semantic_metrics_dg.get('semascore', None),
                                        'Deepgram Nova 3 STS Similarity': semantic_metrics_dg.get('sts_similarity', None),
                                        'Deepgram Nova 3 BERTScore': semantic_metrics_dg.get('bertscore', None),
                                        'Deepgram Nova 3 LLM Meaning Preservation': semantic_metrics_dg.get('llm_meaning_preservation', None),
                                        'Deepgram Nova 3 SWWER': semantic_metrics_dg.get('swwer', None)
                                    })
                                except Exception as e:
                                    file_results['Deepgram Nova 3'] = f"Error: {str(e)}"
                            
                            if st.session_state.google_gemini_key:
                                # Google Gemini Pro 2.5 transcription
                                if st.session_state.use_gemini_pro:
                                    try:
                                        gemini_client = init_gemini_client()
                                        text, metrics = transcribe_with_metrics(
                                            transcribe_gemini_pro_2_5, temp_path, gemini_client, 'Google Gemini Pro 2.5')
                                        file_results['Google Gemini Pro 2.5'] = text
                                        acc_g = calculate_accuracy_metrics(text, reference_text) if reference_text else {}
                                        
                                        # Calculate semantic metrics for Google Gemini Pro 2.5
                                        semantic_metrics_g = calculate_semantic_metrics(text, reference_text, semantic_options) if reference_text else {}
                                        
                                        file_metrics.update({
                                            'Google Gemini Pro 2.5 Processing Time (s)': metrics['processing_time_seconds'],
                                            'Google Gemini Pro 2.5 Cost ($)': metrics['cost'],
                                            'Google Gemini Pro 2.5 Word Count': metrics['words'],
                                            'Google Gemini Pro 2.5 Success': metrics['success'],
                                            'Google Gemini Pro 2.5 Word Error Rate (%)': acc_g.get('word_error_rate', None),
                                            'Google Gemini Pro 2.5 Char Error Rate (%)': acc_g.get('character_error_rate', None),
                                            'Google Gemini Pro 2.5 Token Error Rate (%)': acc_g.get('token_error_rate', None),
                                            'Google Gemini Pro 2.5 Accuracy (%)': acc_g.get('accuracy_percentage', None),
                                            'Google Gemini Pro 2.5 Word Diff': acc_g.get('word_diff', None),
                                            'Google Gemini Pro 2.5 Char Diff': acc_g.get('char_diff', None),
                                            'Google Gemini Pro 2.5 Match Error Rate (%)': acc_g.get('match_error_rate', None),
                                            'Google Gemini Pro 2.5 Word Information Lost (%)': acc_g.get('word_information_lost', None),
                                            'Google Gemini Pro 2.5 Word Information Preserved (%)': acc_g.get('word_information_preserved', None),
                                            'Google Gemini Pro 2.5 SeMaScore': semantic_metrics_g.get('semascore', None),
                                            'Google Gemini Pro 2.5 STS Similarity': semantic_metrics_g.get('sts_similarity', None),
                                            'Google Gemini Pro 2.5 BERTScore': semantic_metrics_g.get('bertscore', None),
                                            'Google Gemini Pro 2.5 LLM Meaning Preservation': semantic_metrics_g.get('llm_meaning_preservation', None),
                                            'Google Gemini Pro 2.5 SWWER': semantic_metrics_g.get('swwer', None)
                                        })
                                    except Exception as e:
                                        file_results['Google Gemini Pro 2.5'] = f"Error: {str(e)}"
                                
                                # Also run Google Gemini 2.0 Flash
                                if st.session_state.use_gemini_flash:
                                    try:
                                        text_f, metrics_f = transcribe_with_metrics(
                                            transcribe_gemini_2flash, temp_path, gemini_client, 'Google Gemini 2.0 Flash')
                                        file_results['Google Gemini 2.0 Flash'] = text_f
                                        acc_f = calculate_accuracy_metrics(text_f, reference_text) if reference_text else {}
                                        
                                        # Calculate semantic metrics for Google Gemini 2.0 Flash
                                        semantic_metrics_f = calculate_semantic_metrics(text_f, reference_text, semantic_options) if reference_text else {}
                                        
                                        file_metrics.update({
                                            'Google Gemini 2.0 Flash Processing Time (s)': metrics_f['processing_time_seconds'],
                                            'Google Gemini 2.0 Flash Cost ($)': metrics_f['cost'],
                                            'Google Gemini 2.0 Flash Word Count': metrics_f['words'],
                                            'Google Gemini 2.0 Flash Success': metrics_f['success'],
                                            'Google Gemini 2.0 Flash Word Error Rate (%)': acc_f.get('word_error_rate', None),
                                            'Google Gemini 2.0 Flash Char Error Rate (%)': acc_f.get('character_error_rate', None),
                                            'Google Gemini 2.0 Flash Token Error Rate (%)': acc_f.get('token_error_rate', None),
                                            'Google Gemini 2.0 Flash Accuracy (%)': acc_f.get('accuracy_percentage', None),
                                            'Google Gemini 2.0 Flash Word Diff': acc_f.get('word_diff', None),
                                            'Google Gemini 2.0 Flash Char Diff': acc_f.get('char_diff', None),
                                            'Google Gemini 2.0 Flash Match Error Rate (%)': acc_f.get('match_error_rate', None),
                                            'Google Gemini 2.0 Flash Word Information Lost (%)': acc_f.get('word_information_lost', None),
                                            'Google Gemini 2.0 Flash Word Information Preserved (%)': acc_f.get('word_information_preserved', None),
                                            'Google Gemini 2.0 Flash SeMaScore': semantic_metrics_f.get('semascore', None),
                                            'Google Gemini 2.0 Flash STS Similarity': semantic_metrics_f.get('sts_similarity', None),
                                            'Google Gemini 2.0 Flash BERTScore': semantic_metrics_f.get('bertscore', None),
                                            'Google Gemini 2.0 Flash LLM Meaning Preservation': semantic_metrics_f.get('llm_meaning_preservation', None),
                                            'Google Gemini 2.0 Flash SWWER': semantic_metrics_f.get('swwer', None)
                                        })
                                    except Exception as e:
                                        file_results['Google Gemini 2.0 Flash'] = f"Error: {str(e)}"
                                
                                # Google Gemini 2.5 Flash transcription
                                if st.session_state.use_gemini_flash_25:
                                    try:
                                        text_f25, metrics_f25 = transcribe_with_metrics(
                                            transcribe_gemini_2_5_flash, temp_path, gemini_client, 'Google Gemini 2.5 Flash')
                                        file_results['Google Gemini 2.5 Flash'] = text_f25
                                        acc_f25 = calculate_accuracy_metrics(text_f25, reference_text) if reference_text else {}
                                        
                                        # Calculate semantic metrics for Google Gemini 2.5 Flash
                                        semantic_metrics_f25 = calculate_semantic_metrics(text_f25, reference_text, semantic_options) if reference_text else {}
                                        
                                        file_metrics.update({
                                            'Google Gemini 2.5 Flash Processing Time (s)': metrics_f25['processing_time_seconds'],
                                            'Google Gemini 2.5 Flash Cost ($)': metrics_f25['cost'],
                                            'Google Gemini 2.5 Flash Word Count': metrics_f25['words'],
                                            'Google Gemini 2.5 Flash Success': metrics_f25['success'],
                                            'Google Gemini 2.5 Flash Word Error Rate (%)': acc_f25.get('word_error_rate', None),
                                            'Google Gemini 2.5 Flash Char Error Rate (%)': acc_f25.get('character_error_rate', None),
                                            'Google Gemini 2.5 Flash Token Error Rate (%)': acc_f25.get('token_error_rate', None),
                                            'Google Gemini 2.5 Flash Accuracy (%)': acc_f25.get('accuracy_percentage', None),
                                            'Google Gemini 2.5 Flash Word Diff': acc_f25.get('word_diff', None),
                                            'Google Gemini 2.5 Flash Char Diff': acc_f25.get('char_diff', None),
                                            'Google Gemini 2.5 Flash Match Error Rate (%)': acc_f25.get('match_error_rate', None),
                                            'Google Gemini 2.5 Flash Word Information Lost (%)': acc_f25.get('word_information_lost', None),
                                            'Google Gemini 2.5 Flash Word Information Preserved (%)': acc_f25.get('word_information_preserved', None),
                                            'Google Gemini 2.5 Flash SeMaScore': semantic_metrics_f25.get('semascore', None),
                                            'Google Gemini 2.5 Flash STS Similarity': semantic_metrics_f25.get('sts_similarity', None),
                                            'Google Gemini 2.5 Flash BERTScore': semantic_metrics_f25.get('bertscore', None),
                                            'Google Gemini 2.5 Flash LLM Meaning Preservation': semantic_metrics_f25.get('llm_meaning_preservation', None),
                                            'Google Gemini 2.5 Flash SWWER': semantic_metrics_f25.get('swwer', None)
                                        })
                                    except Exception as e:
                                        file_results['Google Gemini 2.5 Flash'] = f"Error: {str(e)}"
                            
                            results.append(file_results)
                            performance_metrics.append(file_metrics)
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(uploaded_files))
                            
                            # Clean up temp file
                            os.unlink(temp_path)
                        
                        # Store results in session state
                        st.session_state.results = results
                        st.session_state.performance_metrics = performance_metrics
                        
                        status_text.text("Processing complete!")
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                else:
                    st.error("Please select at least one model in the Models Selection panel.")
            else:
                st.info("Click 'Start Processing' to begin transcription of the uploaded files.")
        
        with tab2:
            st.header("Transcription Results")
            
            if hasattr(st.session_state, 'results') and st.session_state.results:
                # Create DataFrame and display results
                df_transcripts = pd.DataFrame(st.session_state.results)
                st.dataframe(df_transcripts)
                
                # Convert DataFrame to Excel bytes
                excel_data = convert_df_to_excel(df_transcripts)
                
                # Provide download button for transcripts
                st.download_button(
                    label="Download Transcription Results (Excel)",
                    data=excel_data,
                    file_name='transcription_results.xlsx',
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("Process files to see transcription results.")
        
        with tab3:
            st.header("Performance Metrics")
            
            if hasattr(st.session_state, 'performance_metrics') and st.session_state.performance_metrics:
                # Create DataFrame for metrics
                df_metrics = pd.DataFrame(st.session_state.performance_metrics)
                
                # Display summary metrics
                st.subheader("Summary Statistics")
                total_cost = df_metrics[[col for col in df_metrics.columns if 'Cost' in col]].sum().sum()
                total_time = df_metrics[[col for col in df_metrics.columns if 'Processing Time' in col]].sum().sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Cost ($)", f"{total_cost:.4f}")
                with col2:
                    st.metric("Total Processing Time (s)", f"{total_time:.2f}")
                
                # NEW: Calculate per-service totals for time and cost
                st.subheader("Per-Service Totals")
                
                # Get list of services used
                service_names = set()
                for col in df_metrics.columns:
                    for service in ['AssemblyAI', 'Whisper', 'GPT-4o', 'GPT-4o Mini', 'Deepgram Nova 3', 
                                  'Google Gemini Pro 2.5', 'Google Gemini 2.0 Flash', 'Google Gemini 2.5 Flash']:
                        if service in col:
                            service_names.add(service)
                
                service_names = sorted(service_names)
                
                # Create dataframes for per-service totals
                service_totals = pd.DataFrame(columns=["Service", "Total Cost ($)", "Total Time (s)"])
                cost_data = []
                time_data = []
                
                for service in service_names:
                    service_cost_cols = [col for col in df_metrics.columns if service in col and 'Cost' in col]
                    service_time_cols = [col for col in df_metrics.columns if service in col and 'Processing Time' in col]
                    
                    if service_cost_cols and service_time_cols:
                        service_total_cost = df_metrics[service_cost_cols].sum().sum()
                        service_total_time = df_metrics[service_time_cols].sum().sum()
                        
                        # Create a new DataFrame with the service data
                        service_df = pd.DataFrame({
                            "Service": [service],
                            "Total Cost ($)": [service_total_cost],
                            "Total Time (s)": [service_total_time]
                        })
                        
                        # Handle empty or NA values before concatenation
                        service_df = service_df.fillna(0)  # Replace NaN with zeros
                        service_totals = pd.concat([service_totals, service_df], ignore_index=True)
                        
                        # Add to lists for plotting
                        cost_data.append(service_total_cost)
                        time_data.append(service_total_time)
                
                # Display the service totals table
                st.dataframe(service_totals)
                
                # Create tabs for different visualizations
                perf_tabs = st.tabs(["Cost Comparison", "Time Comparison", "Combined View", "Efficiency Analysis"])
                
                with perf_tabs[0]:
                    st.subheader("Cost Comparison by Service")
                    
                    # Create a bar chart with Plotly for cost
                    cost_fig = go.Figure()
                    cost_fig.add_trace(go.Bar(
                        x=service_names,
                        y=cost_data,
                        marker_color=[st.session_state.service_colors.get(service, "#1f77b4") for service in service_names],
                        text=[f"${cost:.4f}" for cost in cost_data],
                        textposition='auto',
                    ))
                    cost_fig.update_layout(
                        title="Total Cost by Service ($)",
                        xaxis_title="Service",
                        yaxis_title="Cost ($)",
                        height=500
                    )
                    st.plotly_chart(cost_fig, use_container_width=True)
                    
                    # Create a pie chart showing cost distribution
                    cost_pie = go.Figure(data=[go.Pie(
                        labels=service_names,
                        values=cost_data,
                        marker=dict(colors=[st.session_state.service_colors.get(service, "#1f77b4") for service in service_names]),
                        textinfo='label+percent',
                        insidetextorientation='radial',
                        hole=.3
                    )])
                    cost_pie.update_layout(
                        title="Cost Distribution by Service",
                        height=500
                    )
                    st.plotly_chart(cost_pie, use_container_width=True)
                
                with perf_tabs[1]:
                    st.subheader("Processing Time Comparison by Service")
                    
                    # Create a bar chart with Plotly for processing time
                    time_fig = go.Figure()
                    time_fig.add_trace(go.Bar(
                        x=service_names,
                        y=time_data,
                        marker_color=[st.session_state.service_colors.get(service, "#1f77b4") for service in service_names],
                        text=[f"{time:.2f}s" for time in time_data],
                        textposition='auto',
                    ))
                    time_fig.update_layout(
                        title="Total Processing Time by Service (seconds)",
                        xaxis_title="Service",
                        yaxis_title="Time (seconds)",
                        height=500
                    )
                    st.plotly_chart(time_fig, use_container_width=True)
                    
                    # Create a pie chart showing time distribution
                    time_pie = go.Figure(data=[go.Pie(
                        labels=service_names,
                        values=time_data,
                        marker=dict(colors=[st.session_state.service_colors.get(service, "#1f77b4") for service in service_names]),
                        textinfo='label+percent',
                        insidetextorientation='radial',
                        hole=.3
                    )])
                    time_pie.update_layout(
                        title="Processing Time Distribution by Service",
                        height=500
                    )
                    st.plotly_chart(time_pie, use_container_width=True)
                
                with perf_tabs[2]:
                    st.subheader("Combined Cost and Time Visualization")
                    
                    # Create a combined bar chart with both cost and time
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name="Total Cost ($)",
                        x=service_names,
                        y=cost_data,
                        marker_color='#1f77b4',
                        text=[f"${cost:.4f}" for cost in cost_data],
                        textposition='auto',
                        opacity=0.7
                    ))
                    
                    # Add normalized time data for better visualization (scale to similar range as cost)
                    # Find max values to normalize
                    max_cost = max(cost_data) if cost_data else 1
                    max_time = max(time_data) if time_data else 1
                    scale_factor = max_cost / max_time if max_time > 0 else 1
                    
                    fig.add_trace(go.Bar(
                        name="Total Time (scaled)",
                        x=service_names,
                        y=[time * scale_factor for time in time_data],
                        marker_color='#ff7f0e',
                        text=[f"{time:.2f}s" for time in time_data],
                        textposition='auto',
                        opacity=0.7
                    ))
                    
                    fig.update_layout(
                        title="Combined Cost and Time Comparison",
                        xaxis_title="Service",
                        yaxis_title="Value",
                        barmode='group',
                        height=500
                    )
                    
                    # Add a secondary y-axis for the actual time values
                    fig.update_layout(
                        yaxis=dict(
                            title="Cost ($)",
                            side="left"
                        ),
                        yaxis2=dict(
                            title="Time (s)",
                            side="right",
                            overlaying="y",
                            range=[0, max_time * 1.1],
                            showgrid=False
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create a scatter plot showing cost vs time
                    scatter_fig = go.Figure()
                    scatter_fig.add_trace(go.Scatter(
                        x=time_data,
                        y=cost_data,
                        mode='markers+text',
                        marker=dict(
                            size=15,
                            color=[st.session_state.service_colors.get(service, "#1f77b4") for service in service_names],
                            opacity=0.8
                        ),
                        text=service_names,
                        textposition="top center",
                        name="Services"
                    ))
                    scatter_fig.update_layout(
                        title="Cost vs. Processing Time",
                        xaxis_title="Processing Time (seconds)",
                        yaxis_title="Cost ($)",
                        height=500
                    )
                    st.plotly_chart(scatter_fig, use_container_width=True)
                
                with perf_tabs[3]:
                    st.subheader("Efficiency Analysis")
                    
                    # Calculate efficiency metrics
                    efficiency_df = pd.DataFrame({
                        "Service": service_names,
                        "Cost ($)": cost_data,
                        "Time (s)": time_data,
                    })
                    
                    # Add derived metrics
                    if len(efficiency_df) > 0:
                        efficiency_df["Cost per Second ($)"] = efficiency_df["Cost ($)"] / efficiency_df["Time (s)"].replace(0, float('nan'))
                        efficiency_df["Cost-Time Index"] = (efficiency_df["Cost ($)"] * efficiency_df["Time (s)"]) ** 0.5
                        efficiency_df["Efficiency Score"] = 100 / (1 + efficiency_df["Cost-Time Index"])
                    
                        # Display the efficiency metrics
                        st.dataframe(efficiency_df)
                        
                        # Create efficiency score chart
                        eff_fig = go.Figure()
                        eff_fig.add_trace(go.Bar(
                            x=efficiency_df["Service"],
                            y=efficiency_df["Efficiency Score"],
                            marker_color=[st.session_state.service_colors.get(service, "#1f77b4") for service in efficiency_df["Service"]],
                            text=[f"{score:.1f}" for score in efficiency_df["Efficiency Score"]],
                            textposition='auto'
                        ))
                        eff_fig.update_layout(
                            title="Efficiency Score by Service (Higher is Better)",
                            xaxis_title="Service",
                            yaxis_title="Efficiency Score",
                            height=500
                        )
                        st.plotly_chart(eff_fig, use_container_width=True)
                        
                        # Add explanation of the efficiency score
                        with st.expander("About Efficiency Score"):
                            st.markdown("""
                            **Efficiency Score** is calculated as:
                            
                            ```
                            Efficiency Score = 100 / (1 + Cost-Time Index)
                            ```
                            
                            where **Cost-Time Index** is the geometric mean of cost and processing time.
                            
                            A higher efficiency score indicates better performance considering both cost and time factors.
                            
                            This provides a balanced metric to compare services considering both dimensions.
                            """)
                    else:
                        st.info("Not enough data to calculate efficiency metrics.")
                
                # Display detailed metrics
                st.subheader("Detailed Metrics")
                st.dataframe(df_metrics)
                
                # Convert DataFrame to Excel bytes
                excel_data = convert_df_to_excel(df_metrics)
                
                # Provide download button for metrics
                st.download_button(
                    label="Download Performance Metrics (Excel)",
                    data=excel_data,
                    file_name='performance_metrics.xlsx',
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.info("Process files to see performance metrics.")
        
        with tab4:
                st.header("Accuracy Analysis")
                
                if reference_files and hasattr(st.session_state, 'performance_metrics') and st.session_state.performance_metrics:
                    # Create accuracy comparison DataFrame
                    df_metrics = pd.DataFrame(st.session_state.performance_metrics)
                    
                    # Process traditional accuracy metrics
                    accuracy_cols = [col for col in df_metrics.columns 
                                   if any(x in col for x in ['Error Rate', 'Accuracy', 'Diff', 'Match Error Rate', 'Word Information'])]
                    
                    # Check for any enabled semantic metrics and add their columns to the dataframe
                    semantic_cols = []
                    if st.session_state.get('use_semascore', False):
                        semantic_cols.extend([col for col in df_metrics.columns if 'semascore' in col.lower()])
                    
                    if st.session_state.get('use_sts', False):
                        semantic_cols.extend([col for col in df_metrics.columns if 'sts similarity' in col.lower() or 'sts_similarity' in col.lower()])
                    
                    if st.session_state.get('use_bertscore', False):
                        semantic_cols.extend([col for col in df_metrics.columns if 'bertscore' in col.lower()])
                    
                    if st.session_state.get('use_llm', False):
                        semantic_cols.extend([col for col in df_metrics.columns if 'llm meaning' in col.lower() or 'llm_meaning' in col.lower()])
                    
                    if st.session_state.get('use_swwer', False):
                        semantic_cols.extend([col for col in df_metrics.columns if 'swwer' in col.lower()])
                    
                    # Combine traditional and semantic metrics
                    all_metrics_cols = accuracy_cols + semantic_cols
                    
                    # Create a comprehensive dataframe with all metrics
                    accuracy_df_original = df_metrics[['File Name'] + all_metrics_cols]
                    
                    # NEW: Get list of services used to reorganize data by service
                    service_names = set()
                    for col in df_metrics.columns:
                        for service in ['AssemblyAI', 'Whisper', 'GPT-4o', 'GPT-4o Mini', 'Deepgram Nova 3', 
                                      'Google Gemini Pro 2.5', 'Google Gemini 2.0 Flash', 'Google Gemini 2.5 Flash']:
                            if service in col:
                                service_names.add(service)
                    
                    service_names = sorted(service_names)
                    
                    # NEW: Reorganize data by service instead of by metric type
                    reorganized_data = []
                    
                    for _, row in df_metrics.iterrows():
                        file_name = row['File Name']
                        row_data = {'File Name': file_name}
                        
                        # For each service, add all its metrics
                        for service in service_names:
                            # Traditional metrics
                            service_metrics = {
                                f"{service} - Word Accuracy (%)": None,  # Changed from Accuracy (%) to Word Accuracy (%)
                                f"{service} - Word Error Rate (%)": None,
                                f"{service} - Char Error Rate (%)": None,
                                f"{service} - Token Error Rate (%)": None,
                                f"{service} - Match Error Rate (%)": None,
                                f"{service} - Word Information Preserved (%)": None,
                                f"{service} - Word Information Lost (%)": None
                            }
                            
                            # Add semantic metrics if enabled
                            if st.session_state.get('use_semascore', False):
                                service_metrics[f"{service} - SeMaScore"] = None
                            if st.session_state.get('use_sts', False):
                                service_metrics[f"{service} - STS Similarity"] = None
                            if st.session_state.get('use_bertscore', False):
                                service_metrics[f"{service} - BERTScore"] = None
                            if st.session_state.get('use_llm', False):
                                service_metrics[f"{service} - LLM Meaning Preservation"] = None
                            if st.session_state.get('use_swwer', False):
                                service_metrics[f"{service} - SWWER"] = None
                            
                            # Now fill in the values from the original data
                            for col in df_metrics.columns:
                                if service in col:
                                    # Get metric type
                                    metric_type = col.replace(service, '').strip()
                                    
                                    # Special handling for Word Accuracy - calculate from WER
                                    if 'Word Error Rate (%)' in metric_type and row[col] is not None:
                                        service_metrics[f"{service} - Word Accuracy (%)"] = 100 - row[col]
                                    
                                    # Handle other metrics normally
                                    new_col_name = f"{service} - {metric_type}"
                                    if new_col_name in service_metrics:
                                        service_metrics[new_col_name] = row[col]
                                    
                                    # Map any other columns that might have different naming patterns
                                    mapping = {
                                        f"{service} - WordErrorRate": f"{service} - Word Error Rate (%)",
                                        f"{service} - CharErrorRate": f"{service} - Char Error Rate (%)",
                                        f"{service} - TokenErrorRate": f"{service} - Token Error Rate (%)",
                                        f"{service} - MatchErrorRate": f"{service} - Match Error Rate (%)",
                                        f"{service} - WordInformationPreserved": f"{service} - Word Information Preserved (%)",
                                        f"{service} - WordInformationLost": f"{service} - Word Information Lost (%)",
                                        f"{service} - SeMaScore": f"{service} - SeMaScore",
                                        f"{service} - STSSimilarity": f"{service} - STS Similarity",
                                        f"{service} - BERTScore": f"{service} - BERTScore",
                                        f"{service} - LLMMeaningPreservation": f"{service} - LLM Meaning Preservation",
                                        f"{service} - SWWER": f"{service} - SWWER"
                                    }
                                    
                                    # Find the closest match
                                    for k, v in mapping.items():
                                        if new_col_name.startswith(k) or k.startswith(new_col_name):
                                            if v in service_metrics:
                                                # Special handling for Word Accuracy calculated from WER
                                                if v == f"{service} - Word Error Rate (%)" and row[col] is not None:
                                                    service_metrics[f"{service} - Word Accuracy (%)"] = 100 - row[col]
                                                service_metrics[v] = row[col]
                                            break
                            
                            # Add the service metrics to the row data
                            row_data.update(service_metrics)
                        
                        reorganized_data.append(row_data)
                    
                    # Create the reorganized DataFrame
                    accuracy_df = pd.DataFrame(reorganized_data)
                    
                    # Display accuracy metrics
                    st.subheader("Accuracy Metrics")
                    
                    # Add metric explanation box
                    st.info("""
                    **Understanding Accuracy Metrics:**
                    - **Word Error Rate (WER)**: Lower is better - Percentage of words transcribed incorrectly
                    - **Character Error Rate (CER)**: Lower is better - Percentage of characters transcribed incorrectly
                    - **Token Error Rate (TER)**: Lower is better - Percentage of tokens (words and punctuation) transcribed incorrectly
                    - **Word Accuracy Percentage**: Higher is better - (100 - WER) - Overall word-level transcription accuracy
                    - **Match Error Rate (MER)**: Lower is better - Measures errors including substitutions, deletions, and insertions
                    - **Word Information Lost (WIL)**: Lower is better - Amount of information lost during transcription
                    
                    **Understanding Semantic Metrics (if enabled):**
                    - **SeMaScore**: Higher is better (0-100) - Semantic similarity based on sentence embeddings
                    - **STS Cosine Similarity**: Higher is better (0-100) - Measures semantic similarity using cosine distance
                    - **BERTScore**: Higher is better (0-100) - Uses contextual embeddings to measure semantic similarity
                    - **LLM-based Meaning Preservation**: Higher is better (0-100) - GPT-4o mini evaluation of meaning preservation
                    - **SWWER**: Lower is better - Word error rate that weighs errors by semantic importance
                    """)
                    
                    # Display the dataframe with all metrics
                    st.dataframe(accuracy_df)
                    
                    # Word Error Rate comparison
                    st.subheader("Word Error Rate by Service")
                    st.warning("**Lower is better** - Percentage of words that were transcribed incorrectly")
                    wer_means = accuracy_df_original[[col for col in accuracy_df_original.columns 
                                           if 'Word Error Rate' in col]].mean()
                    st.bar_chart(wer_means)
                    
                    # Character Error Rate comparison
                    st.subheader("Character Error Rate by Service")
                    st.warning("**Lower is better** - Percentage of characters transcribed incorrectly")
                    cer_means = accuracy_df_original[[col for col in accuracy_df_original.columns 
                                           if 'Char Error Rate' in col]].mean()
                    st.bar_chart(cer_means)
                    
                    # Match Error Rate comparison
                    st.subheader("Match Error Rate by Service")
                    st.warning("**Lower is better** - Overall error rate including substitutions, deletions, and insertions")
                    mer_means = accuracy_df_original[[col for col in accuracy_df_original.columns 
                                           if 'Match Error Rate' in col]].mean()
                    st.bar_chart(mer_means)
                    
                    # Word Information Lost comparison
                    st.subheader("Word Information Lost by Service")
                    st.warning("**Lower is better** - Measures information lost during transcription")
                    wil_means = accuracy_df_original[[col for col in accuracy_df_original.columns 
                                           if 'Word Information Lost' in col]].mean()
                    st.bar_chart(wil_means)
                    
                    # Add Token Error Rate comparison specifically
                    st.subheader("Token Error Rate by Service")
                    st.warning("**Lower is better** - Percentage of tokens (words, punctuation) incorrectly transcribed")
                    ter_cols = [col for col in accuracy_df_original.columns if 'Token Error Rate' in col]
                    if ter_cols:
                        ter_means = accuracy_df_original[ter_cols].mean()
                        st.bar_chart(ter_means)
                        
                        # Add detailed explanation of TER
                        with st.expander("About Token Error Rate (TER)"):
                            st.markdown("""
                            **Token Error Rate (TER)** is a more linguistically informed metric than Word Error Rate.
                            
                            TER uses tokenization to split text into meaningful units (including punctuation and contractions)
                            and then calculates the minimum edit distance between the reference and hypothesis tokens.
                            
                            TER = (S + D + I) / N, where:
                            - S = substitutions
                            - D = deletions
                            - I = insertions 
                            - N = number of tokens in reference
                            
                            Lower values indicate better performance.
                            """)
                    else:
                        st.info("Token Error Rate data not available")
                    
                    # Add a combined error rates visualization
                    st.subheader("Error Rate Comparison by Service")
                    st.info("Comparison of different error rate metrics (lower is better)")
                    
                    # Get columns for each error type
                    wer_cols = [col for col in accuracy_df_original.columns if 'Word Error Rate' in col]
                    cer_cols = [col for col in accuracy_df_original.columns if 'Char Error Rate' in col]
                    ter_cols = [col for col in accuracy_df_original.columns if 'Token Error Rate' in col]
                    mer_cols = [col for col in accuracy_df_original.columns if 'Match Error Rate' in col]
                    
                    # Create comparison dataframe
                    if wer_cols and cer_cols:
                        error_comparison = pd.DataFrame()
                        
                        for service in service_names:
                            service_wer_col = next((col for col in wer_cols if service in col), None)
                            service_cer_col = next((col for col in cer_cols if service in col), None)
                            service_ter_col = next((col for col in ter_cols if service in col), None) if ter_cols else None
                            service_mer_col = next((col for col in mer_cols if service in col), None) if mer_cols else None
                            
                            if service_wer_col and service_cer_col:
                                error_comparison[f"{service} WER"] = accuracy_df_original[service_wer_col]
                                error_comparison[f"{service} CER"] = accuracy_df_original[service_cer_col]
                                if service_ter_col:
                                    error_comparison[f"{service} TER"] = accuracy_df_original[service_ter_col]
                                if service_mer_col:
                                    error_comparison[f"{service} MER"] = accuracy_df_original[service_mer_col]
                        
                        # Calculate means for each service and error type
                        error_means = error_comparison.mean()
                        
                        # Create a multi-bar chart using Plotly
                        
                        # Prepare data for visualization
                        services = []
                        wer_values = []
                        cer_values = []
                        ter_values = []
                        mer_values = []
                        
                        for service in service_names:
                            services.append(service)
                            service_wer = error_means.get(f"{service} WER", 0)
                            service_cer = error_means.get(f"{service} CER", 0)
                            service_ter = error_means.get(f"{service} TER", 0)
                            service_mer = error_means.get(f"{service} MER", 0)
                            wer_values.append(service_wer)
                            cer_values.append(service_cer)
                            ter_values.append(service_ter)
                            mer_values.append(service_mer)
                        
                        # Create figure
                        fig = go.Figure()
                        
                        # Add bars for each error type
                        fig.add_trace(go.Bar(name='Word Error Rate', x=services, y=wer_values, marker_color='#FF6347'))
                        fig.add_trace(go.Bar(name='Character Error Rate', x=services, y=cer_values, marker_color='#4682B4'))
                        
                        # Only add TER and MER if we have data
                        if any(v > 0 for v in ter_values):
                            fig.add_trace(go.Bar(name='Token Error Rate', x=services, y=ter_values, marker_color='#32CD32'))
                        
                        if any(v > 0 for v in mer_values):
                            fig.add_trace(go.Bar(name='Match Error Rate', x=services, y=mer_values, marker_color='#9370DB'))
                        
                        # Update layout
                        fig.update_layout(
                            title='Error Rate Metrics by Service (Lower is Better)',
                            xaxis_title='Service',
                            yaxis_title='Error Rate (%)',
                            barmode='group',
                            height=500
                        )
                        
                        # Display the chart
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Check if any semantic metrics are enabled and have results
                    semantic_cols = []
                    semantic_metrics_exist = False
                    
                    # Check for semantic metrics in the data
                    if any(st.session_state.get(k, False) for k in ['use_semascore', 'use_sts', 'use_bertscore', 'use_llm', 'use_swwer']):
                        # Add separator for semantic analysis
                        st.markdown("---")
                        st.subheader("Semantic Analysis Results")
                        
                        st.info("""
                        **Understanding Semantic Metrics:**
                        - **SeMaScore**: Higher is better (0-100) - Semantic similarity based on sentence embeddings
                        - **STS Cosine Similarity**: Higher is better (0-100) - Measures semantic similarity using cosine distance
                        - **BERTScore**: Higher is better (0-100) - Uses contextual embeddings to measure semantic similarity
                        - **LLM-based Meaning Preservation**: Higher is better (0-100) - GPT-4o mini evaluation of meaning preservation
                        - **SWWER**: Lower is better - Word error rate weighted by semantic importance
                        """)
                        
                        # First add the new combined semantic metrics chart (for the higher-is-better metrics)
                        st.subheader("Semantic Metrics Comparison by Service")
                        st.success("**Higher is better** - Comparison of semantic similarity metrics (0-100)")
                        
                        # Get columns for each semantic metric type (excluding SWWER since it's lower-is-better)
                        semascore_cols = [col for col in accuracy_df_original.columns if 'semascore' in col.lower()]
                        sts_cols = [col for col in accuracy_df_original.columns if 'sts similarity' in col.lower() or 'sts_similarity' in col.lower()]
                        bertscore_cols = [col for col in accuracy_df_original.columns if 'bertscore' in col.lower()]
                        llm_cols = [col for col in accuracy_df_original.columns if 'llm meaning' in col.lower() or 'meaning preservation' in col.lower()]
                        
                        # Check if we have data for at least one semantic metric
                        if any([semascore_cols, sts_cols, bertscore_cols, llm_cols]):
                            # Create comparison dataframe
                            semantic_comparison = pd.DataFrame()
                            
                            for service in service_names:
                                service_semascore_col = next((col for col in semascore_cols if service in col), None)
                                service_sts_col = next((col for col in sts_cols if service in col), None)
                                service_bertscore_col = next((col for col in bertscore_cols if service in col), None)
                                service_llm_col = next((col for col in llm_cols if service in col), None)
                                
                                # Add available metrics to the comparison dataframe
                                if service_semascore_col:
                                    semantic_comparison[f"{service} SeMaScore"] = accuracy_df_original[service_semascore_col]
                                if service_sts_col:
                                    semantic_comparison[f"{service} STS"] = accuracy_df_original[service_sts_col]
                                if service_bertscore_col:
                                    semantic_comparison[f"{service} BERTScore"] = accuracy_df_original[service_bertscore_col]
                                if service_llm_col:
                                    semantic_comparison[f"{service} LLM"] = accuracy_df_original[service_llm_col]
                            
                            # Calculate means for each service and semantic metric
                            semantic_means = semantic_comparison.mean()
                            
                            # Create a multi-bar chart using Plotly
                            
                            # Prepare data for visualization
                            services = []
                            semascore_values = []
                            sts_values = []
                            bertscore_values = []
                            llm_values = []
                            
                            for service in service_names:
                                services.append(service)
                                semascore_values.append(semantic_means.get(f"{service} SeMaScore", 0))
                                sts_values.append(semantic_means.get(f"{service} STS", 0))
                                bertscore_values.append(semantic_means.get(f"{service} BERTScore", 0))
                                llm_values.append(semantic_means.get(f"{service} LLM", 0))
                            
                            # Create figure with only metrics that have data
                            fig = go.Figure()
                            
                            # Add bars for each metric only if they have data
                            if any(v > 0 for v in semascore_values):
                                fig.add_trace(go.Bar(name='SeMaScore', x=services, y=semascore_values, marker_color='#FF6347'))
                            
                            if any(v > 0 for v in sts_values):
                                fig.add_trace(go.Bar(name='STS Similarity', x=services, y=sts_values, marker_color='#4682B4'))
                            
                            if any(v > 0 for v in bertscore_values):
                                fig.add_trace(go.Bar(name='BERTScore', x=services, y=bertscore_values, marker_color='#32CD32'))
                            
                            if any(v > 0 for v in llm_values):
                                fig.add_trace(go.Bar(name='LLM Meaning', x=services, y=llm_values, marker_color='#9370DB'))
                            
                            # Update layout
                            fig.update_layout(
                                title='Semantic Similarity Metrics by Service (Higher is Better)',
                                xaxis_title='Service',
                                yaxis_title='Score (0-100)',
                                barmode='group',
                                height=500
                            )
                            
                            # Display the chart if we have at least one metric to show
                            if fig.data:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("No semantic metrics data available for comparison chart.")
                        else:
                            st.info("No semantic metrics data available. Enable semantic metrics in the sidebar.")
                        
                        # Process semantic metrics results if available
                        
                        # 1. SeMaScore
                        if st.session_state.get('use_semascore', False):
                            semascore_cols = [col for col in accuracy_df_original.columns if 'semascore' in col.lower()]
                            if semascore_cols:
                                semantic_metrics_exist = True
                                st.subheader("SeMaScore by Service")
                                st.success("**Higher is better** - Semantic similarity based on sentence embeddings (0-100)")
                                semascore_means = accuracy_df_original[semascore_cols].mean()
                                if not semascore_means.empty and not semascore_means.isna().all():
                                    st.bar_chart(semascore_means)
                                else:
                                    st.info("No SeMaScore data available.")
                        
                        # 2. STS Similarity
                        if st.session_state.get('use_sts', False):
                            sts_cols = [col for col in accuracy_df_original.columns if 'sts similarity' in col.lower() or 'sts_similarity' in col.lower()]
                            if sts_cols:
                                semantic_metrics_exist = True
                                st.subheader("STS Similarity by Service")
                                st.success("**Higher is better** - Semantic Textual Similarity using cosine distance (0-100)")
                                sts_means = accuracy_df_original[sts_cols].mean()
                                if not sts_means.empty and not sts_means.isna().all():
                                    st.bar_chart(sts_means)
                                else:
                                    st.info("No STS Similarity data available.")
                        
                        # 3. BERTScore
                        if st.session_state.get('use_bertscore', False):
                            bertscore_cols = [col for col in accuracy_df_original.columns if 'bertscore' in col.lower()]
                            if bertscore_cols:
                                semantic_metrics_exist = True
                                st.subheader("BERTScore by Service")
                                st.success("**Higher is better** - Semantic similarity using contextual embeddings (0-100)")
                                bertscore_means = accuracy_df_original[bertscore_cols].mean()
                                if not bertscore_means.empty and not bertscore_means.isna().all():
                                    st.bar_chart(bertscore_means)
                                else:
                                    st.info("No BERTScore data available.")
                        
                        # 4. LLM Meaning Preservation
                        if st.session_state.get('use_llm', False):
                            llm_cols = [col for col in accuracy_df_original.columns if 'llm meaning' in col.lower() or 'meaning preservation' in col.lower()]
                            if llm_cols:
                                semantic_metrics_exist = True
                                st.subheader("LLM Meaning Preservation by Service")
                                st.success("**Higher is better** - GPT-4o mini evaluation of meaning preservation (0-100)")
                                llm_means = accuracy_df_original[llm_cols].mean()
                                if not llm_means.empty and not llm_means.isna().all():
                                    st.bar_chart(llm_means)
                                else:
                                    st.info("No LLM Meaning Preservation data available.")
                        
                        # 5. SWWER (Semantically Weighted Word Error Rate)
                        if st.session_state.get('use_swwer', False):
                            swwer_cols = [col for col in accuracy_df_original.columns if 'swwer' in col.lower()]
                            if swwer_cols:
                                semantic_metrics_exist = True
                                st.subheader("SWWER by Service")
                                st.warning("**Lower is better** - Word error rate weighted by semantic importance")
                                swwer_means = accuracy_df_original[swwer_cols].mean()
                                if not swwer_means.empty and not swwer_means.isna().all():
                                    st.bar_chart(swwer_means)
                                else:
                                    st.info("No SWWER data available.")
                          # Show a message if no semantic metrics were enabled or available
                        if not semantic_metrics_exist:
                            st.info("No semantic metrics enabled or available. You can enable semantic metrics in the sidebar.")
                    
                    # Convert DataFrame to Excel bytes - use the reorganized dataframe for download
                    excel_data = convert_df_to_excel(accuracy_df)
                    
                    # Provide download button for accuracy metrics
                    st.download_button(
                        label="Download Accuracy Metrics (Excel)",
                        data=excel_data,
                        file_name='accuracy_metrics.xlsx',
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    if not reference_files:
                        st.warning("Upload reference transcription files to see accuracy analysis.")
                    else:
                        st.info("Process files to see accuracy analysis.")
        
        with tab5:
            st.header("Overall Analysis")
            
            if reference_files and hasattr(st.session_state, 'performance_metrics') and st.session_state.performance_metrics:
                # Create DataFrame for metrics
                df_metrics = pd.DataFrame(st.session_state.performance_metrics)
                
                # Get list of services used
                service_names = set()
                for col in df_metrics.columns:
                    for service in ['AssemblyAI', 'Whisper', 'GPT-4o', 'GPT-4o Mini', 'Deepgram Nova 3', 
                                  'Google Gemini Pro 2.5', 'Google Gemini 2.0 Flash', 'Google Gemini 2.5 Flash']:
                        if service in col:
                            service_names.add(service)
                
                service_names = sorted(service_names)
                
                # Add chart library selection
                st.sidebar.subheader("Chart Options")
                chart_library = st.sidebar.radio(
                    "Select Chart Library",
                    options=["Plotly", "Matplotlib"],
                    index=0,  # Default to Plotly
                    key="chart_library"
                )
                
                # Create separate sections for AI analysis, radar charts, and heatmap
                analysis_tab, visualization_tab = st.tabs(["AI Analysis", "Visualizations (Radar & Heatmap)"])
                
                # AI Analysis Tab
                with analysis_tab:
                    # Display selected model for analysis
                    selected_model = st.session_state.get('analysis_model', 'GPT-4o')
                    st.subheader(f"Analysis by {selected_model}")
                    
                    # Get LLM analysis of the results
                    with st.spinner(f"Generating analysis with {selected_model}..."):
                        try:
                            analysis_text = generate_llm_analysis(df_metrics, service_names, selected_model)
                            st.markdown(analysis_text)
                        except Exception as e:
                            st.error(f"Error generating analysis: {str(e)}")
                            st.info("Please make sure you have provided the appropriate API key for the selected model.")
                
                # Visualizations Tab
                with visualization_tab:
                    # Create radar charts
                    st.subheader("Radar Chart Analysis")
                    st.info("""
                    Radar charts provide a visual comparison of multiple metrics across different services.
                    - Metrics are normalized so that **higher values represent better performance** for all metrics.
                    - Each service is represented by a different color.
                    """)
                      # Get metrics for radar charts (only include normalized metrics where higher is better)
                    radar_metrics = prepare_metrics_for_radar(df_metrics)
                    
                    # Create tabs for radar charts and heatmap
                    radar_tab, heatmap_tab = st.tabs(["Radar Charts", "Heat Map"])
                    
                    with radar_tab:
                        # Create individual radar charts for each service
                        st.write("### Individual Service Performance")
                        
                        # Use a single column per service instead of 3 columns
                        for i, service in enumerate(service_names):
                            st.subheader(f"{service} Performance")
                            fig = create_radar_chart(radar_metrics, [service], f"{service} Performance")
                            # Use the appropriate display method based on chart library
                            if st.session_state.get('chart_library', 'plotly') == 'plotly':
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.pyplot(fig)                            # Add a separator between charts except after the last one
                            if i < len(service_names) - 1:
                                st.markdown("---")
                        
                        # Create combined radar chart with all services
                        st.subheader("Combined Radar Chart - All Services")
                        st.info("This chart overlays all services for direct comparison. Each service uses a different color.")
                        
                        fig = create_radar_chart(radar_metrics, service_names, "All Services Comparison")
                        # Use the appropriate display method based on chart library
                        if st.session_state.get('chart_library', 'plotly') == 'plotly':
                            st.plotly_chart(fig)
                        else:
                            st.pyplot(fig)
                        
                        # Allow downloading the charts
                        st.download_button(
                            label="Download Radar Charts (PNG)",
                            data=get_combined_radar_png(radar_metrics, service_names),
                            file_name="radar_charts.png",
                            mime="image/png"
                        )
                    
                    with heatmap_tab:
                        # Add heat map analysis
                        st.subheader("Comprehensive Heat Map Analysis")
                        st.info("""
                        This heat map provides a comprehensive view of all services across all metrics in a single visualization.
                        - **Color intensity** indicates performance: Darker colors represent better performance.
                        - Metrics are normalized so that higher values (darker colors) always represent better performance.
                        - This visualization makes it easy to identify patterns and compare services across multiple dimensions.                        """)
                        # Create heat map
                        with st.spinner("Generating comprehensive heat map..."):
                            # Prepare data for heat map (using the same normalized metrics from radar charts)
                            heatmap_data = []
                            category_names = radar_metrics['categories']
                            
                            # Initialize metric and service selection in session state if not already present
                            if 'heatmap_metrics_selection' not in st.session_state:
                                st.session_state.heatmap_metrics_selection = {category: True for category in category_names}
                            
                            if 'heatmap_services_selection' not in st.session_state:
                                st.session_state.heatmap_services_selection = {service: True for service in service_names}
                            
                            # Create selection tabs for metrics and services
                            heatmap_tabs = st.tabs(["Select Metrics", "Select Services"])
                            
                            # Tab for metrics selection
                            with heatmap_tabs[0]:
                                st.subheader("Select Metrics to Display")
                                col1, col2 = st.columns(2)
                                
                                # Create checkboxes for each metric in two columns for better layout
                                for i, category in enumerate(category_names):
                                    if i < len(category_names) // 2 + len(category_names) % 2:
                                        with col1:
                                            st.session_state.heatmap_metrics_selection[category] = st.checkbox(
                                                category, 
                                                value=st.session_state.heatmap_metrics_selection.get(category, True),
                                                key=f"heatmap_metric_{category}"
                                            )
                                    else:
                                        with col2:
                                            st.session_state.heatmap_metrics_selection[category] = st.checkbox(
                                                category, 
                                                value=st.session_state.heatmap_metrics_selection.get(category, True),
                                                key=f"heatmap_metric_{category}"
                                            )
                                
                                # Button to select/deselect all metrics
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("Select All Metrics"):
                                        for category in category_names:
                                            st.session_state.heatmap_metrics_selection[category] = True
                                        st.experimental_rerun()
                                with col2:
                                    if st.button("Deselect All Metrics"):
                                        for category in category_names:
                                            st.session_state.heatmap_metrics_selection[category] = False
                                        st.experimental_rerun()
                              # Tab for services selection
                            with heatmap_tabs[1]:
                                st.subheader("Select Services to Display")
                                col1, col2 = st.columns(2)
                                
                                # Create checkboxes for each service in two columns for better layout
                                for i, service in enumerate(service_names):
                                    if i < len(service_names) // 2 + len(service_names) % 2:
                                        with col1:
                                            st.session_state.heatmap_services_selection[service] = st.checkbox(
                                                service,
                                                value=st.session_state.heatmap_services_selection.get(service, True),
                                                key=f"heatmap_service_{service}"
                                            )
                                    else:
                                        with col2:
                                            st.session_state.heatmap_services_selection[service] = st.checkbox(
                                                service,
                                                value=st.session_state.heatmap_services_selection.get(service, True),
                                                key=f"heatmap_service_{service}"
                                            )
                                
                                # Button to select/deselect all services
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("Select All Services"):
                                        for service in service_names:
                                            st.session_state.heatmap_services_selection[service] = True
                                        st.experimental_rerun()
                                with col2:
                                    if st.button("Deselect All Services"):
                                        for service in service_names:
                                            st.session_state.heatmap_services_selection[service] = False
                                        st.experimental_rerun()                            # Add a reset button to restore default selections
                            if st.button("Reset All Selections"):
                                st.session_state.heatmap_metrics_selection = {category: True for category in category_names}
                                st.session_state.heatmap_services_selection = {service: True for service in service_names}
                                st.experimental_rerun()
                            
                            # Get the selected metrics and services
                            selected_metrics = [category for category in category_names 
                                               if st.session_state.heatmap_metrics_selection.get(category, True)]
                            
                            selected_services = [service for service in service_names
                                                if st.session_state.heatmap_services_selection.get(service, True)]
                            
                            if not selected_metrics:
                                st.warning("Please select at least one metric to display in the heatmap.")
                                return
                            
                            if not selected_services:
                                st.warning("Please select at least one service to display in the heatmap.")
                                return
                                
                            # Display selection summary
                            st.caption(f"Displaying {len(selected_services)}/{len(service_names)} services and {len(selected_metrics)}/{len(category_names)} metrics")                            # Convert the radar metrics to a format suitable for heatmap (only selected metrics and services)
                            for service_name in service_names:
                                if service_name in selected_services:  # Only include selected services
                                    service_values = radar_metrics['services'].get(service_name, [])
                                    if service_values:
                                        for i, category in enumerate(category_names):
                                            if category in selected_metrics:  # Only include selected metrics
                                                heatmap_data.append({
                                                    'Service': service_name,
                                                    'Metric': category,
                                                    'Score': service_values[i]  # Already normalized to 0-100
                                                })
                            
                            if heatmap_data:
                                # Create a DataFrame for the heatmap
                                heatmap_df = pd.DataFrame(heatmap_data)
                                
                                # Create the pivot table for the heatmap
                                heatmap_pivot = heatmap_df.pivot(index='Service', columns='Metric', values='Score')
                                
                                # Create the heatmap using Plotly
                                fig = go.Figure(data=go.Heatmap(
                                    z=heatmap_pivot.values,
                                    x=heatmap_pivot.columns,
                                    y=heatmap_pivot.index,
                                    colorscale='Blues',  # Using Blues colorscale where darker blue means better performance
                                    colorbar=dict(title='Score'),
                                    hoverongaps=False,
                                    text=[[f"{val:.1f}" for val in row] for row in heatmap_pivot.values],
                                    hovertemplate='Service: %{y}<br>Metric: %{x}<br>Score: %{z:.1f}<extra></extra>'
                                ))
                                
                                # Update layout for better readability
                                fig.update_layout(
                                    title=f"Service Performance Heat Map ({len(selected_services)} Services, {len(selected_metrics)} Metrics)",
                                    xaxis_title="Metrics",
                                    yaxis_title="Services",
                                    height=500,
                                    width=800
                                )
                                
                                # Display the heatmap
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Add some insights from the heatmap
                                st.subheader("Heat Map Insights")
                                st.write("""
                                The heat map above provides a view of service performance across the selected metrics.
                                Here are some ways to interpret this visualization:
                                
                                - **Dark blue cells** indicate areas where a service excels
                                - **Light blue cells** highlight potential weaknesses
                                - **Look for rows** with consistently dark colors to identify top performing services overall
                                - **Look for columns** to identify which services excel at specific metrics
                                - **Compare patterns** between accuracy metrics vs. cost and processing speed
                                
                                **Pro Tip:** Use the selection tabs above to customize which metrics and services appear in the heatmap. This can help you focus on specific aspects you're most interested in.
                                """)
                
            else:
                if not reference_files:
                    st.warning("Upload reference transcription files to see overall analysis.")
                else:
                    st.info("Process files to see overall analysis.")
    elif uploaded_files:
        st.warning("Please enter and save API keys in the sidebar before processing files.")

if __name__ == "__main__":

    main()
