"""
Evo 2 AI Model Integration for Variant Effect Prediction

This module provides functions to integrate with the Evo 2 AI model
for predicting the effect of genetic variants on protein function.
"""

import os
import requests
import json
import time
from typing import Dict, Any, Optional, Tuple, List, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_WINDOW_SIZE = 2048
DEFAULT_API_URL = "https://api.togethercomputer.com/models/evo-2-40b"

def get_reference_sequence(
    chromosome: str,
    position: int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    reference_genome: str = "GRCh38"
) -> Optional[str]:
    """
    Retrieve reference sequence from Ensembl REST API.
    
    Args:
        chromosome: Chromosome number or name (e.g., '17', 'X')
        position: Genomic position
        window_size: Size of the sequence window (must be even)
        reference_genome: Reference genome build ('GRCh38' or 'GRCh37')
        
    Returns:
        Reference sequence as string or None if retrieval fails
    """
    # Make sure window_size is even
    if window_size % 2 != 0:
        window_size += 1
    
    # Calculate region start and end positions
    half_window = window_size // 2
    region_start = max(1, position - half_window)
    region_end = position + half_window
    
    # Determine Ensembl API base URL based on reference genome
    if reference_genome == "GRCh37":
        server = "https://grch37.rest.ensembl.org"
    else:
        server = "https://rest.ensembl.org"
    
    # Construct the API endpoint URL
    ext = f"/sequence/region/human/{chromosome}:{region_start}..{region_end}:1?"
    
    try:
        response = requests.get(
            server + ext,
            headers={"Content-Type": "application/json"}
        )
        
        if not response.ok:
            logger.error(f"Failed to retrieve sequence: {response.status_code} - {response.reason}")
            return None
        
        data = response.json()
        
        if isinstance(data, dict) and "seq" in data:
            return data["seq"]
        else:
            logger.error("Unexpected response format from Ensembl API")
            return None
    
    except Exception as e:
        logger.error(f"Error retrieving reference sequence: {str(e)}")
        return None

def create_variant_sequence(
    reference_sequence: str,
    position: int,
    ref_allele: str,
    alt_allele: str,
    window_size: int = DEFAULT_WINDOW_SIZE
) -> Optional[str]:
    """
    Create a variant sequence by introducing the variant into the reference sequence.
    
    Args:
        reference_sequence: Reference DNA sequence
        position: Genomic position within the sequence
        ref_allele: Reference allele
        alt_allele: Alternate allele
        window_size: Size of the sequence window
        
    Returns:
        Variant sequence as string or None if creation fails
    """
    try:
        # Calculate relative position within the reference sequence
        half_window = window_size // 2
        rel_position = half_window
        
        # Validate that reference allele matches the sequence
        seq_ref = reference_sequence[rel_position:rel_position + len(ref_allele)]
        if seq_ref != ref_allele:
            logger.error(f"Reference allele mismatch: expected {ref_allele}, found {seq_ref}")
            return None
        
        # Create variant sequence by replacing the reference allele with the alternate allele
        variant_sequence = (
            reference_sequence[:rel_position] +
            alt_allele +
            reference_sequence[rel_position + len(ref_allele):]
        )
        
        return variant_sequence
    
    except Exception as e:
        logger.error(f"Error creating variant sequence: {str(e)}")
        return None

def call_evo_api(
    sequence: str,
    api_key: Optional[str] = None,
    api_url: str = DEFAULT_API_URL
) -> Dict[str, Any]:
    """
    Call the Evo 2 API to get a prediction for a DNA sequence.
    
    Args:
        sequence: DNA sequence to analyze
        api_key: API key for Evo 2 (from environment or config)
        api_url: URL of the Evo 2 API endpoint
        
    Returns:
        Dictionary with Evo 2 API response or error information
    """
    # If no API key is provided, try to get it from environment variable
    if api_key is None:
        api_key = os.environ.get("EVO_API_KEY")
        
    if not api_key:
        return {"error": "No API key provided for Evo 2 model"}
    
    # Prepare the API request payload
    payload = {
        "sequence": sequence,
        "output_layers": ["embedding_layer"],
        "return_sequences": False,
        "max_generated_length": 0  # We don't need generated sequences, just the embeddings
    }
    
    # API headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        # Make the API request
        response = requests.post(api_url, json=payload, headers=headers)
        
        if not response.ok:
            return {
                "error": f"API request failed with status {response.status_code}",
                "details": response.text
            }
        
        return response.json()
    
    except Exception as e:
        return {"error": f"Error calling Evo 2 API: {str(e)}"}

def compute_delta_score(
    ref_embeddings: List[float],
    alt_embeddings: List[float]
) -> float:
    """
    Compute the delta score between reference and alternate embeddings.
    
    Args:
        ref_embeddings: Embeddings from the reference sequence
        alt_embeddings: Embeddings from the alternate (variant) sequence
        
    Returns:
        Delta score as a float
    """
    # Simple L2 distance for now, but could be improved with more sophisticated metrics
    import numpy as np
    
    ref_array = np.array(ref_embeddings)
    alt_array = np.array(alt_embeddings)
    
    # L2 distance (Euclidean)
    delta = np.linalg.norm(ref_array - alt_array)
    
    return float(delta)

def interpret_delta_score(delta_score: float) -> Dict[str, Any]:
    """
    Interpret the delta score to predict variant effect.
    
    Args:
        delta_score: Delta score between reference and variant sequences
        
    Returns:
        Dictionary with interpretation results
    """
    # These thresholds are placeholders and should be calibrated with actual data
    if delta_score < 0.1:
        effect = "Benign"
        confidence = 0.9
    elif delta_score < 0.3:
        effect = "Likely Benign"
        confidence = 0.7
    elif delta_score < 0.5:
        effect = "Variant of Uncertain Significance"
        confidence = 0.5
    elif delta_score < 0.8:
        effect = "Likely Pathogenic"
        confidence = 0.7
    else:
        effect = "Pathogenic"
        confidence = 0.9
    
    return {
        "effect": effect,
        "confidence": confidence,
        "delta_score": delta_score
    }

def predict_variant_effect(
    chromosome: str,
    position: int,
    ref_allele: str,
    alt_allele: str,
    api_key: Optional[str] = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    reference_genome: str = "GRCh38"
) -> Dict[str, Any]:
    """
    Predict the effect of a genetic variant using Evo 2 AI model.
    
    Args:
        chromosome: Chromosome number or name (e.g., '17', 'X')
        position: Genomic position
        ref_allele: Reference allele
        alt_allele: Alternate allele
        api_key: API key for Evo 2 (optional)
        window_size: Size of the sequence window
        reference_genome: Reference genome build ('GRCh38' or 'GRCh37')
        
    Returns:
        Dictionary with prediction results or error information
    """
    # Step 1: Get reference sequence
    ref_sequence = get_reference_sequence(
        chromosome=chromosome,
        position=position,
        window_size=window_size,
        reference_genome=reference_genome
    )
    
    if not ref_sequence:
        return {"error": "Failed to retrieve reference sequence"}
    
    # Step 2: Create variant sequence
    var_sequence = create_variant_sequence(
        reference_sequence=ref_sequence,
        position=position,
        ref_allele=ref_allele,
        alt_allele=alt_allele,
        window_size=window_size
    )
    
    if not var_sequence:
        return {"error": "Failed to create variant sequence"}
    
    # Step 3: Call Evo 2 API for reference sequence
    ref_result = call_evo_api(ref_sequence, api_key)
    if "error" in ref_result:
        return {"error": f"Error calling Evo 2 API for reference sequence: {ref_result['error']}"}
    
    # Step 4: Call Evo 2 API for variant sequence
    var_result = call_evo_api(var_sequence, api_key)
    if "error" in var_result:
        return {"error": f"Error calling Evo 2 API for variant sequence: {var_result['error']}"}
    
    # Step 5: Extract embeddings
    try:
        ref_embeddings = ref_result.get("embedding_layer", [])
        var_embeddings = var_result.get("embedding_layer", [])
        
        if not ref_embeddings or not var_embeddings:
            return {"error": "Failed to extract embeddings from API response"}
        
        # Step 6: Compute delta score
        delta_score = compute_delta_score(ref_embeddings, var_embeddings)
        
        # Step 7: Interpret delta score
        interpretation = interpret_delta_score(delta_score)
        
        # Step 8: Return comprehensive result
        return {
            "variant": f"{chromosome}:{position}{ref_allele}>{alt_allele}",
            "model": "Evo 2",
            "delta_score": delta_score,
            "effect": interpretation["effect"],
            "confidence": interpretation["confidence"],
            "reference_genome": reference_genome,
            "window_size": window_size
        }
    
    except Exception as e:
        return {"error": f"Error processing Evo 2 predictions: {str(e)}"}

def predict_variant_effect_mock(
    chromosome: str,
    position: int,
    ref_allele: str,
    alt_allele: str,
    api_key: Optional[str] = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    reference_genome: str = "GRCh38"
) -> Dict[str, Any]:
    """
    Generate mock predictions for testing when API access is not available.
    
    Args:
        chromosome: Chromosome number or name (e.g., '17', 'X')
        position: Genomic position
        ref_allele: Reference allele
        alt_allele: Alternate allele
        api_key: API key (not used in mock)
        window_size: Size of the sequence window (not used in mock)
        reference_genome: Reference genome build (not used in mock)
        
    Returns:
        Dictionary with mock prediction results
    """
    import random
    
    # Generate a pseudo-random but deterministic delta score based on input parameters
    # This ensures consistent results for the same input
    import hashlib
    input_string = f"{chromosome}:{position}{ref_allele}>{alt_allele}"
    hash_value = hashlib.md5(input_string.encode()).hexdigest()
    # Convert first 8 chars of hash to integer and scale to 0-1 range
    random_seed = int(hash_value[:8], 16) / (16**8)
    
    # Seed the random generator for deterministic results
    random.seed(random_seed)
    
    # Generate a delta score (slightly biased toward pathogenic for known problematic chromosomes)
    base_delta = random.uniform(0.1, 0.9)
    if chromosome in ["17", "13", "X"] and random.random() < 0.7:
        # Bias toward pathogenic for BRCA1/2 and other known disease chromosomes
        delta_score = min(0.9, base_delta * 1.5)
    else:
        delta_score = base_delta
    
    # Interpret the delta score
    interpretation = interpret_delta_score(delta_score)
    
    # Add a small random delay to simulate API call
    time.sleep(random.uniform(0.5, 1.5))
    
    return {
        "variant": f"{chromosome}:{position}{ref_allele}>{alt_allele}",
        "model": "Evo 2 (Mock)",
        "delta_score": delta_score,
        "effect": interpretation["effect"],
        "confidence": interpretation["confidence"],
        "reference_genome": reference_genome,
        "window_size": window_size,
        "mock": True
    } 