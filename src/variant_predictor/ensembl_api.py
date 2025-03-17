"""
Ensembl Variant Effect Predictor (VEP) API module for EvoBeevos+

This module provides functions to interact with the Ensembl VEP API
to predict the effect of genetic variants.
"""

import requests
import sys
import json
from typing import Dict, List, Any, Optional, Union

def query_ensembl_vep(
    chromosome: str, 
    position: int, 
    ref_allele: str, 
    alt_allele: str, 
    assembly: str = "GRCh38"
) -> Dict[str, Any]:
    """
    Query the Ensembl Variant Effect Predictor (VEP) API for a specific variant.
    
    Args:
        chromosome: Chromosome number or name (e.g., '17', 'X')
        position: Genomic position
        ref_allele: Reference allele
        alt_allele: Alternate allele
        assembly: Genome assembly ('GRCh38' or 'GRCh37')
        
    Returns:
        Dictionary with VEP analysis results
    """
    # Base URL for Ensembl REST API
    server = "https://rest.ensembl.org"
    
    # Determine the correct API endpoint based on assembly
    if assembly == "GRCh37":
        server = "https://grch37.rest.ensembl.org"
    
    # Construct endpoint URL with parameters
    ext = f"/vep/human/region/{chromosome}:{position}-{position}:1/{alt_allele}?"
    
    # Additional parameters for a more comprehensive analysis
    params = {
        "variant_class": 1,           # Include variant class
        "sift": 1,                    # Include SIFT predictions
        "polyphen": 1,                # Include PolyPhen predictions
        "ccds": 1,                    # Include CCDS identifiers
        "hgvs": 1,                    # Include HGVS nomenclature
        "canonical": 1,               # Identify canonical transcripts
        "protein": 1,                 # Include protein sequence information
        "numbers": 1,                 # Include exon/intron numbers
        "domains": 1,                 # Include domain information
        "af": 1,                      # Include allele frequencies
        "af_1kg": 1,                  # Include 1000 Genomes frequencies
        "af_gnomad": 1,               # Include gnomAD frequencies
        "appris": 1,                  # Include APPRIS annotations
        "mane": 1,                    # Include MANE annotations
        "biotype": 1,                 # Include biotype
        "check_existing": 1,          # Check for existing variants
        "transcript_version": 1,      # Include transcript versions
        "tsl": 1,                     # Include transcript support level
        "variant_synonyms": 1,        # Include variant synonyms
    }
    
    # Convert parameters to URL query string
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    if query_string:
        ext += query_string
    
    # Make GET request to the API with appropriate headers
    try:
        response = requests.get(
            server + ext, 
            headers={"Content-Type": "application/json"}
        )
        
        # Check if the request was successful
        if not response.ok:
            return {
                "error": f"API request failed with status {response.status_code}: {response.reason}",
                "details": response.text
            }
            
        # Parse the JSON response
        data = response.json()
        
        # Process the data into a more usable format
        if isinstance(data, list) and len(data) > 0:
            # Extract and process the first result (usually there's only one for a specific variant)
            result = process_vep_response(data[0])
            return result
        else:
            return {"error": "No results returned from Ensembl VEP"}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"Request exception: {str(e)}"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON response from Ensembl VEP"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def process_vep_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the Ensembl VEP response into a more usable format.
    
    Args:
        response_data: Raw VEP response dictionary
        
    Returns:
        Processed VEP data dictionary
    """
    processed_data = {
        "variant_id": response_data.get("id"),
        "assembly_name": response_data.get("assembly_name"),
        "seq_region_name": response_data.get("seq_region_name"),
        "start": response_data.get("start"),
        "end": response_data.get("end"),
        "strand": response_data.get("strand"),
        "allele_string": response_data.get("allele_string"),
        "variant_class": response_data.get("variant_class"),
        "most_severe_consequence": response_data.get("most_severe_consequence"),
        "colocated_variants": process_colocated_variants(response_data.get("colocated_variants", [])),
        "transcript_consequences": process_transcript_consequences(response_data.get("transcript_consequences", [])),
        "regulatory_feature_consequences": response_data.get("regulatory_feature_consequences", []),
        "input": response_data.get("input")
    }
    
    return processed_data

def process_colocated_variants(colocated_variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process colocated variants data from VEP response.
    
    Args:
        colocated_variants: List of colocated variant dictionaries
        
    Returns:
        Processed list of colocated variant dictionaries
    """
    processed_variants = []
    
    for variant in colocated_variants:
        processed_variant = {
            "id": variant.get("id"),
            "start": variant.get("start"),
            "end": variant.get("end"),
            "allele_string": variant.get("allele_string"),
            "strand": variant.get("strand"),
            "somatic": variant.get("somatic", False),
            "frequencies": {
                "minor_allele": variant.get("minor_allele"),
                "minor_allele_freq": variant.get("minor_allele_freq"),
                "afr": variant.get("afr_maf"),
                "amr": variant.get("amr_maf"),
                "eas": variant.get("eas_maf"),
                "eur": variant.get("eur_maf"),
                "sas": variant.get("sas_maf"),
                "gnomad": variant.get("gnomad_maf"),
                "gnomad_afr": variant.get("gnomad_afr_maf"),
                "gnomad_amr": variant.get("gnomad_amr_maf"),
                "gnomad_asj": variant.get("gnomad_asj_maf"),
                "gnomad_eas": variant.get("gnomad_eas_maf"),
                "gnomad_fin": variant.get("gnomad_fin_maf"),
                "gnomad_nfe": variant.get("gnomad_nfe_maf"),
                "gnomad_oth": variant.get("gnomad_oth_maf"),
                "gnomad_sas": variant.get("gnomad_sas_maf")
            },
            "clin_sig": variant.get("clin_sig", []),
            "phenotype_or_disease": variant.get("phenotype_or_disease", False),
            "pubmed": variant.get("pubmed", []),
            "var_synonyms": variant.get("var_synonyms", {})
        }
        processed_variants.append(processed_variant)
    
    return processed_variants

def process_transcript_consequences(transcript_consequences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process transcript consequences data from VEP response.
    
    Args:
        transcript_consequences: List of transcript consequence dictionaries
        
    Returns:
        Processed list of transcript consequence dictionaries
    """
    processed_consequences = []
    
    for consequence in transcript_consequences:
        processed_consequence = {
            "transcript_id": consequence.get("transcript_id"),
            "gene_id": consequence.get("gene_id"),
            "gene_symbol": consequence.get("gene_symbol"),
            "gene_symbol_source": consequence.get("gene_symbol_source"),
            "consequence_terms": consequence.get("consequence_terms", []),
            "impact": consequence.get("impact"),
            "canonical": consequence.get("canonical") == 1,
            "biotype": consequence.get("biotype"),
            "mane": consequence.get("mane"),
            "ccds": consequence.get("ccds"),
            "protein_id": consequence.get("protein_id"),
            "protein_start": consequence.get("protein_start"),
            "protein_end": consequence.get("protein_end"),
            "amino_acids": consequence.get("amino_acids"),
            "codons": consequence.get("codons"),
            "strand": consequence.get("strand"),
            "sift_prediction": consequence.get("sift_prediction"),
            "sift_score": consequence.get("sift_score"),
            "polyphen_prediction": consequence.get("polyphen_prediction"),
            "polyphen_score": consequence.get("polyphen_score"),
            "appris": consequence.get("appris"),
            "tsl": consequence.get("tsl"),
            "hgvsc": consequence.get("hgvsc"),
            "hgvsp": consequence.get("hgvsp"),
            "domains": consequence.get("domains", [])
        }
        processed_consequences.append(processed_consequence)
    
    return processed_consequences

def get_canonical_transcript(vep_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract the canonical transcript from VEP data if available.
    
    Args:
        vep_data: Processed VEP data dictionary
        
    Returns:
        Dictionary with canonical transcript data or None if not found
    """
    transcript_consequences = vep_data.get("transcript_consequences", [])
    
    # First, try to find MANE Select transcript (highest priority)
    for transcript in transcript_consequences:
        mane = transcript.get("mane", "")
        if mane and "MANE_SELECT" in mane:
            return transcript
    
    # Next, look for canonical transcript
    for transcript in transcript_consequences:
        if transcript.get("canonical", False):
            return transcript
    
    # If no canonical found, return the first transcript or None
    return transcript_consequences[0] if transcript_consequences else None

def get_vep_summary(vep_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary of the most important VEP data.
    
    Args:
        vep_data: Processed VEP data dictionary
        
    Returns:
        Dictionary with summarized VEP data
    """
    # Get the canonical transcript
    canonical_transcript = get_canonical_transcript(vep_data)
    
    # Extract clinical significance from colocated variants
    clinical_significance = []
    for variant in vep_data.get("colocated_variants", []):
        if variant.get("clin_sig"):
            clinical_significance.extend(variant.get("clin_sig", []))
    
    # Build the summary
    summary = {
        "variant": vep_data.get("allele_string", ""),
        "variant_class": vep_data.get("variant_class", ""),
        "most_severe_consequence": vep_data.get("most_severe_consequence", ""),
        "clinical_significance": list(set(clinical_significance)),  # Remove duplicates
        "gene": None,
        "transcript": None,
        "protein_change": None,
        "impact": None,
        "prediction_scores": {}
    }
    
    # Add transcript-specific information if available
    if canonical_transcript:
        summary["gene"] = {
            "id": canonical_transcript.get("gene_id"),
            "symbol": canonical_transcript.get("gene_symbol"),
            "biotype": canonical_transcript.get("biotype")
        }
        
        summary["transcript"] = {
            "id": canonical_transcript.get("transcript_id"),
            "is_canonical": canonical_transcript.get("canonical", False),
            "mane": canonical_transcript.get("mane")
        }
        
        summary["protein_change"] = {
            "amino_acids": canonical_transcript.get("amino_acids"),
            "codons": canonical_transcript.get("codons"),
            "position": canonical_transcript.get("protein_start"),
            "hgvsc": canonical_transcript.get("hgvsc"),
            "hgvsp": canonical_transcript.get("hgvsp")
        }
        
        summary["impact"] = canonical_transcript.get("impact")
        
        # Add prediction scores if available
        if canonical_transcript.get("sift_prediction"):
            summary["prediction_scores"]["sift"] = {
                "prediction": canonical_transcript.get("sift_prediction"),
                "score": canonical_transcript.get("sift_score")
            }
            
        if canonical_transcript.get("polyphen_prediction"):
            summary["prediction_scores"]["polyphen"] = {
                "prediction": canonical_transcript.get("polyphen_prediction"),
                "score": canonical_transcript.get("polyphen_score")
            }
    
    return summary 