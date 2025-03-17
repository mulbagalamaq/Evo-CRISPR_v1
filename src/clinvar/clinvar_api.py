"""
ClinVar API module for EvoBeevos+

This module provides functions to interact with the ClinVar database 
to retrieve genetic variant information.
"""

import requests
import urllib.parse
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from bs4 import BeautifulSoup
import re
import json
from collections import Counter

def get_clinvar_data(
    chromosome: str, 
    position: int, 
    ref_allele: str, 
    alt_allele: str, 
    assembly: str = "GRCh38"
) -> Dict[str, Any]:
    """
    Fetch ClinVar data for a specific genetic variant.
    
    Args:
        chromosome: Chromosome number or name (e.g., '17', 'X')
        position: Genomic position
        ref_allele: Reference allele
        alt_allele: Alternate allele
        assembly: Genome assembly ('GRCh38' or 'GRCh37')
    
    Returns:
        Dictionary containing ClinVar variant data
    """
    # Construct the query in UCSC format
    ucsc_coords = f"chr{chromosome}:{position}{ref_allele}>{alt_allele}"
    
    # Base URL for ClinVar API
    base_url = "https://www.ncbi.nlm.nih.gov/clinvar/variation/search/"
    
    # Construct the query for ClinVar
    query = f"({chromosome}[CHR] AND {position}[CPOS] AND {ref_allele}>{alt_allele})"
    
    # URL encode the query
    encoded_query = urllib.parse.quote(query, safe="()[]:>")
    
    # Determine assembly ID based on specified assembly
    assembly_id = "GCF_000001405.38" if assembly == "GRCh38" else "GCF_000001405.25"
    
    # Construct the full URL
    full_url = f"{base_url}?term={encoded_query}&assembly={assembly_id}"
    
    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.ncbi.nlm.nih.gov/clinvar/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(full_url, headers=headers)
        response.raise_for_status()  # Raise an error for HTTP issues
        
        # Parse the JSON response
        data = response.json()
        
        # Process the data into a more usable format
        processed_data = {
            "query": data.get("query", {}),
            "variants": process_variants(data.get("vars", [])),
            "genes": process_genes(data.get("genes", {})),
            "chromosome_info": data.get("chr_info", {})
        }
        
        return processed_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching ClinVar data: {e}")
        return {"error": str(e)}
    except json.JSONDecodeError:
        print("Error decoding JSON response from ClinVar")
        return {"error": "Invalid response from ClinVar"}

def process_variants(variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process ClinVar variant data into a more usable format.
    
    Args:
        variants: List of variant dictionaries from ClinVar
        
    Returns:
        Processed list of variant dictionaries
    """
    processed_variants = []
    
    for variant in variants:
        processed_variant = {
            "id": variant.get("id"),
            "clinical_significance": variant.get("ci"),
            "review_status": variant.get("revstat"),
            "description": variant.get("desc"),
            "locations": variant.get("locs", []),
            "allele_id": extract_allele_id(variant)
        }
        processed_variants.append(processed_variant)
    
    return processed_variants

def process_genes(genes: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Process ClinVar gene data into a more usable format.
    
    Args:
        genes: Dictionary of gene information from ClinVar
        
    Returns:
        Processed dictionary of gene information
    """
    processed_genes = {}
    
    for gene_symbol, gene_data in genes.items():
        processed_genes[gene_symbol] = {
            "id": gene_data.get("id"),
            "locations": {
                assembly: {
                    "strand": location.get("strand"),
                    "start": location.get("from"),
                    "end": location.get("to"),
                    "transcripts": location.get("mrnas", [])
                }
                for assembly, location in gene_data.items() 
                if assembly.startswith("GCF_")
            }
        }
    
    return processed_genes

def extract_allele_id(variant: Dict[str, Any]) -> Optional[str]:
    """
    Extract allele ID from variant data if available.
    
    Args:
        variant: Variant dictionary from ClinVar
        
    Returns:
        Allele ID if available, None otherwise
    """
    # Check if the variant has an "info" field that might contain allele ID
    if "info" in variant and isinstance(variant["info"], str):
        match = re.search(r"ALLELEID=(\d+)", variant["info"])
        if match:
            return match.group(1)
    return None

def parse_clinvar_conditions(variants: List[Dict[str, Any]]) -> List[str]:
    """
    Extract and rank conditions associated with variants in ClinVar.
    
    Args:
        variants: List of processed variant dictionaries
        
    Returns:
        List of top conditions sorted by frequency
    """
    all_conditions = []
    
    for variant in variants:
        # Skip variants without clinical significance or with uncertain significance
        if variant.get("clinical_significance") in ["Uncertain significance", None]:
            continue
            
        # Extract conditions from the description if available
        description = variant.get("description", "")
        if "|" in description:
            conditions = [cond.strip() for cond in description.split("|")]
            all_conditions.extend(conditions)
        elif description and description != "not provided" and description != "not specified":
            all_conditions.append(description)
    
    # Count frequency of each condition
    condition_counts = Counter(all_conditions)
    
    # Get top 5 conditions, or all if fewer than 5
    top_conditions = [cond for cond, _ in condition_counts.most_common(5)]
    
    return top_conditions

def fetch_clinvar_html(ucsc_coords: str) -> Optional[BeautifulSoup]:
    """
    Fetch HTML content from ClinVar for a given UCSC coordinate.
    
    Args:
        ucsc_coords: Genomic coordinates in UCSC format (e.g., 'chr17:41276133T>G')
        
    Returns:
        BeautifulSoup object of the parsed HTML or None if request fails
    """
    encoded_coords = urllib.parse.quote(ucsc_coords)
    url = f"https://www.ncbi.nlm.nih.gov/clinvar/?term={encoded_coords}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return BeautifulSoup(response.text, 'html.parser')
        else:
            print(f"Failed to retrieve HTML. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching HTML: {e}")
        return None

def extract_clinvar_classification_counts(html_soup: BeautifulSoup) -> Dict[str, int]:
    """
    Extract classification counts from ClinVar HTML.
    
    Args:
        html_soup: BeautifulSoup object of the ClinVar HTML
        
    Returns:
        Dictionary of classification counts
    """
    classification_counts = {
        "Pathogenic": 0,
        "Likely pathogenic": 0,
        "Uncertain significance": 0,
        "Likely benign": 0,
        "Benign": 0
    }
    
    # Try to find the classification table
    table = html_soup.find('table', {'class': 'clinical-significance-summary'})
    if table:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                classification = cells[0].get_text(strip=True)
                count_text = cells[1].get_text(strip=True)
                count = int(re.search(r'\d+', count_text).group()) if re.search(r'\d+', count_text) else 0
                
                if classification in classification_counts:
                    classification_counts[classification] = count
    
    return classification_counts 