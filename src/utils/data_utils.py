"""
Data Utilities Module for GenetiXplorer

This module provides functions for data transformation, validation,
and file management in the GenetiXplorer platform.
"""

import re
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import uuid
import logging
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_hgvs_notation(hgvs: str) -> Dict[str, Any]:
    """
    Parse HGVS notation into chromosome, position, ref, and alt alleles.
    
    Args:
        hgvs: HGVS notation string
        
    Returns:
        Dictionary with parsed chromosome, position, ref_allele, alt_allele
    """
    # Handle genomic HGVS (e.g., NC_000001.11:g.12345A>G)
    match = re.match(r'NC_0*(\d+)\.?\d*:g\.(\d+)([ACGT]+)>([ACGT]+)', hgvs)
    if match:
        chromosome = match.group(1)
        if chromosome == "23":
            chromosome = "X"
        elif chromosome == "24":
            chromosome = "Y"
        
        return {
            "chromosome": f"chr{chromosome}",
            "position": int(match.group(2)),
            "ref_allele": match.group(3),
            "alt_allele": match.group(4)
        }
    
    # Handle short genomic format (e.g., chr1:g.12345A>G)
    match = re.match(r'(chr\w+):g\.(\d+)([ACGT]+)>([ACGT]+)', hgvs)
    if match:
        return {
            "chromosome": match.group(1),
            "position": int(match.group(2)),
            "ref_allele": match.group(3),
            "alt_allele": match.group(4)
        }
    
    raise ValueError(f"Unsupported HGVS notation: {hgvs}")

def validate_variant_input(
    chromosome: str = "",
    position: Optional[int] = None,
    ref_allele: str = "",
    alt_allele: str = "",
    hgvs: str = "",
    variant_id: str = ""
) -> Dict[str, Any]:
    """
    Validate and normalize variant input from various formats.
    
    Args:
        chromosome: Chromosome name/number
        position: Genomic position
        ref_allele: Reference allele
        alt_allele: Alternate allele
        hgvs: HGVS notation
        variant_id: Variant identifier string
        
    Returns:
        Dictionary with normalized variant information
    """
    result = {}
    
    # Check if variant_id is provided
    if variant_id:
        try:
            result = parse_variant_identifier(variant_id)
            return result
        except ValueError:
            logger.warning(f"Could not parse variant_id: {variant_id}")
    
    # Check if chr, pos, ref, alt are provided
    if chromosome and position is not None and ref_allele and alt_allele:
        # Normalize chromosome
        chrom = str(chromosome)
        if not chrom.startswith('chr'):
            chrom = f"chr{chrom}"
        
        result = {
            "chromosome": chrom,
            "position": position,
            "ref_allele": ref_allele,
            "alt_allele": alt_allele
        }
        return result
    
    # Check if HGVS notation is provided
    if hgvs:
        try:
            result = parse_hgvs_notation(hgvs)
            return result
        except ValueError:
            logger.warning(f"Could not parse HGVS notation: {hgvs}")
    
    # If we reach here, we couldn't validate the input
    valid_inputs = []
    if chromosome:
        valid_inputs.append("chromosome")
    if position is not None:
        valid_inputs.append("position")
    if ref_allele:
        valid_inputs.append("ref_allele")
    if alt_allele:
        valid_inputs.append("alt_allele")
    if hgvs:
        valid_inputs.append("hgvs")
    if variant_id:
        valid_inputs.append("variant_id")
    
    if valid_inputs:
        error_msg = f"Incomplete variant information. Provided: {', '.join(valid_inputs)}"
    else:
        error_msg = "No variant information provided"
    
    raise ValueError(error_msg)

def parse_vcf_line(line: str) -> Dict[str, Any]:
    """
    Parse a single line from a VCF file.
    
    Args:
        line: VCF line string
        
    Returns:
        Dictionary with parsed variant information
    """
    if line.startswith('#'):
        return {}  # Skip header lines
    
    fields = line.strip().split('\t')
    if len(fields) < 5:
        return {}  # Skip malformed lines
    
    # Extract basic variant information
    chrom = fields[0]
    pos = int(fields[1])
    variant_id = fields[2] if fields[2] != '.' else ''
    ref = fields[3]
    alt = fields[4]
    
    # Convert multiple alt alleles into separate variants
    if ',' in alt:
        alt_alleles = alt.split(',')
        variants = []
        for a in alt_alleles:
            variant = {
                "chromosome": chrom,
                "position": pos,
                "ref_allele": ref,
                "alt_allele": a
            }
            if variant_id:
                variant["id"] = variant_id
            variants.append(variant)
        return {"variants": variants}
    else:
        # Return single variant
        variant = {
            "chromosome": chrom,
            "position": pos,
            "ref_allele": ref,
            "alt_allele": alt
        }
        if variant_id:
            variant["id"] = variant_id
        return variant

def parse_vcf_content(content: str) -> List[Dict[str, Any]]:
    """
    Parse contents of a VCF file.
    
    Args:
        content: VCF file content as string
        
    Returns:
        List of parsed variant dictionaries
    """
    variants = []
    for line in content.splitlines():
        variant_data = parse_vcf_line(line)
        if not variant_data:
            continue
            
        if "variants" in variant_data:
            # Handle multiple alt alleles
            variants.extend(variant_data["variants"])
        else:
            # Single variant
            variants.append(variant_data)
    
    return variants

def is_protein_coding_variant(variant_info: Dict[str, Any]) -> bool:
    """
    Check if a variant is in a protein-coding region.
    
    Args:
        variant_info: Variant information dictionary
        
    Returns:
        True if the variant is in a protein-coding region, False otherwise
    """
    # Check for VEP consequences that imply a coding variant
    coding_consequences = [
        "missense_variant",
        "synonymous_variant",
        "stop_gained",
        "stop_lost",
        "start_lost",
        "frameshift_variant",
        "inframe_insertion",
        "inframe_deletion"
    ]
    
    # Check in VEP data
    if "vep" in variant_info:
        for transcript in variant_info["vep"].get("transcript_consequences", []):
            consequences = transcript.get("consequence_terms", [])
            for cons in consequences:
                if cons in coding_consequences:
                    return True
    
    # Check in formatted effects
    elif "effects" in variant_info:
        for effect in variant_info["effects"]:
            if effect.get("type") in coding_consequences:
                return True
    
    return False

def format_variant_identifier(
    chromosome: str,
    position: int,
    ref_allele: str,
    alt_allele: str
) -> str:
    """
    Create a standardized variant identifier string.
    
    Args:
        chromosome: Chromosome number or name
        position: Genomic position
        ref_allele: Reference allele
        alt_allele: Alternate allele
        
    Returns:
        Formatted variant identifier string (chr-pos-ref-alt)
    """
    # Normalize chromosome name
    chrom = str(chromosome)
    if not chrom.startswith('chr'):
        chrom = f"chr{chrom}"
    
    return f"{chrom}-{position}-{ref_allele}-{alt_allele}"

def parse_variant_identifier(variant_id: str) -> Dict[str, Any]:
    """
    Parse a variant identifier into its components.
    
    Args:
        variant_id: Variant identifier string (chr-pos-ref-alt)
        
    Returns:
        Dictionary with parsed chromosome, position, ref_allele, alt_allele
    """
    # Use regex to parse variant ID format
    match = re.match(r'(chr[^-]+)-(\d+)-([^-]+)-([^-]+)', variant_id)
    if match:
        return {
            "chromosome": match.group(1),
            "position": int(match.group(2)),
            "ref_allele": match.group(3),
            "alt_allele": match.group(4)
        }
    
    # Try alternative format without chr prefix
    match = re.match(r'([^-]+)-(\d+)-([^-]+)-([^-]+)', variant_id)
    if match:
        chromosome = match.group(1)
        if not chromosome.startswith('chr'):
            chromosome = f"chr{chromosome}"
        
        return {
            "chromosome": chromosome,
            "position": int(match.group(2)),
            "ref_allele": match.group(3),
            "alt_allele": match.group(4)
        }
    
    raise ValueError(f"Invalid variant identifier format: {variant_id}")

def save_analysis_results(results: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """
    Save analysis results to files in various formats.
    
    Args:
        results: Results dictionary to save
        output_dir: Directory to save files
        
    Returns:
        Dictionary mapping format names to file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate base filename from timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a unique identifier for this analysis
    results_hash = hashlib.md5(json.dumps(results, sort_keys=True).encode()).hexdigest()[:8]
    base_filename = f"genetix_analysis_{timestamp}_{results_hash}"
    
    # Save results in different formats
    saved_files = {}
    
    # Save JSON
    json_file = os.path.join(output_dir, f"{base_filename}.json")
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    saved_files["json"] = json_file
    
    # Save CSV (if applicable)
    if "guides" in results:
        guides_data = results["guides"]
        if isinstance(guides_data, list) and guides_data:
            # Get guide data from objects if necessary
            if hasattr(guides_data[0], '__dict__'):
                guides_data = [g.__dict__ for g in guides_data]
            
            guides_df = pd.DataFrame(guides_data)
            csv_file = os.path.join(output_dir, f"{base_filename}_guides.csv")
            guides_df.to_csv(csv_file, index=False)
            saved_files["guides_csv"] = csv_file
    
    if "variants" in results:
        variants_data = results["variants"]
        if isinstance(variants_data, list) and variants_data:
            variants_df = pd.DataFrame(variants_data)
            csv_file = os.path.join(output_dir, f"{base_filename}_variants.csv")
            variants_df.to_csv(csv_file, index=False)
            saved_files["variants_csv"] = csv_file
    
    # Log saved files
    logger.info(f"Saved analysis results to {len(saved_files)} files in {output_dir}")
    for fmt, path in saved_files.items():
        logger.info(f"  - {fmt}: {path}")
    
    return saved_files

def load_analysis_results(filepath: str) -> Dict[str, Any]:
    """
    Load analysis results from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Analysis results dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            results = json.load(f)
        return results
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in results file: {filepath}")

def create_summary_table(variant_info: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a summary table of variant information.
    
    Args:
        variant_info: Variant information dictionary
        
    Returns:
        Pandas DataFrame with summarized information
    """
    # Extract key information
    summary = {
        "Variant": variant_info.get("variant_id", ""),
        "Chromosome": variant_info.get("chromosome", ""),
        "Position": variant_info.get("position", ""),
        "Reference": variant_info.get("ref_allele", ""),
        "Alternate": variant_info.get("alt_allele", ""),
        "Effect": variant_info.get("effect", ""),
        "Confidence": variant_info.get("confidence", ""),
    }
    
    # Add ClinVar information if available
    if "clinvar" in variant_info:
        clinvar = variant_info["clinvar"]
        summary["ClinVar Significance"] = clinvar.get("clinical_significance", "")
        summary["ClinVar ID"] = clinvar.get("variation_id", "")
    
    # Add Ensembl VEP information if available
    if "vep" in variant_info:
        vep = variant_info["vep"]
        canonical = extract_canonical_transcript(vep.get("transcript_consequences", []))
        if canonical:
            summary["Gene"] = canonical.get("gene_symbol", "")
            summary["Consequence"] = canonical.get("consequence_terms", [""])[0]
            summary["SIFT"] = canonical.get("sift_prediction", "")
            summary["PolyPhen"] = canonical.get("polyphen_prediction", "")
    
    # Convert to DataFrame
    df = pd.DataFrame([summary])
    
    return df

def extract_canonical_transcript(transcript_consequences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract the canonical transcript from a list of transcript consequences.
    
    Args:
        transcript_consequences: List of transcript consequences from Ensembl VEP
        
    Returns:
        Canonical transcript dictionary, or empty dict if none found
    """
    # Look for canonical flag
    for tc in transcript_consequences:
        if tc.get("canonical") == 1:
            return tc
    
    # If no canonical transcript, return the first one if available
    if transcript_consequences:
        return transcript_consequences[0]
    
    # Return empty dictionary if no transcripts
    return {}

def load_fasta_sequence(fasta_file: str, chromosome: str = "", start: int = 0, end: int = 0) -> str:
    """
    Load a DNA sequence from a FASTA file.
    
    Args:
        fasta_file: Path to FASTA file
        chromosome: Chromosome name
        start: Start position (1-based, inclusive)
        end: End position (1-based, inclusive)
        
    Returns:
        DNA sequence string
    """
    try:
        from pyfaidx import Fasta
        
        # Load the FASTA file
        genome = Fasta(fasta_file)
        
        # Check if chromosome is in the file
        if chromosome and chromosome in genome:
            if start > 0 and end > 0:
                # Convert from 1-based to 0-based indexing
                seq = genome[chromosome][start-1:end].seq
                logger.info(f"Loaded {len(seq)} bp from {chromosome}:{start}-{end}")
                return seq
            else:
                seq = genome[chromosome][:].seq
                logger.info(f"Loaded {len(seq)} bp from {chromosome}")
                return seq
        else:
            # If no chromosome specified or not found, return the first sequence
            first_chrom = next(iter(genome.keys()))
            seq = genome[first_chrom][:].seq
            logger.info(f"Loaded {len(seq)} bp from {first_chrom}")
            return seq
            
    except ImportError:
        logger.error("pyfaidx package not installed, cannot load FASTA file")
        raise ImportError("pyfaidx package required to load FASTA files")
    except Exception as e:
        logger.error(f"Error loading FASTA file: {str(e)}")
        raise

def convert_guides_to_variants(
    guides: List[Dict[str, Any]],
    target_sequence: str,
    chromosome: str = "",
    start_position: int = 0
) -> List[Dict[str, Any]]:
    """
    Convert guide RNA designs to predicted variant effects.
    
    Args:
        guides: List of guide RNA dictionaries
        target_sequence: Target DNA sequence
        chromosome: Chromosome name/number
        start_position: Start position of target sequence in genome
        
    Returns:
        List of variant dictionaries
    """
    variants = []
    
    for guide in guides:
        # Get guide properties
        sequence = guide.get("sequence", "")
        pam = guide.get("pam", "")
        
        if not sequence:
            continue
        
        # Calculate guide position in genome coordinates
        guide_start = guide.get("start_position", 0)
        
        if start_position > 0:
            genome_position = start_position + guide_start
        else:
            genome_position = guide_start
        
        # Create a variant at the cut site (typically 3bp upstream of PAM)
        cut_offset = len(sequence) - 3
        if cut_offset < 0:
            cut_offset = 0
            
        cut_position = genome_position + cut_offset
        
        # Get reference base at cut position
        if cut_offset < len(target_sequence):
            ref_base = target_sequence[guide_start + cut_offset]
        else:
            ref_base = "N"
        
        # For simplicity, create a SNV at the cut site
        if ref_base == "A":
            alt_base = "G"
        elif ref_base == "G":
            alt_base = "A"
        elif ref_base == "C":
            alt_base = "T"
        else:  # ref_base == "T" or ref_base == "N"
            alt_base = "C"
        
        variant = {
            "chromosome": chromosome,
            "position": cut_position,
            "ref_allele": ref_base,
            "alt_allele": alt_base,
            "guide_sequence": sequence,
            "guide_pam": pam,
            "guide_start": guide_start,
            "guide_score": guide.get("overall_score", 0)
        }
        
        variants.append(variant)
    
    return variants

def parse_vcf_variants(vcf_file: str) -> List[Dict[str, Any]]:
    """
    Parse variants from a VCF file.
    
    Args:
        vcf_file: Path to VCF file
        
    Returns:
        List of variant dictionaries
    """
    variants = []
    
    try:
        # Read the VCF file
        with open(vcf_file, 'r') as f:
            for line in f:
                # Skip header lines
                if line.startswith('#'):
                    continue
                
                # Parse data line
                fields = line.strip().split('\t')
                if len(fields) < 5:
                    continue
                
                # Extract basic variant information
                chrom = fields[0]
                pos = int(fields[1])
                ref = fields[3]
                alts = fields[4].split(',')
                
                # Create variant entries for each alternate allele
                for alt in alts:
                    if alt == '.' or alt == ref:
                        continue
                    
                    variant = {
                        "chromosome": chrom,
                        "position": pos,
                        "ref_allele": ref,
                        "alt_allele": alt,
                        "vcf_source": os.path.basename(vcf_file)
                    }
                    
                    # Add variant ID if available
                    if fields[2] and fields[2] != '.':
                        variant["variant_id"] = fields[2]
                    
                    variants.append(variant)
        
        logger.info(f"Parsed {len(variants)} variants from {vcf_file}")
        return variants
    
    except Exception as e:
        logger.error(f"Error parsing VCF file: {str(e)}")
        return []

def generate_project_id() -> str:
    """
    Generate a unique project ID.
    
    Returns:
        Unique project ID string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = hashlib.md5(os.urandom(8)).hexdigest()[:6]
    return f"GenetiX_{timestamp}_{random_suffix}"

def prepare_workspace(project_name: str) -> Dict[str, str]:
    """
    Prepare a workspace directory structure for a new project.
    
    Args:
        project_name: Name of the project
        
    Returns:
        Dictionary with paths to directories
    """
    # Create base directories
    base_dir = os.path.join("projects", project_name)
    paths = {
        "base": base_dir,
        "crispr": os.path.join(base_dir, "crispr"),
        "variants": os.path.join(base_dir, "variants"),
        "simulation": os.path.join(base_dir, "simulation"),
        "reports": os.path.join(base_dir, "reports"),
        "data": os.path.join(base_dir, "data")
    }
    
    # Create directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    # Create a project info file
    info_file = os.path.join(base_dir, "project_info.json")
    project_info = {
        "name": project_name,
        "created": datetime.now().isoformat(),
        "status": "initialized",
        "paths": paths
    }
    
    with open(info_file, "w") as f:
        json.dump(project_info, f, indent=2)
    
    logger.info(f"Prepared workspace for project '{project_name}' at {base_dir}")
    return paths

def export_results_to_format(
    results: Dict[str, Any],
    output_path: str,
    format: str = "json"
) -> str:
    """
    Export results to a specific file format.
    
    Args:
        results: Results dictionary to export
        output_path: Path to save the export
        format: Format to export ("json", "csv", "tsv", "excel")
        
    Returns:
        Path to the exported file
    """
    # Create output directory if necessary
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    if format == "json":
        # Save as JSON
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        return output_path
    
    elif format in ["csv", "tsv", "excel"]:
        # Convert to DataFrame (flattening if necessary)
        if isinstance(results, list):
            df = pd.DataFrame(results)
        else:
            # Try to convert dictionary structure to a dataframe
            try:
                # If there's a key with a list value, use that as the basis
                list_keys = [k for k, v in results.items() if isinstance(v, list) and v]
                if list_keys:
                    key = list_keys[0]
                    df = pd.DataFrame(results[key])
                    
                    # Add metadata columns
                    for k, v in results.items():
                        if k != key and not isinstance(v, (list, dict)):
                            df[k] = v
                else:
                    # Try to make a single-row dataframe from the dictionary
                    df = pd.DataFrame([results])
            except Exception as e:
                logger.error(f"Could not convert results to DataFrame: {str(e)}")
                # Fallback to JSON
                return export_results_to_format(results, f"{output_path}.json", "json")
        
        # Export based on format
        if format == "csv":
            df.to_csv(output_path, index=False)
        elif format == "tsv":
            df.to_csv(output_path, sep='\t', index=False)
        elif format == "excel":
            df.to_excel(output_path, index=False)
        
        return output_path
    
    else:
        logger.warning(f"Unsupported export format: {format}. Using JSON instead.")
        return export_results_to_format(results, f"{output_path}.json", "json") 