"""
CRISPR Designer Module for Evo-CRISPR

This module provides functions for designing and evaluating CRISPR guide RNAs,
predicting off-targets, and preparing CRISPR constructs.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import re
from Bio import SeqIO
from Bio.Seq import Seq
import primer3
import logging
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PAM = "NGG"  # Standard Cas9 PAM
GUIDE_LENGTH = 20    # Standard guide RNA length
MAX_OFFTARGETS = 10  # Maximum number of off-targets to return
DEFAULT_GENOME = "hg38"  # Default human genome

@dataclass
class GuideRNA:
    """Class for storing guide RNA information"""
    sequence: str
    pam: str = "NGG"
    start_position: int = 0
    end_position: int = 0
    strand: str = "+"
    gc_content: float = 0.0
    off_targets: List[Dict[str, Any]] = field(default_factory=list)
    efficiency_score: float = 0.0
    specificity_score: float = 0.0
    overall_score: float = 0.0
    target_gene: str = ""
    notes: str = ""

    def __post_init__(self):
        """Calculate basic properties after initialization"""
        if self.sequence:
            self.gc_content = calculate_gc_content(self.sequence)
            
    def __str__(self) -> str:
        """String representation of GuideRNA"""
        return f"Guide: {self.sequence} PAM: {self.pam} Score: {self.overall_score:.2f}"


def find_pam_sites(sequence: str, pam: str = DEFAULT_PAM) -> List[Tuple[int, str, str]]:
    """
    Find all PAM sites in the given DNA sequence.
    
    Args:
        sequence: DNA sequence to search
        pam: PAM sequence to find (default: NGG for SpCas9)
        
    Returns:
        List of tuples containing (position, strand, PAM sequence)
    """
    # Convert IUPAC notation to regex
    pam_regex = pam.replace("N", "[ATCG]")
    
    # Find PAM sites on forward strand
    forward_matches = []
    for match in re.finditer(pam_regex, sequence):
        pos = match.start()
        forward_matches.append((pos, "+", match.group()))
    
    # Find PAM sites on reverse strand
    rev_seq = str(Seq(sequence).reverse_complement())
    reverse_matches = []
    for match in re.finditer(pam_regex, rev_seq):
        # Adjust position to original forward strand coordinates
        pos = len(sequence) - match.start() - len(match.group())
        reverse_matches.append((pos, "-", match.group()))
    
    return sorted(forward_matches + reverse_matches)


def design_guides(
    target_sequence: str,
    pam: str = DEFAULT_PAM,
    guide_length: int = GUIDE_LENGTH
) -> List[GuideRNA]:
    """
    Design guide RNAs for the target sequence.
    
    Args:
        target_sequence: DNA sequence to target
        pam: PAM sequence (default: NGG for SpCas9)
        guide_length: Length of guide RNA
        
    Returns:
        List of GuideRNA objects
    """
    # Find all PAM sites
    pam_sites = find_pam_sites(target_sequence, pam)
    
    guides = []
    for pos, strand, pam_seq in pam_sites:
        if strand == "+":
            # For forward strand, guide is upstream of PAM
            if pos >= guide_length:
                guide_seq = target_sequence[pos-guide_length:pos]
                guide = GuideRNA(
                    sequence=guide_seq,
                    pam=pam_seq,
                    start_position=pos-guide_length,
                    end_position=pos,
                    strand="+"
                )
                guides.append(guide)
        else:
            # For reverse strand, guide is downstream of PAM
            if pos + len(pam_seq) + guide_length <= len(target_sequence):
                # Get sequence and then reverse complement it
                guide_seq = target_sequence[pos+len(pam_seq):pos+len(pam_seq)+guide_length]
                guide_seq = str(Seq(guide_seq).reverse_complement())
                guide = GuideRNA(
                    sequence=guide_seq,
                    pam=pam_seq,
                    start_position=pos+len(pam_seq),
                    end_position=pos+len(pam_seq)+guide_length,
                    strand="-"
                )
                guides.append(guide)
    
    # Calculate scores for each guide
    for guide in guides:
        guide.efficiency_score = calculate_efficiency_score(guide.sequence)
        guide.specificity_score = 1.0  # Placeholder, would be based on off-target analysis
        guide.overall_score = 0.7 * guide.efficiency_score + 0.3 * guide.specificity_score
    
    # Sort guides by overall score
    guides.sort(key=lambda g: g.overall_score, reverse=True)
    
    return guides


def calculate_efficiency_score(guide_sequence: str) -> float:
    """
    Calculate the predicted efficiency score for a guide RNA.
    
    This function implements a simplified version of the Doench 2016 scoring algorithm.
    
    Args:
        guide_sequence: Guide RNA sequence (20 bp for standard Cas9)
        
    Returns:
        Efficiency score between 0 and 1
    """
    # This is a simplified placeholder for efficiency scoring
    # In a real implementation, this would use a trained model
    
    # Some basic rules that tend to correlate with efficiency:
    score = 0.5  # Base score
    
    # Penalize extreme GC content
    gc = calculate_gc_content(guide_sequence)
    if gc < 0.2 or gc > 0.8:
        score -= 0.2
    elif 0.4 <= gc <= 0.6:
        score += 0.1
    
    # Bonus for G at position 20 (PAM-proximal)
    if guide_sequence[-1] == 'G':
        score += 0.1
    
    # Penalty for homopolymers (4+ of same base)
    if re.search(r'AAAA|TTTT|GGGG|CCCC', guide_sequence):
        score -= 0.2
    
    # Ensure score is between 0 and 1
    return max(0, min(1, score))


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate the GC content of a DNA sequence.
    
    Args:
        sequence: DNA sequence
        
    Returns:
        GC content as a fraction between 0 and 1
    """
    if not sequence:
        return 0
    
    gc_count = sequence.count('G') + sequence.count('C')
    return gc_count / len(sequence)


def predict_off_targets(
    guide: GuideRNA,
    genome: str = DEFAULT_GENOME,
    max_mismatches: int = 3,
    max_results: int = MAX_OFFTARGETS
) -> List[Dict[str, Any]]:
    """
    Predict potential off-target sites for a guide RNA.
    
    Args:
        guide: GuideRNA object
        genome: Reference genome to search
        max_mismatches: Maximum number of mismatches to allow
        max_results: Maximum number of off-target results to return
        
    Returns:
        List of off-target sites with details
    """
    # This is a placeholder function that would interface with
    # existing tools like BWA, Cas-OFFinder, or BLAT
    
    # In a real implementation, this would:
    # 1. Search the genome for similar sequences
    # 2. Calculate mismatch positions
    # 3. Score the off-targets based on mismatch count and positions
    
    # Simulate some results for demonstration
    off_targets = []
    
    # In a real implementation, these would be actual genomic locations
    for i in range(3):
        # Generate a sequence with 1-3 mismatches
        mismatches = min(i+1, max_mismatches)
        
        # Create a copy of the guide with introduced mismatches
        off_target_seq = list(guide.sequence)
        mismatch_positions = np.random.choice(len(guide.sequence), mismatches, replace=False)
        
        for pos in mismatch_positions:
            orig_base = off_target_seq[pos]
            # Replace with a different base
            options = [b for b in "ACGT" if b != orig_base]
            off_target_seq[pos] = np.random.choice(options)
        
        off_target_seq = ''.join(off_target_seq)
        
        # Create an off-target entry
        off_target = {
            "sequence": off_target_seq,
            "chromosome": f"chr{np.random.randint(1, 23)}",
            "position": np.random.randint(1, 250000000),
            "strand": "+" if np.random.random() > 0.5 else "-",
            "mismatches": mismatches,
            "mismatch_positions": sorted(mismatch_positions),
            "score": 1.0 - (mismatches * 0.2)  # Simple scoring
        }
        
        off_targets.append(off_target)
    
    # Sort by score (higher is better)
    off_targets.sort(key=lambda ot: ot["score"], reverse=True)
    
    return off_targets[:max_results]


def design_validation_primers(
    sequence: str,
    target_position: int,
    product_size: Tuple[int, int] = (400, 600)
) -> Dict[str, Any]:
    """
    Design validation primers for a CRISPR target site.
    
    Args:
        sequence: Genomic sequence around the target site
        target_position: Position of the target site within the sequence
        product_size: Desired PCR product size range as (min, max)
        
    Returns:
        Dictionary with primer information
    """
    # Define the primer3 parameters
    primer_params = {
        'SEQUENCE_ID': 'CRISPR_validation',
        'SEQUENCE_TEMPLATE': sequence,
        'SEQUENCE_TARGET': [target_position, 20],  # Target the guide RNA site
        'PRIMER_TASK': 'generic',
        'PRIMER_PICK_LEFT_PRIMER': 1,
        'PRIMER_PICK_RIGHT_PRIMER': 1,
        'PRIMER_OPT_SIZE': 20,
        'PRIMER_MIN_SIZE': 18,
        'PRIMER_MAX_SIZE': 25,
        'PRIMER_OPT_TM': 60.0,
        'PRIMER_MIN_TM': 57.0,
        'PRIMER_MAX_TM': 63.0,
        'PRIMER_MIN_GC': 30.0,
        'PRIMER_MAX_GC': 70.0,
        'PRIMER_PRODUCT_SIZE_RANGE': [product_size[0], product_size[1]]
    }
    
    try:
        # Run primer3 design
        primer_results = primer3.bindings.designPrimers(primer_params)
        
        # Extract and format the results
        return {
            'left_primer': {
                'sequence': primer_results['PRIMER_LEFT_0_SEQUENCE'],
                'tm': primer_results['PRIMER_LEFT_0_TM'],
                'gc_percent': primer_results['PRIMER_LEFT_0_GC_PERCENT'],
                'position': primer_results['PRIMER_LEFT_0'][0]
            },
            'right_primer': {
                'sequence': primer_results['PRIMER_RIGHT_0_SEQUENCE'],
                'tm': primer_results['PRIMER_RIGHT_0_TM'],
                'gc_percent': primer_results['PRIMER_RIGHT_0_GC_PERCENT'],
                'position': primer_results['PRIMER_RIGHT_0'][0]
            },
            'product_size': primer_results['PRIMER_PAIR_0_PRODUCT_SIZE']
        }
    except Exception as e:
        logger.error(f"Primer design failed: {str(e)}")
        return {
            'error': f"Primer design failed: {str(e)}"
        }


def format_crispr_construct(
    guide: GuideRNA,
    backbone: str = "lentiCRISPRv2",
    add_restriction_sites: bool = True
) -> Dict[str, str]:
    """
    Format CRISPR construct for cloning.
    
    Args:
        guide: GuideRNA object
        backbone: Vector backbone to use
        add_restriction_sites: Whether to add restriction sites for cloning
        
    Returns:
        Dictionary with formatted oligos and construct information
    """
    # Standard U6 promoter gRNA scaffold for SpCas9
    scaffold = "GTTTTAGAGCTAGAAATAGCAAGTTAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTTT"
    
    # Format guide RNA for cloning
    if backbone == "lentiCRISPRv2":
        # lentiCRISPRv2 uses BsmBI restriction sites
        if add_restriction_sites:
            top_oligo = f"CACCG{guide.sequence}"
            bottom_oligo = f"AAAC{str(Seq(guide.sequence).reverse_complement())}C"
        else:
            top_oligo = f"G{guide.sequence}"
            bottom_oligo = f"{str(Seq(guide.sequence).reverse_complement())}C"
        
        expression_cassette = f"AGCTTGCATGCCTGCAGGTCGACTCTAGAGGATCCCCGGGTACCGAGCTCGAATTCCCCTTCACCGAGGGCCTATTTCCCATGATTCCTTCATATTTGCATATACGATACAAGGCTGTTAGAGAGATAATTGGAATTAATTTGACTGTAAACACAAAGATATTAGTACAAAATACGTGACGTAGAAAGTAATAATTTCTTGGGTAGTTTGCAGTTTTAAAATTATGTTTTAAAATGGACTATCATATGCTTACCGTAACTTGAAAGTATTTCGATTTCTTGGCTTTATATATCTTGTGGAAAGGACGAAACACC{guide.sequence}{scaffold}TGCGTTCGCTAGGGATGACCCTGCTGATTTTACTAGTTTATTAAGCGCTAGATTCTGTGCGTTGTTTAGCATGAAGAGCTTCAGTACTCCATTGGGCGCACGTCCTTCATCGCGCACATTCACCTCGATGTCGCTACGTTAGAGAGACTGATGAGAGTGGCGTGACACTGTTGACGCGCTCTACTTTCTGTACTAGACACGGGCAGCCGGCCGGCAGGGCCAATGCTACTTGATTGTTTTATCACACGAAGGGCGCTTACGACTTGAAGCCGTTCACCGCA"
        
    else:
        # Generic format for other plasmids
        top_oligo = guide.sequence
        bottom_oligo = str(Seq(guide.sequence).reverse_complement())
        expression_cassette = f"NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN{guide.sequence}{scaffold}NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN"
    
    return {
        'top_oligo': top_oligo,
        'bottom_oligo': bottom_oligo,
        'full_guide_sequence': guide.sequence + scaffold,
        'expression_cassette': expression_cassette,
        'backbone': backbone
    }


def rank_guides(guides: List[GuideRNA]) -> List[GuideRNA]:
    """
    Rank guide RNAs based on their overall scores.
    
    Args:
        guides: List of GuideRNA objects
        
    Returns:
        Sorted list of GuideRNA objects
    """
    return sorted(guides, key=lambda g: g.overall_score, reverse=True)


def analyze_target_sequence(
    sequence: str,
    target_gene: str = "",
    pam: str = DEFAULT_PAM,
    guide_length: int = GUIDE_LENGTH
) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of a target sequence for CRISPR editing.
    
    Args:
        sequence: DNA sequence to analyze
        target_gene: Name of the target gene
        pam: PAM sequence to use
        guide_length: Length of guide RNA
        
    Returns:
        Dictionary with analysis results
    """
    # Step 1: Find all possible guides
    guides = design_guides(sequence, pam, guide_length)
    
    # Set target gene info for all guides
    for guide in guides:
        guide.target_gene = target_gene
    
    # Step 2: Add off-target predictions for top 5 guides
    for guide in guides[:5]:
        guide.off_targets = predict_off_targets(guide)
    
    # Step 3: Design validation primers for top guide
    validation_primers = {}
    if guides:
        mid_point = len(sequence) // 2
        validation_primers = design_validation_primers(sequence, mid_point)
    
    # Step 4: Format CRISPR construct for top guide
    crispr_construct = {}
    if guides:
        crispr_construct = format_crispr_construct(guides[0])
    
    # Return comprehensive analysis
    return {
        'target_sequence': sequence,
        'target_gene': target_gene,
        'guides': guides,
        'validation_primers': validation_primers,
        'crispr_construct': crispr_construct,
        'summary': {
            'total_guides_found': len(guides),
            'top_guide_score': guides[0].overall_score if guides else 0,
            'gc_content': calculate_gc_content(sequence)
        }
    }


def get_sequence_from_genome(
    genome_id: str,
    chromosome: str,
    start: int,
    end: int
) -> str:
    """
    Retrieve a DNA sequence from a reference genome.
    
    Args:
        genome_id: Genome identifier (e.g., 'hg38')
        chromosome: Chromosome name
        start: Start position (1-based)
        end: End position (inclusive)
        
    Returns:
        DNA sequence as string
    """
    # This is a placeholder function that would retrieve sequence from a local
    # genome file or online service like Ensembl or UCSC
    
    # In a real implementation, this would use pyfaidx or similar to
    # extract from a local FASTA file, or call an appropriate API
    
    # For demonstration, return a simulated sequence
    seq_length = end - start + 1
    bases = ['A', 'C', 'G', 'T']
    sequence = ''.join(np.random.choice(bases, seq_length))
    
    return sequence


def export_results_to_csv(analysis_results: Dict[str, Any], output_file: str) -> bool:
    """
    Export CRISPR guide analysis results to a CSV file.
    
    Args:
        analysis_results: Results from analyze_target_sequence
        output_file: Path to the output CSV file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create a DataFrame from the guides
        guides_data = []
        for i, guide in enumerate(analysis_results['guides']):
            guide_data = {
                'Rank': i+1,
                'Guide Sequence': guide.sequence,
                'PAM': guide.pam,
                'Start Position': guide.start_position,
                'End Position': guide.end_position,
                'Strand': guide.strand,
                'GC Content': guide.gc_content,
                'Efficiency Score': guide.efficiency_score,
                'Specificity Score': guide.specificity_score,
                'Overall Score': guide.overall_score,
                'Off-target Count': len(guide.off_targets)
            }
            guides_data.append(guide_data)
        
        # Create DataFrame and export
        df = pd.DataFrame(guides_data)
        df.to_csv(output_file, index=False)
        return True
        
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")
        return False 