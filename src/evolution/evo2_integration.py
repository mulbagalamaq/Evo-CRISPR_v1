"""
Evo2 Integration Module for GenetiXplorer

This module integrates the Evo2 DNA language model with our existing
evolutionary simulation framework, enhancing predictions with state-of-the-art
DNA modeling capabilities.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import json
from pathlib import Path

from src.evolution.evo2_client import Evo2Client
from src.evolution.population_sim import (
    PopulationSimulator,
    PopulationParams,
    SimulationParams,
    fitness_landscape_from_variants
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Evo2Simulation:
    """
    Enhanced evolutionary simulation using Evo2 DNA language model
    """
    
    def __init__(self, model_size: str = "7B", api_key: Optional[str] = None):
        """
        Initialize the Evo2-enhanced simulation
        
        Args:
            model_size: Size of Evo2 model to use ("40B", "7B", or "1B")
            api_key: API key for Evo2 (optional, can be loaded from environment)
        """
        self.client = Evo2Client(api_key=api_key, model_size=model_size)
        logger.info(f"Initialized Evo2 simulation with {model_size} model")
    
    def calculate_variant_fitness(self, 
                                 sequence: str, 
                                 variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculate fitness effects for a list of variants using Evo2
        
        Args:
            sequence: Reference DNA sequence
            variants: List of variant dictionaries
            
        Returns:
            List of variants with added fitness scores
        """
        # Format mutations for Evo2
        mutations = []
        for variant in variants:
            position = variant.get("position", 0)
            
            # Adjust position if it's 1-indexed in variants but 0-indexed for sequence
            seq_position = position - 1 if position > 0 else position
            
            # Get reference base from sequence if possible
            ref_base = sequence[seq_position] if 0 <= seq_position < len(sequence) else variant.get("ref_allele", "")
            
            mutations.append({
                "position": seq_position,
                "ref": ref_base,
                "alt": variant.get("alt_allele", "")
            })
        
        # Get fitness landscape from Evo2
        landscape_data = self.client.get_fitness_landscape(sequence, [m["position"] for m in mutations])
        
        # Map fitness values back to variants
        fitness_map = {}
        for entry in landscape_data.get("landscape", []):
            key = f"{entry['position']}_{entry['ref']}_{entry['alt']}"
            fitness_map[key] = entry
        
        # Add fitness to each variant
        enhanced_variants = []
        for variant in variants:
            position = variant.get("position", 0)
            seq_position = position - 1 if position > 0 else position
            
            ref = variant.get("ref_allele", "")
            alt = variant.get("alt_allele", "")
            
            key = f"{seq_position}_{ref}_{alt}"
            fitness_data = fitness_map.get(key, {})
            
            # Create enhanced variant with fitness data
            enhanced_variant = variant.copy()
            enhanced_variant["evo2_fitness"] = fitness_data.get("fitness", 1.0)
            enhanced_variant["evo2_effect"] = fitness_data.get("effect", "neutral")
            
            enhanced_variants.append(enhanced_variant)
        
        return enhanced_variants
    
    def run_evolutionary_simulation(self,
                                   sequence: str,
                                   variants: List[Dict[str, Any]],
                                   generations: int = 100,
                                   population_size: int = 1000,
                                   initial_frequency: float = 0.01,
                                   use_traditional_sim: bool = True) -> Dict[str, Any]:
        """
        Run an evolutionary simulation with Evo2 DNA model enhancements
        
        Args:
            sequence: Reference DNA sequence
            variants: List of variants to simulate
            generations: Number of generations to simulate
            population_size: Size of the population
            initial_frequency: Initial frequency of variants
            use_traditional_sim: Whether to also run traditional simulation
            
        Returns:
            Dictionary with simulation results
        """
        # Calculate variant fitness using Evo2
        enhanced_variants = self.calculate_variant_fitness(sequence, variants)
        
        # Format mutations for Evo2 trajectory prediction
        mutations = []
        for variant in enhanced_variants:
            position = variant.get("position", 0)
            seq_position = position - 1 if position > 0 else position
            
            # Get reference base from sequence if possible
            ref_base = sequence[seq_position] if 0 <= seq_position < len(sequence) else variant.get("ref_allele", "")
            
            mutations.append({
                "position": seq_position,
                "ref": ref_base,
                "alt": variant.get("alt_allele", ""),
                "initial_frequency": initial_frequency
            })
        
        # Run Evo2 trajectory prediction
        evo2_trajectory = self.client.predict_evolutionary_trajectory(
            sequence=sequence,
            mutations=mutations,
            generations=generations,
            population_size=population_size
        )
        
        # Create fitness landscape for traditional simulation
        fitness_landscape = {}
        for variant in enhanced_variants:
            variant_id = f"{variant.get('chromosome', 'chr1')}-{variant.get('position', 0)}-{variant.get('ref_allele', '')}-{variant.get('alt_allele', '')}"
            fitness_landscape[variant_id] = variant.get("evo2_fitness", 1.0)
        
        results = {
            "evo2_trajectory": evo2_trajectory,
            "enhanced_variants": enhanced_variants,
            "fitness_landscape": fitness_landscape
        }
        
        # Run traditional simulation if requested
        if use_traditional_sim:
            try:
                # Set up traditional simulation
                pop_params = PopulationParams(
                    size=population_size,
                    carrying_capacity=population_size,
                    initial_edited_frequency=initial_frequency,
                    num_loci=len(enhanced_variants)
                )
                
                sim_params = SimulationParams(
                    generations=generations,
                    selection_model="multiplicative",
                    fitness_landscape=fitness_landscape,
                    stochastic=True
                )
                
                # Create and run simulator
                simulator = PopulationSimulator(pop_params, sim_params)
                trad_results = simulator.run_simulation()
                
                # Add traditional results
                results["traditional_simulation"] = trad_results
                
                # Compare results
                results["comparison"] = self._compare_trajectories(
                    evo2_trajectory=evo2_trajectory,
                    traditional_trajectory=trad_results
                )
                
            except Exception as e:
                logger.error(f"Error running traditional simulation: {str(e)}")
                results["traditional_simulation"] = {"error": str(e)}
        
        return results
    
    def _compare_trajectories(self, 
                            evo2_trajectory: Dict[str, Any], 
                            traditional_trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare trajectories from Evo2 and traditional simulation
        
        Args:
            evo2_trajectory: Results from Evo2 simulation
            traditional_trajectory: Results from traditional simulation
            
        Returns:
            Dictionary with comparison metrics
        """
        # Extract key metrics
        comparison = {
            "correlation": {},
            "final_state_diff": {},
            "convergence_metrics": {}
        }
        
        # Compare final frequencies
        try:
            # Evo2 final frequencies
            evo2_final = evo2_trajectory.get("trajectory", {}).get("allele_frequencies", [])[-1].get("frequencies", {})
            
            # Traditional final frequencies
            trad_final = {}
            if "history" in traditional_trajectory and "edited_frequency" in traditional_trajectory["history"]:
                trad_final_freqs = traditional_trajectory["history"]["edited_frequency"]
                if trad_final_freqs:
                    # Get the last value
                    trad_final["average"] = trad_final_freqs[-1]
            
            # Calculate differences
            for mutation_id, evo2_freq in evo2_final.items():
                trad_freq = trad_final.get("average", 0)
                comparison["final_state_diff"][mutation_id] = abs(evo2_freq - trad_freq)
            
            # Overall difference
            if evo2_final and trad_final:
                avg_evo2 = sum(evo2_final.values()) / len(evo2_final)
                avg_trad = trad_final.get("average", 0)
                comparison["final_state_diff"]["overall"] = abs(avg_evo2 - avg_trad)
        
        except Exception as e:
            logger.error(f"Error comparing final states: {str(e)}")
            comparison["final_state_diff"]["error"] = str(e)
        
        return comparison
    
    def calculate_sequence_likelihoods(self, 
                                    reference_sequence: str, 
                                    edited_sequences: List[str]) -> Dict[str, Any]:
        """
        Calculate likelihood scores for edited sequences using Evo2
        
        Args:
            reference_sequence: Original DNA sequence
            edited_sequences: List of edited sequences to evaluate
            
        Returns:
            Dictionary with likelihood scores
        """
        results = {
            "reference": self.client.sequence_likelihood(reference_sequence),
            "edited": []
        }
        
        for seq in edited_sequences:
            likelihood = self.client.sequence_likelihood(seq)
            results["edited"].append(likelihood)
        
        # Calculate relative likelihoods
        reference_score = results["reference"].get("likelihood", 1.0)
        for edit in results["edited"]:
            edit_score = edit.get("likelihood", 0.0)
            relative_score = edit_score / reference_score if reference_score > 0 else 0
            edit["relative_likelihood"] = relative_score
        
        return results
    
    def design_optimal_edit(self, 
                          sequence: str, 
                          target_positions: List[int],
                          desired_effect: str = "beneficial") -> Dict[str, Any]:
        """
        Design optimal edits for target positions to achieve desired effect
        
        Args:
            sequence: Reference DNA sequence
            target_positions: Positions to consider for editing
            desired_effect: Desired effect ("beneficial", "neutral", "deleterious")
            
        Returns:
            Dictionary with optimal edit recommendations
        """
        # Get fitness landscape for all possible edits at target positions
        landscape_data = self.client.get_fitness_landscape(sequence, target_positions)
        
        # Filter landscape based on desired effect
        filtered_landscape = []
        for entry in landscape_data.get("landscape", []):
            # Skip entries without effect information
            if "effect" not in entry:
                continue
                
            # Filter by desired effect
            if entry["effect"] == desired_effect:
                filtered_landscape.append(entry)
        
        # Sort by fitness (for beneficial: higher is better, for deleterious: lower is better)
        if desired_effect == "beneficial":
            filtered_landscape.sort(key=lambda x: x.get("fitness", 0), reverse=True)
        elif desired_effect == "deleterious":
            filtered_landscape.sort(key=lambda x: x.get("fitness", 1))
        else:  # neutral
            # Sort by how close to 1.0 the fitness is
            filtered_landscape.sort(key=lambda x: abs(x.get("fitness", 1) - 1.0))
        
        # Take top recommendations
        recommendations = filtered_landscape[:10] if len(filtered_landscape) > 10 else filtered_landscape
        
        return {
            "recommendations": recommendations,
            "desired_effect": desired_effect,
            "num_candidates_found": len(filtered_landscape)
        }
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> str:
        """
        Save simulation results to a file
        
        Args:
            results: Simulation results dictionary
            output_path: Path to save results
            
        Returns:
            Path to saved file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert non-serializable objects
        serializable_results = json.dumps(results, default=lambda o: str(o) if not isinstance(o, (dict, list, str, int, float, bool, type(None))) else o)
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(serializable_results)
        
        logger.info(f"Saved Evo2 simulation results to {output_path}")
        return output_path 