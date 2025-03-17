"""
Evo2 Client Module for GenetiXplorer

This module provides functions to interact with the Evo2 DNA language model
from ARC Institute (https://github.com/ArcInstitute/evo2). Evo2 is used
for DNA sequence modeling, likelihood scoring, and evolutionary predictions.
"""

import os
import logging
import requests
import json
from typing import Dict, List, Union, Optional, Any
import numpy as np
import time
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables if using .env file
load_dotenv()

class Evo2Client:
    """Client for interfacing with the Evo2 DNA language model API"""
    
    def __init__(self, api_key: Optional[str] = None, model_size: str = "7B"):
        """
        Initialize the Evo2 client
        
        Args:
            api_key: API key for Evo2 access (defaults to environment variable)
            model_size: Size of the Evo2 model to use ("40B", "7B", or "1B")
        """
        # Set up API key - try multiple sources
        self.api_key = api_key
        
        if self.api_key is None:
            # Try to get from streamlit secrets
            try:
                self.api_key = st.secrets.get("EVO2_API_KEY")
            except:
                pass
                
        if self.api_key is None:
            # Try to get from environment
            self.api_key = os.environ.get("EVO2_API_KEY")
            
        if self.api_key is None:
            logger.warning("No Evo2 API key found. Some functionality will be limited.")
            
        # Set up API endpoints
        self.base_url = "https://api.arc.io/v1/evo2"
        self.model_size = model_size
        
        # Validate model size
        valid_sizes = ["40B", "7B", "1B"]
        if model_size not in valid_sizes:
            logger.warning(f"Invalid model size: {model_size}. Using default 7B.")
            self.model_size = "7B"
            
        # Test connection
        if self.api_key:
            try:
                self.test_connection()
                logger.info(f"Successfully connected to Evo2 API using {self.model_size} model")
            except Exception as e:
                logger.error(f"Failed to connect to Evo2 API: {str(e)}")
    
    def test_connection(self) -> bool:
        """
        Test the connection to the Evo2 API
        
        Returns:
            Boolean indicating successful connection
        """
        try:
            headers = self._get_headers()
            response = requests.get(
                f"{self.base_url}/health",
                headers=headers
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get the headers for API requests
        
        Returns:
            Dictionary of headers
        """
        headers = {
            "Content-Type": "application/json",
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        return headers
    
    def sequence_likelihood(self, sequence: str) -> Dict[str, Any]:
        """
        Calculate the likelihood of a DNA sequence
        
        Args:
            sequence: DNA sequence to analyze
            
        Returns:
            Dictionary with likelihood scores
        """
        if not self.api_key:
            return self._mock_likelihood(sequence)
            
        try:
            headers = self._get_headers()
            payload = {
                "sequence": sequence,
                "model": f"evo2-{self.model_size.lower()}"
            }
            
            response = requests.post(
                f"{self.base_url}/score",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to calculate sequence likelihood: {str(e)}")
            return self._mock_likelihood(sequence)
    
    def _mock_likelihood(self, sequence: str) -> Dict[str, Any]:
        """
        Create mock likelihood results when API is unavailable
        
        Args:
            sequence: DNA sequence to analyze
            
        Returns:
            Dictionary with mock likelihood scores
        """
        # Calculate a deterministic but pseudo-random score based on sequence content
        gc_content = sum(1 for base in sequence if base in "GC") / len(sequence) if sequence else 0
        
        # More complex scores based on sequence properties
        complexity = len(set(sequence[i:i+3] for i in range(len(sequence)-2))) / (len(sequence)-2) if len(sequence) > 2 else 0
        
        # Mock likelihood score (higher is more likely)
        likelihood_score = 0.5 + (gc_content - 0.5) * 0.3 + (complexity - 0.25) * 0.2
        likelihood_score = max(0.01, min(0.99, likelihood_score))
        
        return {
            "sequence": sequence,
            "likelihood": likelihood_score,
            "per_base_scores": [likelihood_score] * len(sequence),
            "is_mock": True,
            "gc_content": gc_content,
            "sequence_complexity": complexity
        }
    
    def embed_sequence(self, sequence: str) -> Dict[str, Any]:
        """
        Generate embeddings for a DNA sequence
        
        Args:
            sequence: DNA sequence to embed
            
        Returns:
            Dictionary with sequence embeddings
        """
        if not self.api_key:
            return self._mock_embedding(sequence)
            
        try:
            headers = self._get_headers()
            payload = {
                "sequence": sequence,
                "model": f"evo2-{self.model_size.lower()}"
            }
            
            response = requests.post(
                f"{self.base_url}/embed",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to embed sequence: {str(e)}")
            return self._mock_embedding(sequence)
    
    def _mock_embedding(self, sequence: str) -> Dict[str, Any]:
        """
        Create mock embedding results when API is unavailable
        
        Args:
            sequence: DNA sequence to embed
            
        Returns:
            Dictionary with mock embeddings
        """
        # Generate a deterministic pseudo-random embedding
        np.random.seed(sum(ord(c) for c in sequence[:100]))
        
        # Create a small mock embedding (actual embeddings would be larger)
        embedding_dim = 64
        embedding = np.random.normal(0, 1, embedding_dim).tolist()
        
        return {
            "sequence": sequence,
            "embedding": embedding,
            "is_mock": True
        }
    
    def generate_sequence(self, 
                         prompt: str, 
                         max_length: int = 1000, 
                         temperature: float = 0.8) -> Dict[str, Any]:
        """
        Generate a DNA sequence from a prompt
        
        Args:
            prompt: Initial DNA sequence as prompt
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature (higher = more diversity)
            
        Returns:
            Dictionary with generated sequence
        """
        if not self.api_key:
            return self._mock_generation(prompt, max_length)
            
        try:
            headers = self._get_headers()
            payload = {
                "prompt": prompt,
                "max_length": max_length,
                "temperature": temperature,
                "model": f"evo2-{self.model_size.lower()}"
            }
            
            response = requests.post(
                f"{self.base_url}/generate",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to generate sequence: {str(e)}")
            return self._mock_generation(prompt, max_length)
    
    def _mock_generation(self, prompt: str, max_length: int) -> Dict[str, Any]:
        """
        Create mock sequence generation when API is unavailable
        
        Args:
            prompt: Initial DNA sequence as prompt
            max_length: Maximum sequence length to generate
            
        Returns:
            Dictionary with mock generated sequence
        """
        # Use the prompt as a seed
        np.random.seed(sum(ord(c) for c in prompt[:20]))
        
        # Start with the prompt
        sequence = prompt
        
        # Add random bases to reach the desired length
        bases = "ACGT"
        length_to_add = min(max_length - len(sequence), 1000)
        
        if length_to_add > 0:
            # Simple Markov-like generation
            prev_base = sequence[-1] if sequence else np.random.choice(list(bases))
            
            # Transition probabilities (base -> next base)
            transitions = {
                'A': [0.3, 0.2, 0.3, 0.2],  # A -> A,C,G,T
                'C': [0.2, 0.3, 0.2, 0.3],
                'G': [0.3, 0.2, 0.3, 0.2],
                'T': [0.2, 0.3, 0.2, 0.3]
            }
            
            for _ in range(length_to_add):
                probs = transitions.get(prev_base, [0.25, 0.25, 0.25, 0.25])
                next_base = np.random.choice(list(bases), p=probs)
                sequence += next_base
                prev_base = next_base
        
        return {
            "prompt": prompt,
            "generated_sequence": sequence,
            "is_mock": True
        }
    
    def predict_evolutionary_trajectory(self, 
                                      sequence: str, 
                                      mutations: List[Dict[str, Any]],
                                      generations: int = 100,
                                      population_size: int = 1000) -> Dict[str, Any]:
        """
        Predict evolutionary trajectory of a sequence with specific mutations
        
        Args:
            sequence: Reference DNA sequence
            mutations: List of mutations to introduce
            generations: Number of generations to simulate
            population_size: Size of the population
            
        Returns:
            Dictionary with evolutionary trajectory prediction
        """
        if not self.api_key:
            return self._mock_evolutionary_trajectory(sequence, mutations, generations, population_size)
            
        try:
            headers = self._get_headers()
            payload = {
                "sequence": sequence,
                "mutations": mutations,
                "generations": generations,
                "population_size": population_size,
                "model": f"evo2-{self.model_size.lower()}"
            }
            
            response = requests.post(
                f"{self.base_url}/evolve",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to predict evolutionary trajectory: {str(e)}")
            return self._mock_evolutionary_trajectory(sequence, mutations, generations, population_size)
    
    def _mock_evolutionary_trajectory(self, 
                                    sequence: str, 
                                    mutations: List[Dict[str, Any]],
                                    generations: int,
                                    population_size: int) -> Dict[str, Any]:
        """
        Create mock evolutionary trajectory when API is unavailable
        
        Args:
            sequence: Reference DNA sequence
            mutations: List of mutations to introduce
            generations: Number of generations to simulate
            population_size: Size of the population
            
        Returns:
            Dictionary with mock evolutionary trajectory
        """
        # Create a seed based on sequence and mutations
        seed = sum(ord(c) for c in sequence[:50])
        for mutation in mutations:
            seed += sum(ord(str(v)) for v in mutation.values())
        
        np.random.seed(seed)
        
        # Generate fitness values for each mutation
        mutation_fitness = {}
        for mutation in mutations:
            # Assign a fitness effect to each mutation
            # Simplistic model: 10% deleterious, 80% neutral, 10% beneficial
            effect = np.random.choice(["deleterious", "neutral", "beneficial"], 
                                     p=[0.1, 0.8, 0.1])
            
            if effect == "deleterious":
                fitness = np.random.uniform(0.7, 0.99)
            elif effect == "neutral":
                fitness = np.random.uniform(0.99, 1.01)
            else:  # beneficial
                fitness = np.random.uniform(1.01, 1.3)
                
            mutation_id = f"{mutation.get('position')}_{mutation.get('ref')}>{mutation.get('alt')}"
            mutation_fitness[mutation_id] = fitness
        
        # Simulate allele frequency trajectories
        allele_frequencies = []
        mean_fitness = []
        
        # Initial frequency (everyone is wildtype)
        current_frequencies = {mutation_id: 0.01 for mutation_id in mutation_fitness}
        
        for gen in range(generations + 1):
            # Record current state
            allele_frequencies.append({
                "generation": gen,
                "frequencies": current_frequencies.copy()
            })
            
            # Calculate mean fitness
            current_mean_fitness = 1.0
            for mutation_id, freq in current_frequencies.items():
                # Contribution to fitness from this mutation
                mut_fitness = mutation_fitness[mutation_id]
                # Weight by frequency
                current_mean_fitness += (mut_fitness - 1.0) * freq
            
            mean_fitness.append({
                "generation": gen,
                "mean_fitness": current_mean_fitness
            })
            
            # Update frequencies based on selection
            for mutation_id, fitness in mutation_fitness.items():
                # Calculate selection coefficient (s)
                s = fitness - 1.0
                
                # Current frequency
                p = current_frequencies[mutation_id]
                
                # Apply selection formula: Δp = sp(1-p)/(1 + sp)
                delta_p = (s * p * (1 - p)) / (1 + s * p)
                
                # Add some random drift based on population size
                drift = np.random.normal(0, np.sqrt(p * (1 - p) / (2 * population_size)))
                
                # Update frequency
                new_p = p + delta_p + drift
                
                # Ensure frequency stays between 0 and 1
                new_p = max(0, min(1, new_p))
                
                # Store new frequency
                current_frequencies[mutation_id] = new_p
        
        return {
            "trajectory": {
                "allele_frequencies": allele_frequencies,
                "mean_fitness": mean_fitness
            },
            "mutation_fitness": [
                {"mutation": mutation_id, "fitness": fitness}
                for mutation_id, fitness in mutation_fitness.items()
            ],
            "is_mock": True
        }
    
    def get_fitness_landscape(self, sequence: str, positions: List[int]) -> Dict[str, Any]:
        """
        Calculate the fitness landscape for specified positions in a sequence
        
        Args:
            sequence: Reference DNA sequence
            positions: List of positions to analyze
            
        Returns:
            Dictionary with fitness landscape data
        """
        if not self.api_key:
            return self._mock_fitness_landscape(sequence, positions)
            
        try:
            headers = self._get_headers()
            payload = {
                "sequence": sequence,
                "positions": positions,
                "model": f"evo2-{self.model_size.lower()}"
            }
            
            response = requests.post(
                f"{self.base_url}/fitness_landscape",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get fitness landscape: {str(e)}")
            return self._mock_fitness_landscape(sequence, positions)
    
    def _mock_fitness_landscape(self, sequence: str, positions: List[int]) -> Dict[str, Any]:
        """
        Create mock fitness landscape when API is unavailable
        
        Args:
            sequence: Reference DNA sequence
            positions: List of positions to analyze
            
        Returns:
            Dictionary with mock fitness landscape
        """
        # Create a seed based on sequence and positions
        seed = sum(ord(c) for c in sequence[:50]) + sum(positions)
        np.random.seed(seed)
        
        landscape = []
        for pos in positions:
            if pos >= len(sequence):
                continue
                
            ref_base = sequence[pos] if pos < len(sequence) else "N"
            
            for alt_base in "ACGT":
                if alt_base == ref_base:
                    continue
                    
                # Generate a fitness effect for this mutation
                # Most mutations should be neutral or slightly deleterious
                rand_val = np.random.random()
                
                if rand_val < 0.1:  # 10% chance of beneficial
                    fitness = np.random.uniform(1.01, 1.2)
                    effect = "beneficial"
                elif rand_val < 0.3:  # 20% chance of deleterious
                    fitness = np.random.uniform(0.8, 0.99)
                    effect = "deleterious"
                else:  # 70% chance of neutral
                    fitness = np.random.uniform(0.99, 1.01)
                    effect = "neutral"
                
                landscape.append({
                    "position": pos,
                    "ref": ref_base,
                    "alt": alt_base,
                    "fitness": fitness,
                    "effect": effect
                })
        
        return {
            "landscape": landscape,
            "is_mock": True
        } 