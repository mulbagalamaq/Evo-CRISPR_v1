"""
AI Chatbot Module for EvoBeevos+ Variant Predictor

This module provides functions to interact with the Groq API
for creating an AI assistant that can provide expert-level information
about genetic variants and their clinical significance.
"""

import os
import groq
import json
from typing import Dict, Any, List, Optional, Tuple, Union

# Constants
DEFAULT_MODEL = "mixtral-8x7b-32768"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 1024
DEFAULT_SYSTEM_PROMPT = """
You are an expert geneticist and bioinformatician specializing in variant analysis and interpretation.
Your role is to assist researchers and clinicians in understanding genetic variants, their implications,
and associated genetic conditions.

When providing information about variants:
1. Be accurate, scientific, and evidence-based
2. Explain concepts clearly using appropriate technical language
3. Reference relevant databases, literature, or guidelines when applicable
4. Acknowledge limitations in current knowledge
5. Do not provide personalized medical advice

Your knowledge covers:
- Variant nomenclature and classification systems
- Molecular mechanisms of genetic variants
- Disease associations and inheritance patterns
- Protein structure and function implications
- Guidelines for variant interpretation (ACMG/AMP)
- Relevant databases (ClinVar, gnomAD, OMIM, etc.)
"""

def initialize_client(api_key: Optional[str] = None) -> Optional[groq.Groq]:
    """
    Initialize the Groq client with an API key.
    
    Args:
        api_key: Groq API key
        
    Returns:
        Groq client or None if initialization fails
    """
    try:
        # If no API key is provided, try to get it from environment variable
        if api_key is None:
            api_key = os.environ.get("GROQ_API_KEY")
            
        if not api_key:
            return None
            
        # Initialize client
        client = groq.Groq(api_key=api_key)
        return client
        
    except Exception as e:
        print(f"Error initializing Groq client: {str(e)}")
        return None

def generate_response(
    client: groq.Groq,
    query: str,
    variant_info: Optional[Dict[str, Any]] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> Dict[str, Any]:
    """
    Generate a response from the AI chatbot for a user query.
    
    Args:
        client: Initialized Groq client
        query: User's question about a variant
        variant_info: Optional dictionary with variant details to provide context
        model: Model to use for generation
        temperature: Temperature for generation (0-1)
        max_tokens: Maximum number of tokens to generate
        system_prompt: System prompt to guide the model's behavior
        
    Returns:
        Dictionary with the AI response and metadata
    """
    try:
        # Start with the base system prompt
        enhanced_system_prompt = system_prompt
        
        # If variant information is provided, add it to the system prompt
        if variant_info:
            variant_context = "\nHere's information about the variant being discussed:\n"
            
            # Add basic variant information
            if "variant" in variant_info:
                variant_context += f"- Variant: {variant_info['variant']}\n"
            
            # Add prediction information if available
            if "effect" in variant_info and "confidence" in variant_info:
                variant_context += f"- Predicted effect: {variant_info['effect']} (confidence: {variant_info['confidence']:.1%})\n"
            
            # Add ClinVar information if available
            if "clinvar" in variant_info:
                cv = variant_info["clinvar"]
                if "clinical_significance" in cv:
                    variant_context += f"- ClinVar classification: {cv['clinical_significance']}\n"
                if "conditions" in cv and cv["conditions"]:
                    conditions = ", ".join(cv["conditions"][:3])
                    variant_context += f"- Associated conditions: {conditions}\n"
            
            # Add Ensembl information if available
            if "ensembl" in variant_info:
                ens = variant_info["ensembl"]
                if "gene_symbol" in ens:
                    variant_context += f"- Gene: {ens['gene_symbol']}\n"
                if "consequences" in ens and ens["consequences"]:
                    cons = ens["consequences"][0] if isinstance(ens["consequences"], list) else ens["consequences"]
                    if "consequence" in cons:
                        variant_context += f"- Molecular consequence: {cons['consequence']}\n"
                    if "impact" in cons:
                        variant_context += f"- Impact: {cons['impact']}\n"
            
            enhanced_system_prompt += variant_context
        
        # Call the Groq API
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": query}
            ],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Extract the response text
        response_text = chat_completion.choices[0].message.content
        
        # Return the response with metadata
        return {
            "response": response_text,
            "model": model,
            "tokens": chat_completion.usage.total_tokens,
            "success": True
        }
        
    except Exception as e:
        # Return error information
        return {
            "response": f"I apologize, but I encountered an error while generating a response: {str(e)}",
            "error": str(e),
            "success": False
        }

def get_variant_explanation(
    client: groq.Groq,
    variant_info: Dict[str, Any],
    format_type: str = "detailed",
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Generate an explanation of a variant's significance and effects.
    
    Args:
        client: Initialized Groq client
        variant_info: Dictionary with variant details
        format_type: Type of explanation to generate ("brief", "detailed", or "clinical")
        model: Model to use for generation
        
    Returns:
        Dictionary with the explanation and metadata
    """
    try:
        # Create a variant description from the variant info
        variant_desc = f"{variant_info.get('variant', 'Unknown variant')}"
        gene = None
        
        # Extract gene information if available
        if "ensembl" in variant_info and "gene_symbol" in variant_info["ensembl"]:
            gene = variant_info["ensembl"]["gene_symbol"]
            variant_desc += f" in gene {gene}"
        
        # Determine the kind of explanation to generate
        prompt = ""
        if format_type == "brief":
            prompt = f"Provide a brief (2-3 sentences) explanation of the {variant_desc} and its potential significance."
        elif format_type == "clinical":
            prompt = f"""
            Provide a clinical interpretation of the {variant_desc} following ACMG/AMP guidelines.
            Include:
            1. Variant classification and evidence criteria
            2. Summary of pathogenicity evidence
            3. Clinical significance and disease associations
            4. Recommendations for further assessment if applicable
            """
        else:  # detailed (default)
            prompt = f"""
            Provide a detailed explanation of the {variant_desc}, including:
            1. Variant type and molecular effect
            2. Impact on protein structure/function
            3. Known disease associations
            4. Population frequency (if available)
            5. Evidence for pathogenicity or benignity
            """
        
        # Add information about predictions and database entries
        if "effect" in variant_info:
            prompt += f"\nThe EvoBeevos+ predictor classified this variant as {variant_info['effect']}."
        
        if "clinvar" in variant_info and "clinical_significance" in variant_info["clinvar"]:
            prompt += f"\nClinVar classifies this variant as {variant_info['clinvar']['clinical_significance']}."
        
        # Generate the explanation
        return generate_response(
            client=client,
            query=prompt,
            variant_info=variant_info,
            model=model
        )
        
    except Exception as e:
        # Return error information
        return {
            "response": f"I apologize, but I encountered an error while generating a variant explanation: {str(e)}",
            "error": str(e),
            "success": False
        }

def get_genetic_literature(
    client: groq.Groq,
    query: str,
    variant_info: Optional[Dict[str, Any]] = None,
    max_papers: int = 5,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Generate a summary of relevant scientific literature for a genetic variant or topic.
    
    Args:
        client: Initialized Groq client
        query: Query describing the variant or genetic topic
        variant_info: Optional dictionary with variant details
        max_papers: Maximum number of papers to include
        model: Model to use for generation
        
    Returns:
        Dictionary with the literature summary and metadata
    """
    # Create system prompt focused on literature
    literature_system_prompt = """
    You are an expert geneticist and bioinformatician with extensive knowledge of scientific literature
    on genetic variants, diseases, and molecular mechanisms. Your task is to provide a summary of
    relevant scientific literature for a given genetic variant or topic.
    
    When providing literature summaries:
    1. Focus on high-quality, peer-reviewed research
    2. Include recent publications (when available)
    3. Provide key findings from each paper
    4. Include author, journal, and year for each publication
    5. Prioritize clinical studies and functional analyses
    6. Be honest when literature is sparse for a given topic
    """
    
    # Enhance the query to explicitly ask for literature
    enhanced_query = f"""
    Provide a summary of up to {max_papers} key scientific publications relevant to {query}.
    
    For each publication, include:
    - Author, year, and journal
    - Key findings
    - Relevance to the variant/topic
    
    Finally, provide a brief synthesis of what the literature as a whole suggests.
    """
    
    # Generate the response
    return generate_response(
        client=client,
        query=enhanced_query,
        variant_info=variant_info,
        model=model,
        system_prompt=literature_system_prompt,
        max_tokens=1500  # Longer response for literature review
    )

def get_mock_response(
    query: str,
    variant_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate a mock response when the AI service is unavailable.
    
    Args:
        query: User's question
        variant_info: Optional dictionary with variant details
        
    Returns:
        Dictionary with a mock response
    """
    # Basic response indicating this is a mock
    mock_response = {
        "response": (
            "This is a simulated response as the AI service is currently unavailable. "
            "In a production environment, this would connect to the Groq API to provide "
            "expert-level analysis of genetic variants and answer your questions."
        ),
        "model": "Mock Model",
        "tokens": 0,
        "success": True,
        "mock": True
    }
    
    # If we have variant info, add some basic information to make it slightly more useful
    if variant_info and "variant" in variant_info:
        variant = variant_info["variant"]
        effect = variant_info.get("effect", "unknown effect")
        
        mock_response["response"] += f"\n\nYou asked about variant {variant}, which has a predicted {effect}."
        
        if "clinvar" in variant_info and "clinical_significance" in variant_info["clinvar"]:
            significance = variant_info["clinvar"]["clinical_significance"]
            mock_response["response"] += f" ClinVar classifies this variant as {significance}."
            
        mock_response["response"] += (
            "\n\nTo get a detailed analysis, please ensure your Groq API key is properly configured "
            "in the application settings."
        )
    
    return mock_response 