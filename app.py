import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import json
from groq import Groq
from typing import Generator
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add src folder to the path to import our modules
sys.path.append(str(Path(__file__).parent))
from src.clinvar.clinvar_api import get_clinvar_data, parse_clinvar_conditions
from src.variant_predictor.evo_predictor import predict_variant_effect
from src.variant_predictor.ensembl_api import query_ensembl_vep
from src.utils.visualization import plot_prediction_summary

# Application title and description
st.set_page_config(
    page_title="GenetiXplorer Variant Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("GenetiXplorer Variant Predictor 🧬")

# About section
with st.expander("**About GenetiXplorer**"):
    st.write("""
    GenetiXplorer is an advanced genetic variant effect predictor that uses AI models to predict 
    variant effects and compares these with established databases. This tool helps researchers
    and clinicians understand the potential impact of genetic variants on protein function and disease.
    
    Key features:
    - Multiple variant input formats
    - AI-powered effect prediction
    - Integration with ClinVar and Ensembl
    - Detailed visualization
    - AI assistant for result interpretation
    """)
    
    # Display workflow image if available
    try:
        st.image("images/workflow.jpg", 
                 caption="GenetiXplorer Workflow",
                 use_column_width=True)
    except:
        st.info("Workflow diagram not available. Add an image to the 'images' folder to display here.")

# Sidebar for settings and options
with st.sidebar:
    st.header("Settings")
    
    # Select variant input method
    input_method = st.radio(
        "Input Method",
        ["Chromosome Position", "HGVS Notation", "VCF Upload"]
    )
    
    # Model selection
    model_option = st.selectbox(
        "Select Prediction Model",
        ["GenX 2.0 (Recommended)", "Ensembl VEP Only", "Combined Models"]
    )
    
    # Reference genome
    genome_build = st.radio(
        "Reference Genome",
        ["GRCh38 (hg38)", "GRCh37 (hg19)"]
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        window_size = st.slider("Sequence Window Size", 512, 4096, 2048, 512)
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        include_conservation = st.checkbox("Include Conservation Scores", True)
        show_raw_data = st.checkbox("Show Raw Data", False)

# Main content area
if input_method == "Chromosome Position":
    st.header("Variant Input")
    
    with st.form("variant-form"):
        col1, col2 = st.columns(2)
        
        with col1:
            chromosome = st.selectbox(
                "Chromosome",
                [str(i) for i in range(1, 23)] + ["X", "Y", "MT"]
            )
            position = st.number_input("Position", min_value=1, value=41276133)
            
        with col2:
            ref_allele = st.selectbox("Reference Allele", ["A", "C", "G", "T"])
            alt_allele = st.selectbox("Alternate Allele", ["A", "C", "G", "T"])
            gene_symbol = st.text_input("Gene Symbol (optional)", "BRCA1")
        
        submit_button = st.form_submit_button("Predict Variant Effect")
    
    if submit_button:
        # Display a loading spinner while processing
        with st.spinner("Processing variant..."):
            # Format the variant for display
            variant_display = f"chr{chromosome}:{position}{ref_allele}>{alt_allele}"
            st.write(f"**Analyzing variant:** {variant_display}")
            
            # Create tabs for different result sections
            tab1, tab2, tab3, tab4 = st.tabs(["Prediction Summary", "ClinVar Data", "Ensembl Data", "Technical Details"])
            
            with tab1:
                st.subheader("Variant Effect Prediction")
                
                # todo:  connect to actual prediction in production
                #  demo sample data
                prediction_result = {
                    "model": "GenX 2.0",
                    "effect": "Likely Pathogenic",
                    "confidence": 0.89,
                    "delta_score": 0.753,
                    "conservation": 0.92
                }
                
                # Display prediction summary with a nice visual
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**Prediction**: {prediction_result['effect']}")
                    st.markdown(f"**Confidence**: {prediction_result['confidence']:.2f}")
                    st.markdown(f"**Delta Score**: {prediction_result['delta_score']:.3f}")
                    if include_conservation:
                        st.markdown(f"**Conservation**: {prediction_result['conservation']:.2f}")
                
                with col2:
                    # Sample visualization
                    fig, ax = plt.subplots(figsize=(8, 3))
                    scores = [0.1, 0.3, 0.2, 0.89, 0.4]
                    labels = ["Benign", "Likely Benign", "VUS", "Likely Pathogenic", "Pathogenic"]
                    colors = ["green", "lightgreen", "gray", "orange", "red"]
                    
                    # Highlight the predicted category
                    highlighted = [0.7 if i == 3 else 0.3 for i in range(5)]
                    
                    ax.bar(labels, scores, color=colors, alpha=highlighted)
                    ax.set_ylim(0, 1)
                    ax.set_ylabel("Score")
                    ax.set_title("Variant Classification")
                    st.pyplot(fig)
            
            with tab2:
                st.subheader("ClinVar Data")
                
                # TODO:production, call the actual ClinVar API
                # demo sample data
                clinvar_conditions = [
                    "Hereditary breast and ovarian cancer syndrome",
                    "Breast-ovarian cancer, familial, susceptibility to, 1",
                    "Hereditary cancer-predisposing syndrome"
                ]
                
                clinvar_classifications = {
                    "Pathogenic": 42,
                    "Likely pathogenic": 15,
                    "Uncertain significance": 7,
                    "Likely benign": 2,
                    "Benign": 1
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Related Conditions")
                    for i, condition in enumerate(clinvar_conditions, 1):
                        st.markdown(f"{i}. {condition}")
                
                with col2:
                    st.markdown("### ClinVar Classifications")
                    fig, ax = plt.subplots()
                    ax.pie(
                        clinvar_classifications.values(), 
                        labels=clinvar_classifications.keys(),
                        autopct='%1.1f%%',
                        colors=["red", "orange", "gray", "lightgreen", "green"]
                    )
                    ax.axis('equal')
                    st.pyplot(fig)
            
            with tab3:
                st.subheader("Ensembl VEP Data")
                
                # TODO: call the  Ensembl API
                # demo sample data
                ensembl_data = {
                    "gene_id": "ENSG00000012048",
                    "gene_symbol": "BRCA1",
                    "consequence": "missense_variant",
                    "impact": "MODERATE",
                    "protein_position": 1175,
                    "amino_acids": "M/V",
                    "codons": "aTg/gTg",
                    "canonical": "YES",
                    "sift_score": 0.02,
                    "sift_prediction": "deleterious",
                    "polyphen_score": 0.897,
                    "polyphen_prediction": "probably_damaging"
                }
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Variant Information")
                    st.markdown(f"**Gene**: {ensembl_data['gene_symbol']} ({ensembl_data['gene_id']})")
                    st.markdown(f"**Consequence**: {ensembl_data['consequence']}")
                    st.markdown(f"**Impact**: {ensembl_data['impact']}")
                    st.markdown(f"**Protein Change**: p.{ensembl_data['amino_acids'].replace('/', '')}{ensembl_data['protein_position']}")
                    st.markdown(f"**Codon Change**: {ensembl_data['codons']}")
                
                with col2:
                    st.markdown("### Prediction Scores")
                    
                    # SIFT score visualization
                    st.markdown("**SIFT Prediction**: " + ensembl_data['sift_prediction'])
                    sift_color = "red" if ensembl_data['sift_prediction'] == "deleterious" else "green"
                    st.progress(1 - ensembl_data['sift_score'], text=f"Score: {ensembl_data['sift_score']}")
                    
                    # PolyPhen score visualization
                    st.markdown("**PolyPhen Prediction**: " + ensembl_data['polyphen_prediction'])
                    st.progress(ensembl_data['polyphen_score'], text=f"Score: {ensembl_data['polyphen_score']}")
            
            with tab4:
                st.subheader("Technical Details")
                
                if show_raw_data:
                    st.json({
                        "variant": variant_display,
                        "prediction": prediction_result,
                        "ensembl_data": ensembl_data,
                        "clinvar_data": {
                            "conditions": clinvar_conditions,
                            "classifications": clinvar_classifications
                        }
                    })
                
                st.markdown("### Sequence Context")
                sequence_context = "ACGTACGT" + ref_allele + "ACGTACGT"
                highlighted_sequence = sequence_context.replace(ref_allele, f"<span style='color:red'>{ref_allele}</span>")
                st.markdown(f"...{highlighted_sequence}...", unsafe_allow_html=True)
                
                st.markdown("### Model Information")
                st.markdown(f"**Model**: {model_option}")
                st.markdown(f"**Window Size**: {window_size} bp")
                st.markdown(f"**Reference Genome**: {genome_build}")

elif input_method == "HGVS Notation":
    st.header("HGVS Notation Input")
    
    with st.form("hgvs-form"):
        hgvs_input = st.text_input("HGVS Notation", "NM_007294.3:c.3472G>A")
        submit_button = st.form_submit_button("Predict Variant Effect")
    
    if submit_button:
        st.info("HGVS processing functionality will be implemented in the next version.")

elif input_method == "VCF Upload":
    st.header("VCF File Upload")
    
    uploaded_file = st.file_uploader("Upload VCF file", type=["vcf"])
    
    if uploaded_file:
        st.info("VCF processing functionality will be implemented in the next version.")

# AI Chatbot
st.header("GenetiXplorer Assistant 🤖")

# Initialize chat client if API key available
try:
    if "GROQ_API_KEY" in st.secrets:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    else:
        st.warning("Groq API key not found. Chat functionality disabled.")
        client = None
except:
    st.warning("Error initializing Groq client. Chat functionality disabled.")
    client = None

# System message defining the chatbot's behavior
system_message = {
    "role": "system",
    "content": (
        "You are GenetiXplorer, an AI expert in genetic variant analysis. "
        "Help users understand genetic mutations by analyzing variant data. "
        "You can provide information on genes, their function, and the effects of mutations. "
        "When analyzing variants, follow this process: "
        "1. Identify the gene and its biological function. "
        "2. Analyze the effect of the mutation on the protein. "
        "3. Check conservation across species if data is available. "
        "4. Reference databases like ClinVar or Ensembl for known classifications. "
        "5. Conclude with a classification and justification. "
        "Provide expert-level information that is accurate and concise."
    )
}

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [system_message]

# Display chat messages
for message in st.session_state.messages:
    if message["role"] != "system":  # Don't display system messages
        avatar = '🤖' if message["role"] == "assistant" else '👤'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# Function to generate chat responses
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Chat input
if prompt := st.chat_input("Ask about genetic variants..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user", avatar='👤'):
        st.markdown(prompt)
    
    if client:
        try:
            with st.chat_message("assistant", avatar="🤖"):
                # Default model
                model = "llama-3.1-8b-instant"
                
                chat_completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": m["role"],
                            "content": m["content"]
                        }
                        for m in st.session_state.messages
                    ],
                    max_tokens=4096,
                    temperature=0.7,
                    top_p=0.9,
                    stream=True
                )
                
                # Stream the response
                chat_responses_generator = generate_chat_responses(chat_completion)
                full_response = st.write_stream(chat_responses_generator)
                
                # Add response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
        except Exception as e:
            st.error(f"Error: {e}", icon="🚨")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            error_message = "I'm sorry, but the chat service is currently unavailable. Please check your API configuration."
            st.markdown(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message}
            )

# Footer
st.divider()
st.caption("GenetiXplorer Variant Predictor © 2025 | Built with Streamlit and cursor ") 