import os
from openai import OpenAI
import pandas as pd
import tiktoken
from scipy import spatial
import streamlit as st
from typing import Optional, Tuple, List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
GPT_MODELS = ["gpt-3.5-turbo", "gpt-4"]
EMBEDDING_MODEL = "text-embedding-ada-002"
TOKEN_BUDGET = 4096 - 500
MIN_SIMILARITY_SCORE = 0.5

class TaxAnalysisError(Exception):
    """Custom exception for tax analysis errors"""
    pass

@st.cache_resource
def load_data(data_path: str = "tax_embeddings.json") -> Optional[pd.DataFrame]:
    """Loads and caches tax legislation embeddings data."""
    try:
        df = pd.read_json(data_path)
        logger.info(f"Successfully loaded {len(df)} records from {data_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def preprocess_query(query: str) -> str:
    """Enhances the query with context and standardizes terminology."""
    enhanced_query = f"Australian tax legislation regarding {query}"
    tax_terms_mapping = {
        "CGT": "Capital Gains Tax",
        "GST": "Goods and Services Tax",
        "FBT": "Fringe Benefits Tax",
    }
    for term, replacement in tax_terms_mapping.items():
        enhanced_query = enhanced_query.replace(term, replacement)
    return enhanced_query

def get_relevant_sections(
    query: str,
    df: pd.DataFrame,
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 5,
    similarity_threshold: float = MIN_SIMILARITY_SCORE
) -> Tuple[List[Dict], List[float]]:
    """Retrieves relevant legislative sections based on query similarity."""
    try:
        query_embedding_response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query,
        )
        query_embedding = query_embedding_response.data[0].embedding

        sections_and_scores = [
            ({
                'content': row["text"],
                'title': row['title'],
                'url': row['url'],
                'source': row['source'],
                'section_number': row['section_number']
            }, relatedness_fn(query_embedding, row["embedding"]))
            for _, row in df.iterrows()
        ]

        sections_and_scores = [
            (section, score) for section, score in sections_and_scores
            if score >= similarity_threshold
        ]

        sections_and_scores.sort(key=lambda x: x[1], reverse=True)
        top_sections = sections_and_scores[:top_n]

        if not top_sections:
            return [], []

        sections, scores = zip(*top_sections)
        return list(sections), list(scores)

    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        raise TaxAnalysisError(f"Failed to retrieve relevant sections: {str(e)}")

def create_analysis_prompt(query: str, relevant_sections: List[Dict]) -> str:
    """Creates a detailed prompt for legislative analysis."""
    sections_text = "\n\n".join([
        f"Section {section['section_number']} from {section['source']} - {section['title']}:\n{section['content']}"
        for section in relevant_sections
    ])

    return f"""As an expert tax legislation analyst, provide a comprehensive and insightful analysis of the following query based on Australian tax law:

Query: {query}

Relevant Legislative Sections:
{sections_text}

Please provide a clear and structured response that:
1. Directly answers the query using the provided legislative sections
2. Explains key concepts and requirements
3. References specific sections when discussing provisions
4. Highlights any important conditions or exceptions
5. Notes any limitations in the available information

Base your analysis primarily on the provided sections. If the sections don't fully address the query, acknowledge this limitation."""

def generate_response(
    query: str,
    df: pd.DataFrame,
    model: str = "gpt-4",
    top_n: int = 8
) -> Tuple[str, List[Dict]]:
    """Generates a comprehensive response to the tax query."""
    try:
        enhanced_query = preprocess_query(query)
        relevant_sections, scores = get_relevant_sections(
            enhanced_query,
            df,
            top_n=top_n
        )

        if not relevant_sections:
            return ("I apologize, but I couldn't find sufficiently relevant legislative sections to provide an accurate response. "
                   "Please try rephrasing your query or consult a tax professional."), []

        prompt = create_analysis_prompt(query, relevant_sections)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": """You are an expert Australian tax legislation analyst.
                Provide clear, authoritative analysis based strictly on the provided legislative sections.
                Always cite specific sections and maintain professional tone."""},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )

        filtered_sections = [
            section for section, score in zip(relevant_sections, scores)
            if score >= MIN_SIMILARITY_SCORE
        ]

        return response.choices[0].message.content, filtered_sections

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise TaxAnalysisError(f"Failed to generate response: {str(e)}")

def main():
    st.set_page_config(
        page_title="Tax Legislation Reference",
        page_icon="⚖️",
        layout="wide"
    )

    st.title("⚖️ Australian Tax Legislation Reference")

    # API Key handling
    api_key = st.secrets["OPENAI_API_KEY"]
    if not api_key:
        api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if not api_key:
            st.error("Please set OPENAI_API_KEY environment variable or enter your API key!")
            st.stop()

    # Initialize OpenAI client
    global client
    client = OpenAI(api_key=api_key)

    # Load data with error handling
    df = load_data()
    if df is None:
        st.error("Failed to load embeddings file. Please check if 'tax_embeddings.json' exists.")
        return

    # Query interface
    query = st.text_area(
        "Enter your tax legislation query:",                                                     
        placeholder="e.g., What are the specific requirements for claiming deductions on prepaid expenses?"
    )

    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            num_results = st.slider("Number of relevant sections:", 1, 10, 5)
        with col2:
            similarity_threshold = st.slider("Minimum similarity score:", 0.0, 1.0, MIN_SIMILARITY_SCORE)
        with col3:
            model_choice = st.selectbox("Model", GPT_MODELS, index=1)

    if st.button("Search Legislation", type="primary"):
        if not query:
            st.warning("Please enter a query.")
            return

        try:
            with st.spinner("Searching legislation..."):
                response, relevant_sections = generate_response(
                    query,
                    df,
                    model=model_choice,
                    top_n=num_results
                )

            # Display analysis
            st.markdown("### Analysis")
            st.markdown(response)

            # Display legislative references
            st.markdown("### Legislative References")
            for section in relevant_sections:
                with st.expander(f"{section['source']} - Section {section['section_number']}"):
                    st.markdown(f"**{section['title']}**")
                    st.markdown(f"```{section['content']}```")
                    st.markdown(f"[View Full Section Online]({section['url']})")

        except TaxAnalysisError as e:
            st.error(f"Analysis Error: {str(e)}")
        except Exception as e:
            st.error(f"Unexpected Error: {str(e)}")

    # Sidebar information
    with st.sidebar:
        st.markdown("### About This Tool")
        st.markdown("""
        This tool provides:
        - Search functionality for tax legislation
        - References to relevant sections
        - Links to source legislation
        - AI-assisted interpretation of legislative sections

        **Coverage:**
        - Income Tax Assessment Act 1936
        - Income Tax Assessment Act 1997
        """)

        st.markdown("### Important Disclaimer")
        st.markdown("""
        **Please Note:**

        - This tool uses AI to search through embeddings of tax legislation and generate responses
        - It is not pre-trained on tax law and only searches for close matches in the available text
        - Responses may not always be accurate or complete
        - Only covers Income Tax Assessment Acts 1936 and 1997
        - Does not include other tax legislation, rulings, or case law
        - Should not be relied upon for tax or legal advice
        - Always consult with a qualified tax professional for specific advice
        """)

if __name__ == "__main__":
    main()

# Created/Modified files during execution:
# Reads from: tax_embeddings.json
