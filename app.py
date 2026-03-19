"""
InterSpectrum - Visual Interface
Reading the internal spectrum of AI models.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from spectrum_analyzer import SpectrumAnalyzer
from louise import Louise


st.set_page_config(
    page_title="InterSpectrum",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 InterSpectrum")
st.caption("Reading the internal spectrum of AI models")

@st.cache_resource
def load_analyzer():
    return SpectrumAnalyzer()

@st.cache_resource
def load_louise():
    try:
        return Louise()
    except Exception:
        return None

analyzer = load_analyzer()
louise = load_louise()

# Mode selector
mode = st.radio(
    "Mode",
    ["Single Text", "Compare Two Texts", "Louise Memory"],
    horizontal=True
)

st.markdown("---")

def show_spectrum(spectrum, label=""):
    if label:
        st.subheader(label)

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Size", f"{spectrum['size']:.3f}",
            help="Semantic space occupied")
    with col2:
        st.metric("Symmetry", f"{spectrum['symmetry']:.3f}",
            help="Dimensional balance")
    with col3:
        st.metric("Dispersion", f"{spectrum['dispersion']:.3f}",
            help="The peaks - scatter from center")
    with col4:
        st.metric("Density", f"{spectrum['density']:.3f}",
            help="Gravitational center strength")
    with col5:
        st.metric("Dimensions", spectrum['shape']['dominant_dimensions'],
            help="Dominant dimensions carrying 90% of meaning")

    coords = np.array(spectrum['coords_3d'])
    sentences = spectrum['sentences']

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers+text',
        marker=dict(
            size=8,
            color=coords[:, 2],
            colorscale='Viridis',
            opacity=0.8
        ),
        text=[f"{i+1}" for i in range(len(sentences))],
        hovertext=sentences,
        hoverinfo='text'
    ))

    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            fig.add_trace(go.Scatter3d(
                x=[coords[i, 0], coords[j, 0]],
                y=[coords[i, 1], coords[j, 1]],
                z=[coords[i, 2], coords[j, 2]],
                mode='lines',
                line=dict(color='rgba(100,200,255,0.2)', width=1),
                hoverinfo='none',
                showlegend=False
            ))

    fig.update_layout(
        scene=dict(
            bgcolor='rgb(10,10,30)',
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            zaxis=dict(showgrid=False, zeroline=False)
        ),
        paper_bgcolor='rgb(10,10,30)',
        height=500,
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)


# SINGLE MODE
if mode == "Single Text":
    text = st.text_area(
        "Enter text to read its internal spectrum",
        height=200,
        placeholder="Type or paste any text here..."
    )
    
    label = st.text_input(
        "Label (optional)",
        placeholder="e.g. Shakespeare Sonnet 18, News article, Technical doc..."
    )

    if st.button("Read Spectrum", type="primary"):
        if text.strip():
            with st.spinner("Reading internal spectrum..."):
                spectrum = analyzer.extract_spectrum(text)
            if spectrum:
                st.markdown("---")
                st.subheader("Internal Structure")
                st.caption("The figure that forms before words appear")
                show_spectrum(spectrum)

                # Save to Louise
                if louise and label.strip():
                    try:
                        louise.remember(label, text, spectrum)
                        st.success(f"💾 Louise remembered: {label}")
                    except Exception as e:
                        st.error(f"Louise error: {e}")
                elif not label.strip():
                    st.info("Add a label to save this spectrum to Louise's memory.")
        else:
            st.warning("Please enter some text first.")

# COMPARE MODE
elif mode == "Compare Two Texts":
    col_a, col_b = st.columns(2)
    
    with col_a:
        text_a = st.text_area("Text A", height=200, placeholder="First text...")
    
    with col_b:
        text_b = st.text_area("Text B", height=200, placeholder="Second text...")

    if st.button("Compare Spectrums", type="primary"):
        if text_a.strip() and text_b.strip():
            with st.spinner("Reading both spectrums..."):
                result = analyzer.compare(text_a, text_b)
            
            if result:
                st.markdown("---")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    show_spectrum(result['spectrum_a'], "Text A — Internal Structure")
                
                with col_b:
                    show_spectrum(result['spectrum_b'], "Text B — Internal Structure")
                
                st.markdown("---")
                st.subheader("Spectral Differences")
                st.caption("Where the two internal structures diverge")
                
                d_col1, d_col2, d_col3, d_col4 = st.columns(4)
                
                with d_col1:
                    st.metric("Size Δ", f"{result['size_difference']:.3f}")
                with d_col2:
                    st.metric("Symmetry Δ", f"{result['symmetry_difference']:.3f}")
                with d_col3:
                    st.metric("Dispersion Δ", f"{result['dispersion_difference']:.3f}")
                with d_col4:
                    st.metric("Density Δ", f"{result['density_difference']:.3f}")
        else:
            st.warning("Please enter both texts first.")

# LOUISE MEMORY
else:
    st.subheader("💾 Louise Memory")
    st.caption("Everything Louise has remembered so far")

    if louise:
        try:
            records = louise.recall()
            if records:
                import pandas as pd
                df = pd.DataFrame(records)
                st.dataframe(df, use_container_width=True)
                st.caption(f"{len(records)} spectrums in memory")
            else:
                st.info("Louise hasn't remembered anything yet. Analyze a text with a label to start building the dictionary.")
        except Exception as e:
            st.error(f"Louise couldn't recall: {e}")
    else:
        st.warning("Louise is not connected. Credentials file needed.")