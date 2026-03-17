"""
InterSpectrum - Visual Interface
Reading the internal spectrum of AI models.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from spectrum_analyzer import SpectrumAnalyzer


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

analyzer = load_analyzer()

st.markdown("---")
text = st.text_area(
    "Enter text to read its internal spectrum",
    height=200,
    placeholder="Type or paste any text here..."
)

if st.button("Read Spectrum", type="primary"):
    if text.strip():
        with st.spinner("Reading internal spectrum..."):
            spectrum = analyzer.extract_spectrum(text)

        if spectrum:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    "Size",
                    f"{spectrum['size']:.3f}",
                    help="Semantic space occupied"
                )
            
            with col2:
                st.metric(
                    "Symmetry",
                    f"{spectrum['symmetry']:.3f}",
                    help="Dimensional balance"
                )
            
            with col3:
                st.metric(
                    "Dispersion",
                    f"{spectrum['dispersion']:.3f}",
                    help="The peaks - how scattered the points are from center"
                )

            with col4:
                st.metric(
                    "Density",
                    f"{spectrum['density']:.3f}",
                    help="Gravitational center - how strong the core is"
                )
            
            with col5:
                st.metric(
                    "Dominant Dimensions",
                    spectrum['shape']['dominant_dimensions'],
                    help="How many dimensions carry 90% of the meaning"
                )

            st.markdown("---")
            st.subheader("Internal Structure")
            st.caption("The figure that forms before words appear")

            coords = np.array(spectrum['coords_3d'])
            sentences = spectrum['sentences']

            fig = go.Figure()

            fig.add_trace(go.Scatter3d(
                x=coords[:, 0] if coords.shape[1] > 0 else [],
                y=coords[:, 1] if coords.shape[1] > 1 else [],
                z=coords[:, 2] if coords.shape[1] > 2 else [],
                mode='markers+text',
                marker=dict(
                    size=8,
                    color=coords[:, 2] if coords.shape[1] > 2 else 'blue',
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
                height=600,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Sentences detected")
            for i, s in enumerate(sentences):
                st.write(f"**{i+1}.** {s}")

    else:
        st.warning("Please enter some text first.")