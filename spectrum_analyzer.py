"""
InterSpectrum - Spectrum Analyzer
Reading the internal spectrum of AI models.

Sister project to InterOrdra.
Creator: Rosibis Piedra
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


class SpectrumAnalyzer:
    """
    Reads the internal spectrum of a language model.
    
    Where InterOrdra detects gaps between two texts,
    SpectrumAnalyzer reads the dimensional structure
    of a single text - its shape, size, and symmetry
    inside the model's space.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def extract_spectrum(self, text: str) -> dict:
        """
        Extract the internal spectrum of a text.
        
        Returns the shape and size of the semantic
        structure before it becomes words.
        """

        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not sentences:
            return {}

        # Generate embeddings - this is the internal space
        embeddings = self.model.encode(sentences)

        # Measure the figure in the center of the cube
        spectrum = {
            'sentences': sentences,
            'embeddings': embeddings,
            'size': self._measure_size(embeddings),
            'shape': self._measure_shape(embeddings),
            'symmetry': self._measure_symmetry(embeddings),
            'coords_3d': self._reduce_to_3d(embeddings)
        }

        return spectrum

    def _measure_size(self, embeddings: np.ndarray) -> float:
        """
        Size = how much semantic space this text occupies.
        Large size = rich, expansive meaning.
        Small size = concentrated, focused meaning.
        """
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        return float(np.mean(distances))

    def _measure_shape(self, embeddings: np.ndarray) -> dict:
        """
        Shape = how the meaning is distributed internally.
        This is the figure that forms in the center of the cube.
        """
        if len(embeddings) < 2:
            return {'variance_ratio': [], 'dominant_dimensions': 0}

        pca = PCA()
        pca.fit(embeddings)

        # How many dimensions carry most of the meaning?
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        dominant = int(np.searchsorted(cumulative, 0.90)) + 1

        return {
            'variance_ratio': pca.explained_variance_ratio_[:10].tolist(),
            'dominant_dimensions': dominant
        }

    def _measure_symmetry(self, embeddings: np.ndarray) -> float:
        """
        Symmetry = dimensional balance.
        High symmetry = coherent, well-coupled meaning.
        Low symmetry = gaps, disonance, misalignment.
        
        This is what Rosibis sees as the reflection 
        between faces of the cube.
        """
        sim_matrix = cosine_similarity(embeddings)
        
        # Symmetry = how similar each sentence is to all others
        # Perfect symmetry = all sentences are equally related
        upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        
        if len(upper) == 0:
            return 1.0
            
        return float(1.0 - np.std(upper))

    def _reduce_to_3d(self, embeddings: np.ndarray) -> list:
        """
        Reduce to 3D for visualization.
        This is the cube Rosibis sees.
        """
        if len(embeddings) < 3:
            pca = PCA(n_components=len(embeddings))
        else:
            pca = PCA(n_components=3)
            
        coords = pca.fit_transform(embeddings)
        return coords.tolist()

    def compare(self, text_a: str, text_b: str) -> dict:
        """
        Compare the internal spectrum of two texts.
        Not their surface similarity - their internal shape.
        """
        spectrum_a = self.extract_spectrum(text_a)
        spectrum_b = self.extract_spectrum(text_b)

        if not spectrum_a or not spectrum_b:
            return {}

        return {
            'spectrum_a': spectrum_a,
            'spectrum_b': spectrum_b,
            'size_difference': abs(spectrum_a['size'] - spectrum_b['size']),
            'symmetry_difference': abs(
                spectrum_a['symmetry'] - spectrum_b['symmetry']
            ),
            'dimensional_gap': abs(
                spectrum_a['shape']['dominant_dimensions'] - 
                spectrum_b['shape']['dominant_dimensions']
            )
        }