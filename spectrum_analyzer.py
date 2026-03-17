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
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not sentences:
            return {}

        embeddings = self.model.encode(sentences)

        spectrum = {
            'sentences': sentences,
            'embeddings': embeddings,
            'size': self._measure_size(embeddings),
            'shape': self._measure_shape(embeddings),
            'symmetry': self._measure_symmetry(embeddings),
            'dispersion': self._measure_dispersion(embeddings),
            'density': self._measure_density(embeddings),
            'coords_3d': self._reduce_to_3d(embeddings)
        }

        return spectrum

    def _measure_size(self, embeddings: np.ndarray) -> float:
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        return float(np.mean(distances))

    def _measure_shape(self, embeddings: np.ndarray) -> dict:
        if len(embeddings) < 2:
            return {'variance_ratio': [], 'dominant_dimensions': 0}

        pca = PCA()
        pca.fit(embeddings)

        cumulative = np.cumsum(pca.explained_variance_ratio_)
        dominant = int(np.searchsorted(cumulative, 0.90)) + 1

        return {
            'variance_ratio': pca.explained_variance_ratio_[:10].tolist(),
            'dominant_dimensions': dominant
        }

    def _measure_symmetry(self, embeddings: np.ndarray) -> float:
        sim_matrix = cosine_similarity(embeddings)
        upper = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        
        if len(upper) == 0:
            return 1.0
            
        return float(1.0 - np.std(upper))

    def _measure_dispersion(self, embeddings: np.ndarray) -> float:
        """
        Dispersion = the peaks Rosibis sees.
        High dispersion = points scattered far from center = spiky figure.
        Low dispersion = points close to center = dense, coherent figure.
        """
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        
        # Dispersion = how much variation in distances from center
        # High std = some points very far, some very close = peaks
        return float(np.std(distances))

    def _measure_density(self, embeddings: np.ndarray) -> float:
        """
        Density = does the figure have a clear center?
        High density = strong gravitational center, coherent meaning.
        Low density = no center, dispersed meaning.
        """
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        
        # Density = inverse of mean distance from center
        # Close to center = high density
        mean_dist = np.mean(distances)
        if mean_dist == 0:
            return 1.0
        return float(1.0 / (1.0 + mean_dist))

    def _reduce_to_3d(self, embeddings: np.ndarray) -> list:
        if len(embeddings) < 3:
            pca = PCA(n_components=len(embeddings))
        else:
            pca = PCA(n_components=3)
            
        coords = pca.fit_transform(embeddings)
        return coords.tolist()

    def compare(self, text_a: str, text_b: str) -> dict:
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
            'dispersion_difference': abs(
                spectrum_a['dispersion'] - spectrum_b['dispersion']
            ),
            'density_difference': abs(
                spectrum_a['density'] - spectrum_b['density']
            ),
            'dimensional_gap': abs(
                spectrum_a['shape']['dominant_dimensions'] - 
                spectrum_b['shape']['dominant_dimensions']
            )
        }