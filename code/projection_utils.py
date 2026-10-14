import numpy as np
from typing import Any, List
from deap import base as deap_base, creator as deap_creator, tools as deap_tools, algorithms as deap_algorithms
import random
import time
import logging

base: Any = deap_base
creator: Any = deap_creator
tools: Any = deap_tools
algorithms: Any = deap_algorithms

def orthogonal_projection(embedding: np.ndarray, concept_embeddings: List[np.ndarray]) -> np.ndarray:
    """Project an embedding onto the subspace orthogonal to the concepts."""
    if not concept_embeddings:
        return embedding

    Q, _ = np.linalg.qr(np.array(concept_embeddings).T)
    projection_matrix = np.eye(embedding.shape[0]) - Q @ Q.T
    return projection_matrix @ embedding

def matrix_projection(embedding: np.ndarray, concept_matrix: np.ndarray) -> np.ndarray:
    """Project an embedding onto the subspace orthogonal to a concept matrix."""
    # concept_matrix is expected to have concepts as columns.
    # If it is shaped (n_concepts, embedding_dim), transpose it.
    if concept_matrix.shape[0] < concept_matrix.shape[1]:
        concept_matrix = concept_matrix.T
    
    Q, _ = np.linalg.qr(concept_matrix)
    projection_matrix = np.eye(embedding.shape[0]) - Q @ Q.T
    return projection_matrix @ embedding

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def genetic_algorithm_optimization(
    first_message_embedding: np.ndarray,
    long_embedding: np.ndarray,
    concept_embeddings: List[np.ndarray],
    population_size: int = 50,
    generations: int = 20
) -> List[float]:
    """Optimize concept weights with a genetic algorithm."""
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox: Any = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(concept_embeddings))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        weights = np.array(individual)
        weighted_concepts = [concept * weight for concept, weight in zip(concept_embeddings, weights)]
        projected_embedding = orthogonal_projection(long_embedding, weighted_concepts)
        similarity = cosine_similarity(first_message_embedding, projected_embedding)
        return (similarity,)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    population = toolbox.population(n=population_size)
    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=generations, verbose=False)

    best_individual = tools.selBest(population, k=1)[0]
    return best_individual


def cosine_similarity_batch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between A and B."""
    t0 = time.time()
    dot_products = np.sum(A * B, axis=1)
    norms_A = np.linalg.norm(A, axis=1)
    norms_B = np.linalg.norm(B, axis=1)
    similarities = dot_products / (norms_A * norms_B + 1e-10)

    elapsed = time.time() - t0
    logging.debug(f"cosine_similarity_batch: {len(A)} pairs in {elapsed:.3f}s")

    return similarities


def orthogonal_projection_batch(vectors: np.ndarray, concept: np.ndarray) -> np.ndarray:
    """Project every row of vectors onto the subspace orthogonal to concept."""
    t0 = time.time()
    dot_products = np.dot(vectors, concept)
    concept_norm_sq = np.dot(concept, concept)
    scalars = dot_products / (concept_norm_sq + 1e-10)
    projected = vectors - np.outer(scalars, concept)

    elapsed = time.time() - t0
    logging.debug(f"orthogonal_projection_batch: {len(vectors)} vectors in {elapsed:.3f}s")

    return projected


def compute_projection_load(vectors: np.ndarray, concept: np.ndarray) -> dict:
    """
    Compute projection load diagnostics when projecting `vectors` orthogonally to `concept`.

    Returns a dict with per-row absolute and relative norm drops and summary statistics.
    This helper is intentionally small and pure-numpy so tests can reason about
    raw vs normalized behaviour without relying on torch.
    """
    if vectors.size == 0:
        return {
            "absolute_drop": np.array([], dtype=float),
            "relative_drop": np.array([], dtype=float),
            "mean_abs_drop": 0.0,
            "mean_rel_drop": 0.0,
        }

    projected = orthogonal_projection_batch(vectors, concept)
    norms_before = np.linalg.norm(vectors, axis=1)
    norms_after = np.linalg.norm(projected, axis=1)
    absolute_drop = norms_before - norms_after
    # avoid division by zero
    relative_drop = absolute_drop / (norms_before + 1e-12)
    return {
        "absolute_drop": absolute_drop,
        "relative_drop": relative_drop,
        "mean_abs_drop": float(np.mean(absolute_drop)) if absolute_drop.size else 0.0,
        "mean_rel_drop": float(np.mean(relative_drop)) if relative_drop.size else 0.0,
    }
