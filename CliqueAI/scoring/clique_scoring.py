from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np
from CliqueAI.graph.model import LambdaGraph


class CliqueScoreCalculator:
    """Compute optimality and diversity scores for maximum clique responses."""

    def __init__(
        self,
        graph: LambdaGraph,
        difficulty: float,
        responses: List[List[int]],
    ) -> None:
        """
        Args:
            graph: Graph used for clique validation.
            difficulty: Difficulty multiplier for rewards.
            responses: Node sets returned by miners.
        """
        self.graph = graph
        self.difficulty = difficulty
        self.responses = responses

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def is_valid_maximum_clique(self, nodes: Sequence[int]) -> bool:
        """
        Return True if `nodes` form a valid maximum clique.
        """
        if not nodes:
            return False

        node_set = set(nodes)

        # Duplicates check
        if len(node_set) != len(nodes):
            return False

        # Range check
        if not node_set.issubset(range(self.graph.number_of_nodes)):
            return False

        # Clique check (all pairs connected)
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                if v not in self.graph.adjacency_list[u]:
                    return False

        # Maximality check (cannot be extended)
        remaining_nodes = set(range(self.graph.number_of_nodes)) - node_set
        for candidate in remaining_nodes:
            if node_set.issubset(self.graph.adjacency_list[candidate]):
                return False

        return True

    def _validity_mask(self) -> np.ndarray:
        """Return a binary mask indicating valid maximum cliques."""
        return np.array(
            [self.is_valid_maximum_clique(r) for r in self.responses],
            dtype=float,
        )

    # ------------------------------------------------------------------ #
    # Optimality
    # ------------------------------------------------------------------ #

    def optimality(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute optimality-related metrics.

        Returns:
            rel: Relative clique size
            pr: Percentile rank
            omega: Raw optimality score
            omega_normalized: Normalized optimality score
        """
        n = len(self.responses)
        if n == 0:
            zeros = np.zeros(0)
            return zeros, zeros, zeros, zeros

        validity = self._validity_mask()
        sizes = np.array([len(r) for r in self.responses], dtype=float) * validity

        max_size = sizes.max(initial=0)
        if max_size <= 0:
            zeros = np.zeros(n)
            return zeros, zeros, zeros, zeros

        rel = sizes / max_size
        pr = np.array([(sizes > sizes[i]).sum() / n for i in range(n)])

        omega = np.zeros(n)
        valid_indices = validity.astype(bool)

        omega[valid_indices] = np.exp(-pr[valid_indices] / rel[valid_indices])

        max_omega = omega.max(initial=0)
        omega_normalized = omega if max_omega == 0 else omega / max_omega

        return rel, pr, omega, omega_normalized

    # ------------------------------------------------------------------ #
    # Diversity
    # ------------------------------------------------------------------ #

    def diversity(self) -> np.ndarray:
        """
        Compute normalized diversity scores.
        """
        n = len(self.responses)
        if n == 0:
            return np.zeros(0)

        validity = self._validity_mask()

        canonical = [tuple(sorted(r)) for r in self.responses]
        counts = Counter(canonical)

        uniqueness = np.array(
            [1.0 / counts[sol] for sol in canonical],
            dtype=float,
        )

        delta = validity * uniqueness
        max_delta = delta.max(initial=0)

        return delta if max_delta == 0 else delta / max_delta

    # ------------------------------------------------------------------ #
    # Final Scores
    # ------------------------------------------------------------------ #

    def get_scores(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute final scoring components.

        Returns:
            rel: Relative size
            pr: Percentile rank
            omega: Raw optimality
            optimality: Normalized optimality
            diversity: Normalized diversity
            rewards: Final reward score
        """
        rel, pr, omega, optimality = self.optimality()
        diversity = self.diversity()

        rewards = optimality * (1.0 + self.difficulty) + diversity

        return rel, pr, omega, optimality, diversity, rewards
