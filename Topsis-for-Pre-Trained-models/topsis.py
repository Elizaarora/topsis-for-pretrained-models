"""
TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
Implementation for Multi-Criteria Decision Making
"""

import numpy as np
import pandas as pd


class TOPSIS:
    """
    TOPSIS class for multi-criteria decision analysis.
    
    TOPSIS ranks alternatives based on their distance from:
    - Positive Ideal Solution (best possible values)
    - Negative Ideal Solution (worst possible values)
    """
    
    def __init__(self, decision_matrix, weights, criteria_types):
        """
        Initialize TOPSIS with decision matrix, weights, and criteria types.
        
        Parameters:
        -----------
        decision_matrix : array-like
            Matrix where rows are alternatives and columns are criteria
        weights : array-like
            Weights for each criterion (should sum to 1)
        criteria_types : list
            List of 'max' or 'min' for each criterion
            'max' means higher is better, 'min' means lower is better
        """
        self.decision_matrix = np.array(decision_matrix)
        self.weights = np.array(weights)
        self.criteria_types = criteria_types
        
        # Validate inputs
        if len(weights) != self.decision_matrix.shape[1]:
            raise ValueError("Number of weights must match number of criteria")
        if len(criteria_types) != self.decision_matrix.shape[1]:
            raise ValueError("Number of criteria types must match number of criteria")
        if not np.isclose(np.sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.0")
    
    def normalize_matrix(self):
        """Normalize the decision matrix using vector normalization."""
        # Calculate sum of squares for each criterion
        sum_squares = np.sum(self.decision_matrix ** 2, axis=0)
        # Normalize
        normalized = self.decision_matrix / np.sqrt(sum_squares)
        return normalized
    
    def apply_weights(self, normalized_matrix):
        """Apply weights to the normalized matrix."""
        return normalized_matrix * self.weights
    
    def find_ideal_solutions(self, weighted_matrix):
        """Find positive and negative ideal solutions."""
        positive_ideal = []
        negative_ideal = []
        
        for i, criterion_type in enumerate(self.criteria_types):
            if criterion_type == 'max':
                positive_ideal.append(np.max(weighted_matrix[:, i]))
                negative_ideal.append(np.min(weighted_matrix[:, i]))
            else:  # 'min'
                positive_ideal.append(np.min(weighted_matrix[:, i]))
                negative_ideal.append(np.max(weighted_matrix[:, i]))
        
        return np.array(positive_ideal), np.array(negative_ideal)
    
    def calculate_distances(self, weighted_matrix, positive_ideal, negative_ideal):
        """Calculate distances from positive and negative ideal solutions."""
        positive_distances = np.sqrt(np.sum((weighted_matrix - positive_ideal) ** 2, axis=1))
        negative_distances = np.sqrt(np.sum((weighted_matrix - negative_ideal) ** 2, axis=1))
        return positive_distances, negative_distances
    
    def calculate_scores(self, positive_distances, negative_distances):
        """Calculate TOPSIS scores (closeness coefficients)."""
        scores = negative_distances / (positive_distances + negative_distances)
        return scores
    
    def rank(self):
        """
        Perform TOPSIS ranking and return results.
        
        Returns:
        --------
        results : dict
            Dictionary containing:
            - 'rankings': ranked alternatives with scores
            - 'positive_ideal': positive ideal solution
            - 'negative_ideal': negative ideal solution
            - 'positive_distances': distances from positive ideal
            - 'negative_distances': distances from negative ideal
        """
        # Step 1: Normalize the decision matrix
        normalized = self.normalize_matrix()
        
        # Step 2: Apply weights
        weighted = self.apply_weights(normalized)
        
        # Step 3: Find ideal solutions
        positive_ideal, negative_ideal = self.find_ideal_solutions(weighted)
        
        # Step 4: Calculate distances
        positive_distances, negative_distances = self.calculate_distances(
            weighted, positive_ideal, negative_ideal
        )
        
        # Step 5: Calculate TOPSIS scores
        scores = self.calculate_scores(positive_distances, negative_distances)
        
        # Step 6: Rank alternatives (higher score is better)
        rankings = np.argsort(scores)[::-1]
        
        return {
            'rankings': rankings,
            'scores': scores,
            'positive_ideal': positive_ideal,
            'negative_ideal': negative_ideal,
            'positive_distances': positive_distances,
            'negative_distances': negative_distances,
            'weighted_matrix': weighted
        }
