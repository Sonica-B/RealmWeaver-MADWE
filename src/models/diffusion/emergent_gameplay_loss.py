"""
Emergent Gameplay Loss Components for MADWE
Day 7: Tuesday, June 11 - NWSG Development
Implements emergent gameplay loss, diversity metrics, and agent coherence scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from collections import defaultdict
import json
from pathlib import Path
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
import itertools


class GameplayElement(Enum):
    """Types of gameplay elements that can emerge"""

    CHALLENGE = "challenge"
    REWARD = "reward"
    OBSTACLE = "obstacle"
    PATH = "path"
    INTERACTION = "interaction"
    NARRATIVE = "narrative"
    DISCOVERY = "discovery"
    CHOICE = "choice"


@dataclass
class GameplayPattern:
    """Represents an emergent gameplay pattern"""

    pattern_id: str
    element_types: List[GameplayElement]
    spatial_arrangement: np.ndarray  # 2D grid representation
    temporal_sequence: List[Tuple[float, str]]  # (time, event)
    player_affordances: Set[str]  # What actions are available
    difficulty_curve: List[float]
    coherence_score: float = 0.0
    diversity_score: float = 0.0

    def to_tensor(self) -> torch.Tensor:
        """Convert pattern to tensor representation"""
        # Flatten spatial arrangement
        spatial_flat = self.spatial_arrangement.flatten()

        # Encode element types
        element_encoding = np.zeros(len(GameplayElement))
        for elem in self.element_types:
            element_encoding[elem.value] = 1

        # Encode difficulty curve
        if self.difficulty_curve:
            diff_array = np.array(self.difficulty_curve[:10])  # Limit to 10
            diff_array = np.pad(diff_array, (0, 10 - len(diff_array)))
        else:
            diff_array = np.zeros(10)

        # Combine all features
        features = np.concatenate(
            [
                spatial_flat[:100],  # First 100 spatial features
                element_encoding,
                diff_array,
                [self.coherence_score, self.diversity_score],
            ]
        )

        return torch.from_numpy(features).float()


class EmergentGameplayLoss(nn.Module):
    """
    Loss function for encouraging emergent gameplay patterns
    Based on the paper's equation combining multiple objectives
    """

    def __init__(
        self,
        lambda_hierarchical: float = 1.0,
        lambda_diversity: float = 0.5,
        lambda_coherence: float = 0.7,
        lambda_emergence: float = 0.8,
        lambda_player: float = 0.6,
    ):
        super().__init__()

        # Loss weights (λ values from paper)
        self.lambda_hierarchical = lambda_hierarchical
        self.lambda_diversity = lambda_diversity
        self.lambda_coherence = lambda_coherence
        self.lambda_emergence = lambda_emergence
        self.lambda_player = lambda_player

        # Pattern analyzer
        self.pattern_analyzer = GameplayPatternAnalyzer()

        # Diversity metrics
        self.diversity_calculator = DiversityMetrics()

        # Coherence scorer
        self.coherence_scorer = CoherenceScorer()

        # Player model (placeholder for PIGP integration)
        self.player_predictor = PlayerPredictor()

    def forward(
        self,
        generated_content: Dict[str, torch.Tensor],
        context: Dict[str, Any],
        agent_states: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute emergent gameplay loss

        Args:
            generated_content: Dictionary of generated content from different agents
            context: Current game context (player state, narrative, etc.)
            agent_states: States of different agents for coherence

        Returns:
            Dictionary of loss components
        """
        losses = {}

        # 1. Hierarchical Consistency Loss
        losses["hierarchical"] = self._compute_hierarchical_loss(generated_content)

        # 2. Diversity Loss
        losses["diversity"] = self._compute_diversity_loss(generated_content)

        # 3. Agent Coherence Loss
        if agent_states:
            losses["coherence"] = self._compute_coherence_loss(
                generated_content, agent_states
            )
        else:
            losses["coherence"] = torch.tensor(0.0)

        # 4. Emergence Loss
        losses["emergence"] = self._compute_emergence_loss(generated_content, context)

        # 5. Player Adaptivity Loss
        losses["player"] = self._compute_player_loss(generated_content, context)

        # Total weighted loss
        total_loss = (
            self.lambda_hierarchical * losses["hierarchical"]
            + self.lambda_diversity * losses["diversity"]
            + self.lambda_coherence * losses["coherence"]
            + self.lambda_emergence * losses["emergence"]
            + self.lambda_player * losses["player"]
        )

        losses["total"] = total_loss
        return losses

    def _compute_hierarchical_loss(
        self, content: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Ensure consistency across hierarchical levels"""
        loss = 0.0

        # Check if we have multi-scale content
        if "global_map" in content and "local_map" in content:
            # Downsample local to match global scale
            local_downsampled = F.adaptive_avg_pool2d(
                content["local_map"], output_size=content["global_map"].shape[-2:]
            )

            # Consistency loss
            loss += F.mse_loss(local_downsampled, content["global_map"])

        # Check narrative consistency
        if "narrative_embedding" in content and "environment_embedding" in content:
            # Embeddings should be somewhat aligned
            cos_sim = F.cosine_similarity(
                content["narrative_embedding"], content["environment_embedding"], dim=-1
            )
            loss += 1 - cos_sim.mean()  # Encourage similarity

        return loss

    def _compute_diversity_loss(self, content: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Encourage diversity in generated content"""
        diversity_scores = []

        # Spatial diversity
        if "environment_map" in content:
            spatial_div = self.diversity_calculator.compute_spatial_diversity(
                content["environment_map"]
            )
            diversity_scores.append(spatial_div)

        # Asset diversity
        if "generated_assets" in content:
            asset_div = self.diversity_calculator.compute_asset_diversity(
                content["generated_assets"]
            )
            diversity_scores.append(asset_div)

        # Pattern diversity
        if "gameplay_patterns" in content:
            pattern_div = self.diversity_calculator.compute_pattern_diversity(
                content["gameplay_patterns"]
            )
            diversity_scores.append(pattern_div)

        if diversity_scores:
            # We want to maximize diversity, so minimize negative diversity
            return -torch.stack(diversity_scores).mean()
        else:
            return torch.tensor(0.0)

    def _compute_coherence_loss(
        self, content: Dict[str, torch.Tensor], agent_states: Dict[str, Any]
    ) -> torch.Tensor:
        """Ensure coherence between different agents' outputs"""
        coherence_scores = []

        # Get all agent pairs
        agents = list(agent_states.keys())
        for agent1, agent2 in itertools.combinations(agents, 2):
            if agent1 in content and agent2 in content:
                score = self.coherence_scorer.compute_pairwise_coherence(
                    content[agent1],
                    content[agent2],
                    agent_states[agent1],
                    agent_states[agent2],
                )
                coherence_scores.append(score)

        if coherence_scores:
            # We want high coherence, so minimize negative coherence
            return -torch.stack(coherence_scores).mean()
        else:
            return torch.tensor(0.0)

    def _compute_emergence_loss(
        self, content: Dict[str, torch.Tensor], context: Dict[str, Any]
    ) -> torch.Tensor:
        """Encourage emergent gameplay patterns"""
        emergence_score = 0.0

        # Analyze generated content for emergent patterns
        patterns = self.pattern_analyzer.extract_patterns(content)

        # Score each pattern for emergence potential
        for pattern in patterns:
            # Check for interesting combinations
            if len(pattern.element_types) >= 3:
                emergence_score += 0.2

            # Check for player choice opportunities
            if len(pattern.player_affordances) >= 2:
                emergence_score += 0.3

            # Check for narrative potential
            if GameplayElement.NARRATIVE in pattern.element_types:
                emergence_score += 0.2

            # Check difficulty progression
            if pattern.difficulty_curve:
                # Reward interesting difficulty curves
                curve_variance = np.var(pattern.difficulty_curve)
                if 0.1 < curve_variance < 0.5:  # Not too flat, not too spiky
                    emergence_score += 0.3

        # Normalize by number of patterns
        if patterns:
            emergence_score /= len(patterns)

        # We want to maximize emergence
        return -torch.tensor(emergence_score)

    def _compute_player_loss(
        self, content: Dict[str, torch.Tensor], context: Dict[str, Any]
    ) -> torch.Tensor:
        """Adapt content to predicted player behavior"""
        if "player_state" not in context:
            return torch.tensor(0.0)

        # Predict player preferences
        player_prefs = self.player_predictor.predict_preferences(
            context["player_state"]
        )

        # Score content against preferences
        alignment_score = 0.0

        if "difficulty" in player_prefs and "environment_map" in content:
            # Check if difficulty matches preference
            actual_difficulty = self._estimate_difficulty(content["environment_map"])
            diff_error = abs(actual_difficulty - player_prefs["difficulty"])
            alignment_score += 1 - diff_error

        if "play_style" in player_prefs:
            # Check if content supports preferred play style
            supported_styles = self._analyze_supported_playstyles(content)
            if player_prefs["play_style"] in supported_styles:
                alignment_score += 1.0

        # We want to maximize alignment
        return -torch.tensor(alignment_score / 2.0)  # Normalize

    def _estimate_difficulty(self, environment_map: torch.Tensor) -> float:
        """Estimate difficulty from environment"""
        # Simple heuristic: more obstacles = higher difficulty
        if environment_map.dim() == 4:
            obstacle_density = (environment_map > 0.7).float().mean()
            return obstacle_density.item()
        return 0.5

    def _analyze_supported_playstyles(
        self, content: Dict[str, torch.Tensor]
    ) -> Set[str]:
        """Analyze what playstyles the content supports"""
        supported = set()

        # Check for different playstyle indicators
        if "environment_map" in content:
            env_map = content["environment_map"]

            # Multiple paths indicate exploration playstyle
            if self._has_multiple_paths(env_map):
                supported.add("explorer")

            # Hidden areas indicate completionist playstyle
            if self._has_hidden_areas(env_map):
                supported.add("completionist")

            # Combat zones indicate action playstyle
            if self._has_combat_zones(env_map):
                supported.add("fighter")

        return supported

    def _has_multiple_paths(self, env_map: torch.Tensor) -> bool:
        """Check if environment has multiple paths"""
        # Simplified check - would use path finding in practice
        return True

    def _has_hidden_areas(self, env_map: torch.Tensor) -> bool:
        """Check for hidden/secret areas"""
        # Simplified check
        return torch.rand(1).item() > 0.5

    def _has_combat_zones(self, env_map: torch.Tensor) -> bool:
        """Check for combat zones"""
        # Simplified check
        return torch.rand(1).item() > 0.5


class DiversityMetrics:
    """Calculate diversity metrics for generated content"""

    def compute_spatial_diversity(self, spatial_map: torch.Tensor) -> torch.Tensor:
        """Compute diversity in spatial arrangement"""
        if spatial_map.dim() == 4:
            # Batch of maps
            batch_size = spatial_map.shape[0]
            diversity_scores = []

            for i in range(batch_size):
                single_map = spatial_map[i]

                # Compute entropy of spatial distribution
                # Flatten and discretize
                flat_map = single_map.flatten()
                hist = torch.histc(flat_map, bins=10, min=0, max=1)
                hist = hist / hist.sum()

                # Entropy as diversity measure
                entropy_val = -torch.sum(hist * torch.log(hist + 1e-8))
                diversity_scores.append(entropy_val)

            return torch.stack(diversity_scores).mean()

        return torch.tensor(0.0)

    def compute_asset_diversity(self, assets: torch.Tensor) -> torch.Tensor:
        """Compute diversity in generated assets"""
        if assets.dim() >= 3:
            # Compute pairwise distances
            batch_size = assets.shape[0]

            if batch_size > 1:
                # Flatten assets for comparison
                flat_assets = assets.view(batch_size, -1)

                # Compute pairwise distances
                distances = torch.cdist(flat_assets, flat_assets)

                # Average distance as diversity measure
                # Exclude diagonal (self-distances)
                mask = ~torch.eye(batch_size, dtype=torch.bool, device=assets.device)
                avg_distance = distances[mask].mean()

                return avg_distance

        return torch.tensor(0.0)

    def compute_pattern_diversity(self, patterns: torch.Tensor) -> torch.Tensor:
        """Compute diversity in gameplay patterns"""
        # Assuming patterns is a tensor of pattern features
        if patterns.dim() >= 2 and patterns.shape[0] > 1:
            # Compute variance along feature dimensions
            pattern_variance = torch.var(patterns, dim=0).mean()
            return pattern_variance

        return torch.tensor(0.0)


class CoherenceScorer:
    """Score coherence between different agents' outputs"""

    def compute_pairwise_coherence(
        self,
        content1: torch.Tensor,
        content2: torch.Tensor,
        state1: Dict[str, Any],
        state2: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute coherence between two agents' outputs"""
        coherence_score = 0.0

        # Style coherence
        if content1.shape == content2.shape:
            # Use cosine similarity in feature space
            feat1 = self._extract_features(content1)
            feat2 = self._extract_features(content2)

            style_coherence = F.cosine_similarity(feat1, feat2, dim=-1).mean()
            coherence_score += style_coherence

        # Semantic coherence based on agent states
        if "semantic_embedding" in state1 and "semantic_embedding" in state2:
            sem_coherence = F.cosine_similarity(
                state1["semantic_embedding"], state2["semantic_embedding"], dim=-1
            ).mean()
            coherence_score += sem_coherence

        # Boundary coherence (for spatial content)
        if self._are_spatial_neighbors(state1, state2):
            boundary_coherence = self._compute_boundary_coherence(content1, content2)
            coherence_score += boundary_coherence

        return coherence_score / 3.0  # Average of components

    def _extract_features(self, content: torch.Tensor) -> torch.Tensor:
        """Extract features for coherence comparison"""
        # Simple feature extraction - in practice would use pretrained model
        if content.dim() == 4:
            # Global average pooling
            return F.adaptive_avg_pool2d(content, (1, 1)).flatten(1)
        elif content.dim() == 3:
            return content.mean(dim=-1)
        else:
            return content

    def _are_spatial_neighbors(
        self, state1: Dict[str, Any], state2: Dict[str, Any]
    ) -> bool:
        """Check if two agents generate spatially adjacent content"""
        if "spatial_coords" in state1 and "spatial_coords" in state2:
            coords1 = state1["spatial_coords"]
            coords2 = state2["spatial_coords"]

            # Manhattan distance
            distance = abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1])
            return distance == 1

        return False

    def _compute_boundary_coherence(
        self, content1: torch.Tensor, content2: torch.Tensor
    ) -> torch.Tensor:
        """Compute coherence at boundaries between adjacent content"""
        # Extract edges
        if content1.dim() == 4 and content2.dim() == 4:
            # Assume content1 is to the left of content2
            right_edge = content1[:, :, :, -1]  # Last column
            left_edge = content2[:, :, :, 0]  # First column

            # Compare edges
            edge_similarity = 1 - F.mse_loss(right_edge, left_edge)
            return edge_similarity

        return torch.tensor(0.5)


class GameplayPatternAnalyzer:
    """Analyze content for emergent gameplay patterns"""

    def extract_patterns(
        self, content: Dict[str, torch.Tensor]
    ) -> List[GameplayPattern]:
        """Extract gameplay patterns from generated content"""
        patterns = []

        if "environment_map" in content:
            env_patterns = self._extract_environmental_patterns(
                content["environment_map"]
            )
            patterns.extend(env_patterns)

        if "interaction_points" in content:
            interaction_patterns = self._extract_interaction_patterns(
                content["interaction_points"]
            )
            patterns.extend(interaction_patterns)

        if "narrative_graph" in content:
            narrative_patterns = self._extract_narrative_patterns(
                content["narrative_graph"]
            )
            patterns.extend(narrative_patterns)

        return patterns

    def _extract_environmental_patterns(
        self, env_map: torch.Tensor
    ) -> List[GameplayPattern]:
        """Extract patterns from environmental layout"""
        patterns = []

        # Convert to numpy for analysis
        if env_map.dim() == 4:
            env_np = env_map[0].cpu().numpy()  # Take first in batch
        else:
            env_np = env_map.cpu().numpy()

        # Find interesting spatial arrangements
        # Example: Bottlenecks (narrow passages)
        bottlenecks = self._find_bottlenecks(env_np)
        for bn in bottlenecks:
            pattern = GameplayPattern(
                pattern_id=f"bottleneck_{len(patterns)}",
                element_types=[GameplayElement.CHALLENGE, GameplayElement.PATH],
                spatial_arrangement=bn,
                temporal_sequence=[],
                player_affordances={"move", "fight", "sneak"},
                difficulty_curve=[0.3, 0.7, 0.5],  # Spike at bottleneck
            )
            patterns.append(pattern)

        # Example: Open areas (potential combat/exploration zones)
        open_areas = self._find_open_areas(env_np)
        for area in open_areas:
            pattern = GameplayPattern(
                pattern_id=f"open_area_{len(patterns)}",
                element_types=[GameplayElement.DISCOVERY, GameplayElement.CHOICE],
                spatial_arrangement=area,
                temporal_sequence=[],
                player_affordances={"explore", "fight", "gather"},
                difficulty_curve=[0.2, 0.2, 0.3],  # Low consistent difficulty
            )
            patterns.append(pattern)

        return patterns

    def _extract_interaction_patterns(
        self, interaction_points: torch.Tensor
    ) -> List[GameplayPattern]:
        """Extract patterns from interaction points"""
        patterns = []

        # Analyze clustering of interaction points
        if interaction_points.numel() > 0:
            # Example: Quest hub pattern
            clusters = self._find_clusters(interaction_points)
            for cluster in clusters:
                pattern = GameplayPattern(
                    pattern_id=f"quest_hub_{len(patterns)}",
                    element_types=[
                        GameplayElement.INTERACTION,
                        GameplayElement.NARRATIVE,
                    ],
                    spatial_arrangement=cluster,
                    temporal_sequence=[
                        ("0", "discover"),
                        ("1", "interact"),
                        ("2", "quest"),
                    ],
                    player_affordances={"talk", "trade", "accept_quest"},
                    difficulty_curve=[0.1, 0.1, 0.1],  # Non-combat area
                )
                patterns.append(pattern)

        return patterns

    def _extract_narrative_patterns(
        self, narrative_graph: Any
    ) -> List[GameplayPattern]:
        """Extract patterns from narrative structure"""
        patterns = []

        # This would analyze the narrative graph for patterns like:
        # - Branching storylines (choices)
        # - Convergence points (mandatory story beats)
        # - Side quests

        return patterns

    def _find_bottlenecks(self, env_map: np.ndarray) -> List[np.ndarray]:
        """Find bottleneck areas in environment"""
        # Simplified - would use proper pathfinding/flow analysis
        bottlenecks = []

        # Look for narrow passages
        h, w = env_map.shape[-2:]
        window_size = 5

        for i in range(0, h - window_size, window_size):
            for j in range(0, w - window_size, window_size):
                window = env_map[..., i : i + window_size, j : j + window_size]

                # Check if this looks like a bottleneck
                if self._is_bottleneck_pattern(window):
                    bottlenecks.append(window)

        return bottlenecks

    def _find_open_areas(self, env_map: np.ndarray) -> List[np.ndarray]:
        """Find open areas in environment"""
        open_areas = []

        # Look for large connected regions
        h, w = env_map.shape[-2:]
        window_size = 10

        for i in range(0, h - window_size, window_size):
            for j in range(0, w - window_size, window_size):
                window = env_map[..., i : i + window_size, j : j + window_size]

                # Check if mostly open
                if np.mean(window < 0.3) > 0.7:  # 70% traversable
                    open_areas.append(window)

        return open_areas

    def _find_clusters(self, points: torch.Tensor) -> List[np.ndarray]:
        """Find clusters in point data"""
        # Simplified clustering
        return []

    def _is_bottleneck_pattern(self, window: np.ndarray) -> bool:
        """Check if window contains bottleneck pattern"""
        # Simplified check - would use proper pattern matching
        center = window.shape[-1] // 2

        # Check for narrow passage in center
        center_col = window[..., :, center]
        traversable = np.sum(center_col < 0.3)

        return 1 <= traversable <= 2  # 1-2 traversable cells


class PlayerPredictor:
    """Predict player preferences and behavior (placeholder for PIGP)"""

    def predict_preferences(self, player_state: Dict[str, Any]) -> Dict[str, Any]:
        """Predict player preferences from state"""
        # This would integrate with the full PIGP system
        # For now, return mock preferences

        preferences = {
            "difficulty": 0.5,  # Medium difficulty
            "play_style": "explorer",
            "preferred_elements": [
                GameplayElement.DISCOVERY,
                GameplayElement.NARRATIVE,
            ],
        }

        # Adjust based on player state
        if "skill_level" in player_state:
            preferences["difficulty"] = player_state["skill_level"]

        if "recent_actions" in player_state:
            # Infer play style from actions
            actions = player_state["recent_actions"]
            if "combat" in actions:
                preferences["play_style"] = "fighter"
            elif "dialogue" in actions:
                preferences["play_style"] = "socializer"

        return preferences


def main():
    """Test emergent gameplay loss"""
    # Create loss function
    loss_fn = EmergentGameplayLoss()

    # Create mock generated content
    generated_content = {
        "environment_map": torch.rand(4, 3, 64, 64),
        "global_map": torch.rand(4, 3, 32, 32),
        "local_map": torch.rand(4, 3, 64, 64),
        "narrative_embedding": torch.rand(4, 256),
        "environment_embedding": torch.rand(4, 256),
        "generated_assets": torch.rand(10, 3, 64, 64),
        "gameplay_patterns": torch.rand(5, 50),  # 5 patterns, 50 features each
    }

    # Create mock context
    context = {
        "player_state": {
            "skill_level": 0.7,
            "recent_actions": ["explore", "combat", "dialogue"],
            "position": (10, 20),
        },
        "narrative_state": "chapter_2",
        "time_of_day": "evening",
    }

    # Create mock agent states
    agent_states = {
        "environment_agent": {
            "semantic_embedding": torch.rand(256),
            "spatial_coords": (0, 0),
        },
        "asset_agent": {
            "semantic_embedding": torch.rand(256),
            "spatial_coords": (0, 1),
        },
        "narrative_agent": {
            "semantic_embedding": torch.rand(256),
            "spatial_coords": (1, 0),
        },
    }

    # Compute loss
    losses = loss_fn(generated_content, context, agent_states)

    # Print results
    print("Emergent Gameplay Loss Components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")


if __name__ == "__main__":
    main()
