"""
Configuration and hyperparameters for the neural architecture
"""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ArchitectureConfig:
    """Configuration for the neural architecture"""
    
    # Component dimensions
    input_dim: int = 128
    state_dim: int = 256
    memory_dim: int = 512
    reasoning_dim: int = 256
    action_dim: int = 64
    
    # Memory system
    num_memories: int = 1000
    memory_consolidation_rate: float = 0.01
    
    # Reasoning system
    reasoning_nodes: int = 64
    reasoning_iterations: int = 3
    reasoning_connectivity_rate: float = 0.02
    
    # Learning parameters
    base_learning_rate: float = 0.001
    stability_threshold: float = 0.1
    gradient_clip_norm: float = 1.0
    
    # Meta-optimization
    mutation_rate: float = 0.01
    population_size: int = 10
    performance_window: int = 20
    
    # Stability monitoring
    stability_history_length: int = 100
    catastrophic_threshold: float = 0.05
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'component_dims': {
                'input_dim': self.input_dim,
                'state_dim': self.state_dim,
                'memory_dim': self.memory_dim,
                'reasoning_dim': self.reasoning_dim,
                'action_dim': self.action_dim
            },
            'memory_config': {
                'num_memories': self.num_memories,
                'consolidation_rate': self.memory_consolidation_rate
            },
            'reasoning_config': {
                'num_nodes': self.reasoning_nodes,
                'iterations': self.reasoning_iterations,
                'connectivity_rate': self.reasoning_connectivity_rate
            },
            'learning_config': {
                'base_lr': self.base_learning_rate,
                'stability_threshold': self.stability_threshold,
                'gradient_clip': self.gradient_clip_norm
            },
            'meta_config': {
                'mutation_rate': self.mutation_rate,
                'population_size': self.population_size,
                'performance_window': self.performance_window
            }
        }

# Predefined configurations for different use cases
CONFIGS = {
    'lightweight': ArchitectureConfig(
        input_dim=64,
        state_dim=128,
        memory_dim=256,
        reasoning_dim=128,
        action_dim=32,
        num_memories=500,
        reasoning_nodes=32
    ),
    
    'standard': ArchitectureConfig(
        input_dim=128,
        state_dim=256,
        memory_dim=512,
        reasoning_dim=256,
        action_dim=64,
        num_memories=1000,
        reasoning_nodes=64
    ),
    
    'laptop_moe': ArchitectureConfig(
        input_dim=128,
        state_dim=256,
        memory_dim=512,
        reasoning_dim=256,
        action_dim=64,
        num_memories=1000,
        reasoning_nodes=64
    ),
    
    'ultra_light': ArchitectureConfig(
        input_dim=32,
        state_dim=64,
        memory_dim=128,
        reasoning_dim=64,
        action_dim=16,
        num_memories=100,
        reasoning_nodes=16
    ),
    
    'research_heavy': ArchitectureConfig(
        input_dim=512,
        state_dim=1024,
        memory_dim=2048,
        reasoning_dim=1024,
        action_dim=256,
        num_memories=5000,
        reasoning_nodes=256
    )
}