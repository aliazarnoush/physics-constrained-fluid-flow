"""
Core PINN module for physics-constrained neural networks with JAX.

This module defines the basic neural network architecture with boundary condition enforcement
for physics-constrained deep learning of fluid flows.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Callable, Optional, Tuple, Dict, Any


class PINN(nn.Module):
    """
    Base Physics-Informed Neural Network module.
    
    This is the core architecture that enforces physics constraints through
    automatic differentiation and boundary condition enforcement.
    
    Attributes:
        features: List of integers defining the layer sizes for the network
    """
    features: List[int]  # Layer sizes for the network
    
    @nn.compact
    def __call__(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input coordinates [x, y]
        
        Returns:
            Concatenated [u, v, p] with enforced boundary conditions
        """
        # Neural network backbone
        y = x
        for feat in self.features[:-1]:
            y = nn.Dense(feat)(y)
            y = nn.tanh(y)
        y = nn.Dense(self.features[-1])(y)
        
        # Extract raw network outputs
        u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        
        # Extract coordinate values for boundary condition enforcement
        x_coord, y_coord = x[:, 0:1], x[:, 1:2]
        
        # Note: Boundary conditions are problem-specific
        # Subclasses will override this with appropriate boundary enforcement
        
        return jnp.concatenate([u, v, p], axis=1)


class ParameterizedPINN(nn.Module):
    """
    Parameterized Physics-Informed Neural Network for handling families of geometries or flow conditions.
    
    This extends the base PINN to include parameter inputs that control the flow or geometry,
    enabling generalization across a family of related problems.
    
    Attributes:
        features: List of integers defining the layer sizes for the network
    """
    features: List[int]  # Layer sizes for the network
    
    @nn.compact
    def __call__(self, x, params):
        """
        Forward pass through the network with parameter inputs.
        
        Args:
            x: Input coordinates [x, y]
            params: Problem parameters (e.g., geometry parameters, flow conditions)
        
        Returns:
            Concatenated [u, v, p] with enforced boundary conditions
        """
        # Concatenate coordinates and parameters
        inputs = jnp.concatenate([x, params], axis=1)
        
        # Neural network backbone
        y = inputs
        for feat in self.features[:-1]:
            y = nn.Dense(feat)(y)
            y = nn.tanh(y)
        y = nn.Dense(self.features[-1])(y)
        
        # Extract raw network outputs
        u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        
        # Extract coordinate values for boundary condition enforcement
        x_coord, y_coord = x[:, 0:1], x[:, 1:2]
        
        # Boundary conditions will be implemented in subclasses
        
        return jnp.concatenate([u, v, p], axis=1)


class PoiseuillePINN(nn.Module):
    """
    Physics-informed neural network for Poiseuille flow with embedded boundary conditions.
    
    This class implements the specific boundary conditions for a Poiseuille flow
    between two parallel plates with a pressure gradient.
    
    Attributes:
        features: List of integers defining the layer sizes for the network
        height: Float representing the channel height
        dp_dx: Float representing the pressure gradient
    """
    features: List[int]
    height: float
    dp_dx: float
    
    @nn.compact
    def __call__(self, x):
        """
        Forward pass with embedded boundary conditions for Poiseuille flow.
        
        Args:
            x: Input coordinates [x, y]
            
        Returns:
            Concatenated [u, v, p] with enforced boundary conditions
        """
        # Neural network backbone
        y = x
        for feat in self.features[:-1]:
            y = nn.Dense(feat)(y)
            y = nn.tanh(y)
        y = nn.Dense(self.features[-1])(y)
        
        # Extract raw predictions and coordinates
        u_nn, v_nn, p_nn = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        x_coord, y_coord = x[:, 0:1], x[:, 1:2]
        
        # Enforce boundary conditions
        # For Poiseuille flow: u = 0 at y = 0 and y = height
        # This creates a parabolic modulation that ensures zero velocity at walls
        dist_to_wall = y_coord * (self.height - y_coord) / (self.height**2 / 4)
        
        # Boundary condition enforcement through network architecture (hard constraints)
        u = dist_to_wall * u_nn  # No-slip walls
        v = jnp.zeros_like(v_nn)  # Zero vertical velocity
        p = self.dp_dx * x_coord + p_nn  # Linear pressure gradient + learned perturbation
        
        return jnp.concatenate([u, v, p], axis=1)


class StenoticPINN(nn.Module):
    """
    Physics-informed neural network for stenotic flow with embedded boundary conditions.
    
    This class implements the specific boundary conditions for a stenotic channel flow,
    with a parameterized constriction to simulate different stenosis severities.
    
    Attributes:
        features: List of integers defining the layer sizes for the network
    """
    features: List[int]
    
    @nn.compact
    def __call__(self, x, alpha=0.5):
        """
        Forward pass through the network with stenotic boundary conditions.
        
        Args:
            x: Input coordinates [x, y]
            alpha: Stenosis severity parameter (0 to 1)
            
        Returns:
            Concatenated [u, v, p] with enforced boundary conditions
        """
        # Concatenate coordinates and parameter to allow parameterized learning
        inputs = jnp.concatenate([x, jnp.ones((x.shape[0], 1)) * alpha], axis=1)
        
        # Neural network backbone
        y = inputs
        for feat in self.features[:-1]:
            y = nn.Dense(feat)(y)
            y = nn.tanh(y)
        y = nn.Dense(self.features[-1])(y)
        
        # Extract raw predictions and coordinates
        u_nn, v_nn, p_nn = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        x_coord, y_coord = x[:, 0:1], x[:, 1:2]
        
        # Compute stenotic channel boundaries
        # Upper boundary with Gaussian constriction
        y_upper = 1.0 - alpha * jnp.exp(-50.0 * (x_coord - 0.5)**2)
        y_lower = 0.0
        
        # Normalized distance from boundaries for enforcing boundary conditions
        dist_to_upper = jnp.abs(y_coord - y_upper)
        dist_to_lower = jnp.abs(y_coord - y_lower)
        dist_to_boundary = jnp.minimum(dist_to_upper, dist_to_lower)
        normalized_dist = dist_to_boundary / (y_upper - y_lower) * jnp.pi
        
        # Boundary condition mask - ensures zero velocity at walls
        bc_mask = jnp.sin(normalized_dist)
        
        # Apply boundary conditions
        u = u_nn * bc_mask
        v = v_nn * bc_mask
        
        # For inflow boundary condition, gradually enforce parabolic profile
        inflow_mask = jnp.exp(-50.0 * (x_coord)**2)
        inflow_profile = 4.0 * y_coord * (1.0 - y_coord)  # Parabolic profile
        u = u * (1.0 - inflow_mask) + inflow_profile * inflow_mask
        
        return jnp.concatenate([u, v, p_nn], axis=1)


def apply_boundary_conditions(network_output, x, boundaries, boundary_type='dirichlet'):
    """
    Helper function to apply different types of boundary conditions.
    
    Args:
        network_output: Raw neural network output [u, v, p]
        x: Input coordinates [x, y]
        boundaries: Function that returns boundary distance/mask
        boundary_type: Type of boundary condition ('dirichlet', 'neumann', etc.)
    
    Returns:
        Modified network output with boundary conditions enforced
    """
    u, v, p = network_output[:, 0:1], network_output[:, 1:2], network_output[:, 2:3]
    
    if boundary_type == 'dirichlet':
        # Zero Dirichlet boundary condition (no-slip)
        boundary_mask = boundaries(x)
        u = u * boundary_mask
        v = v * boundary_mask
    elif boundary_type == 'neumann':
        # Zero Neumann boundary condition
        # For implementation, normal derivatives would need to be computed
        # This is more complex and would require additional computation
        pass
    else:
        raise ValueError(f"Unsupported boundary condition type: {boundary_type}")
    
    return jnp.concatenate([u, v, p], axis=1)


def create_distance_function(geometry_fn):
    """
    Create a distance function for boundary condition enforcement.
    
    Args:
        geometry_fn: Function that defines the geometry boundaries
    
    Returns:
        A function that computes the distance to the boundaries
    """
    def distance_fn(x):
        # Get upper and lower boundaries at given x coordinates
        upper_boundary, lower_boundary = geometry_fn(x)
        
        # Extract y coordinates
        y_coord = x[:, 1:2]
        
        # Compute distances to boundaries
        dist_to_upper = jnp.abs(y_coord - upper_boundary)
        dist_to_lower = jnp.abs(y_coord - lower_boundary)
        
        # Minimum distance to any boundary
        min_distance = jnp.minimum(dist_to_upper, dist_to_lower)
        
        # Normalize by channel height
        channel_height = upper_boundary - lower_boundary
        normalized_dist = min_distance / channel_height * jnp.pi
        
        # Apply sine modulation for smooth transition
        return jnp.sin(normalized_dist)
    
    return distance_fn