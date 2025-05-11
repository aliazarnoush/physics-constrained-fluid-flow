"""
Training implementation for physics-constrained neural networks using JAX.

This module implements the core training logic using JAX's functional approach
for physics-constrained deep learning of fluid flows.
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from functools import partial
from typing import Tuple, List, Dict, Any, Callable, Optional, Union
from physics import compute_ns_residuals


@jax.jit
def train_step(params, opt_state, batch, rho, mu, model, optimizer, lambda_cont=1.0):
    """
    Single training step for physics-constrained neural network.
    
    JAX's functional approach means computation of gradients with respect to a pure function
    and update of parameters using the optimizer's update function.
    
    Args:
        params: Current model parameters
        opt_state: Current optimizer state
        batch: Batch of collocation points
        rho: Fluid density
        mu: Dynamic viscosity
        model: PINN model instance
        optimizer: Optax optimizer
        lambda_cont: Weight for continuity equation residual
    
    Returns:
        Updated parameters, optimizer state, loss value, and auxiliary info
    """
    
    def loss_fn(params):
        """Loss function combining momentum and continuity residuals"""
        # Compute residuals for current batch
        x_momentum, y_momentum, continuity = compute_ns_residuals(
            model, params, batch, rho, mu
        )
        
        # Compute mean squared residuals
        x_momentum_loss = jnp.mean(x_momentum**2)
        y_momentum_loss = jnp.mean(y_momentum**2)
        continuity_loss = jnp.mean(continuity**2)
        
        # Total loss with weighting
        total_loss = x_momentum_loss + y_momentum_loss + lambda_cont * continuity_loss
        
        return total_loss, (x_momentum_loss, y_momentum_loss, continuity_loss)
    
    # Compute loss and gradients using JAX's value_and_grad
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Update parameters using optimizer
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss, aux


def adaptive_loss_weights(momentum_grad_norm, continuity_grad_norm, alpha=0.1):
    """
    Compute adaptive weights for balancing different loss terms.
    
    This function implements the adaptive weighting scheme to balance
    competing gradient contributions from different physics terms.
    
    Args:
        momentum_grad_norm: Norm of momentum equation gradients
        continuity_grad_norm: Norm of continuity equation gradients
        alpha: Adaptation rate parameter
    
    Returns:
        Tuple of (momentum_weight, continuity_weight)
    """
    ratio_m = momentum_grad_norm / continuity_grad_norm
    ratio_c = continuity_grad_norm / momentum_grad_norm
    
    # Apply exponential scaling
    lambda_m = jnp.exp(-alpha * ratio_m)
    lambda_c = jnp.exp(-alpha * ratio_c)
    
    return lambda_m, lambda_c


@partial(jax.jit, static_argnums=(4, 5, 6, 7, 8))
def train_step_adaptive(params, opt_state, batch, epoch, rho, mu, model, optimizer, n_adapt):
    """
    Training step with adaptive loss weighting.
    
    This function implements the adaptive training procedure that balances
    different physics components through dynamic weight adjustment.
    
    Args:
        params: Current model parameters
        opt_state: Current optimizer state
        batch: Batch of collocation points
        epoch: Current training epoch
        rho: Fluid density
        mu: Dynamic viscosity
        model: PINN model instance
        optimizer: Optax optimizer
        n_adapt: Interval for weight adaptation
    
    Returns:
        Updated parameters, optimizer state, loss value, weights, and auxiliary info
    """
    # Initial weights
    momentum_weight = 1.0
    continuity_weight = 1.0
    
    # Separate loss functions for computing gradients
    def momentum_loss_fn(params):
        x_momentum, y_momentum, _ = compute_ns_residuals(model, params, batch, rho, mu)
        return jnp.mean(x_momentum**2) + jnp.mean(y_momentum**2)
    
    def continuity_loss_fn(params):
        _, _, continuity = compute_ns_residuals(model, params, batch, rho, mu)
        return jnp.mean(continuity**2)
    
    # Combined loss for training
    def total_loss_fn(params):
        x_momentum, y_momentum, continuity = compute_ns_residuals(model, params, batch, rho, mu)
        
        # Compute individual losses
        x_momentum_loss = jnp.mean(x_momentum**2)
        y_momentum_loss = jnp.mean(y_momentum**2)
        continuity_loss = jnp.mean(continuity**2)
        
        # Total loss with weighting
        momentum_loss = x_momentum_loss + y_momentum_loss
        total_loss = momentum_weight * momentum_loss + continuity_weight * continuity_loss
        
        return total_loss, (momentum_loss, continuity_loss)
    
    # Adapt weights every n_adapt epochs
    do_adapt = (epoch % n_adapt == 0)
    
    def adapt_weights():
        # Compute gradient norms
        momentum_grad = jax.grad(momentum_loss_fn)(params)
        continuity_grad = jax.grad(continuity_loss_fn)(params)
        
        # Compute L2 norms
        momentum_grad_norm = optax.global_norm(momentum_grad)
        continuity_grad_norm = optax.global_norm(continuity_grad)
        
        # Update weights
        new_momentum_weight, new_continuity_weight = adaptive_loss_weights(
            momentum_grad_norm, continuity_grad_norm
        )
        
        return new_momentum_weight, new_continuity_weight
    
    # Only adapt weights periodically
    momentum_weight, continuity_weight = jax.lax.cond(
        do_adapt,
        lambda _: adapt_weights(),
        lambda _: (momentum_weight, continuity_weight),
        operand=None
    )
    
    # Compute loss and gradients
    (loss, aux), grads = jax.value_and_grad(total_loss_fn, has_aux=True)(params)
    
    # Update parameters using optimizer
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss, (momentum_weight, continuity_weight), aux


def generate_collocation_points(domain_bounds, n_points, adaptive=False, boundary_ratio=0.2):
    """
    Generate collocation points for training.
    
    Args:
        domain_bounds: List of (min, max) tuples for each dimension
        n_points: Total number of points to generate
        adaptive: Whether to use adaptive sampling (higher density near boundaries)
        boundary_ratio: Ratio of points to allocate near boundaries if adaptive is True
    
    Returns:
        Array of collocation points
    """
    if not adaptive:
        # Uniform sampling across domain
        points = []
        for dim_bounds in domain_bounds:
            dim_min, dim_max = dim_bounds
            points.append(np.random.uniform(dim_min, dim_max, n_points))
        
        return np.stack(points, axis=1)
    else:
        # Adaptive sampling with higher density near boundaries
        n_boundary = int(n_points * boundary_ratio)
        n_interior = n_points - n_boundary
        
        # Interior points (uniform sampling)
        interior_points = []
        for dim_bounds in domain_bounds:
            dim_min, dim_max = dim_bounds
            interior_points.append(np.random.uniform(dim_min, dim_max, n_interior))
        
        interior_points = np.stack(interior_points, axis=1)
        
        # Boundary points (closer to domain boundaries)
        boundary_points = []
        for i, dim_bounds in enumerate(domain_bounds):
            dim_min, dim_max = dim_bounds
            
            # Points for this dimension's boundaries
            dim_boundary_pts = np.random.uniform(dim_min, dim_max, n_boundary)
            
            # Other dimensions (uniform sampling)
            other_dims = []
            for j, other_bounds in enumerate(domain_bounds):
                if i == j:
                    # Randomly choose near min or max boundary
                    near_min = np.random.rand(n_boundary) < 0.5
                    
                    # Add small offset from boundaries
                    epsilon = 0.02 * (dim_max - dim_min)
                    boundary_vals = np.where(
                        near_min,
                        dim_min + epsilon * np.random.rand(n_boundary),
                        dim_max - epsilon * np.random.rand(n_boundary)
                    )
                    other_dims.append(boundary_vals)
                else:
                    other_min, other_max = other_bounds
                    other_dims.append(np.random.uniform(other_min, other_max, n_boundary))
            
            boundary_points.append(np.stack(other_dims, axis=1))
        
        # Combine all boundary points
        all_boundary_points = np.vstack(boundary_points)
        
        # Randomly select n_boundary points from all boundary points
        idx = np.random.choice(len(all_boundary_points), n_boundary, replace=False)
        selected_boundary_points = all_boundary_points[idx]
        
        # Combine interior and boundary points
        all_points = np.vstack([interior_points, selected_boundary_points])
        
        # Shuffle points
        np.random.shuffle(all_points)
        
        return all_points


def generate_collocation_points_stenotic(n_points, alpha=0.5, adaptive=True):
    """
    Generate collocation points specifically for stenotic channel geometry.
    
    Args:
        n_points: Total number of points to generate
        alpha: Stenosis severity parameter
        adaptive: Whether to use adaptive sampling (higher density near stenosis)
    
    Returns:
        Array of collocation points
    """
    # Domain bounds for stenotic channel
    x_range = (0.0, 1.0)
    y_range = (0.0, 1.0)
    
    # Basic uniform sampling
    x = np.random.uniform(x_range[0], x_range[1], n_points)
    y = np.random.uniform(y_range[0], y_range[1], n_points)
    
    if adaptive:
        # Concentrate more points near the stenosis region (x=0.5)
        stenosis_center = 0.5
        stenosis_width = 0.2
        
        # Generate more points around stenosis
        ratio_near_stenosis = 0.6  # 60% of points near stenosis
        n_stenosis = int(n_points * ratio_near_stenosis)
        n_uniform = n_points - n_stenosis
        
        # Uniform points
        x_uniform = np.random.uniform(x_range[0], x_range[1], n_uniform)
        y_uniform = np.random.uniform(y_range[0], y_range[1], n_uniform)
        
        # Points concentrated near stenosis
        x_stenosis = np.random.normal(stenosis_center, stenosis_width/2, n_stenosis)
        x_stenosis = np.clip(x_stenosis, x_range[0], x_range[1])
        
        # Upper boundary with Gaussian constriction
        def y_upper(x):
            return 1.0 - alpha * np.exp(-50.0 * (x - 0.5)**2)
        
        # Generate points near the boundary with higher probability
        y_stenosis = []
        for x_i in x_stenosis:
            upper = y_upper(x_i)
            
            # 70% chance to be in the upper half (including near stenosis)
            if np.random.rand() < 0.7:
                # Distribute points in upper half with higher density near stenosis
                if np.random.rand() < 0.7:
                    # Very close to stenosis boundary
                    margin = 0.1 * upper
                    y_i = upper - margin * np.random.rand()
                else:
                    # Upper half but not necessarily near boundary
                    y_i = 0.5 + 0.5 * upper * np.random.rand()
            else:
                # Lower half
                y_i = np.random.uniform(0, 0.5)
            
            y_stenosis.append(y_i)
        
        y_stenosis = np.array(y_stenosis)
        
        # Combine and shuffle all points
        x = np.concatenate([x_uniform, x_stenosis])
        y = np.concatenate([y_uniform, y_stenosis])
        
        # Check points outside the domain and fix them
        for i in range(len(x)):
            y_top = y_upper(x[i])
            if y[i] > y_top:
                y[i] = y_top - 0.01 * np.random.rand()
    
    # Remove points outside the geometry
    valid_points = []
    for i in range(len(x)):
        # Upper boundary with Gaussian constriction
        y_top = 1.0 - alpha * np.exp(-50.0 * (x[i] - 0.5)**2)
        if 0 <= y[i] <= y_top:
            valid_points.append([x[i], y[i]])
    
    # If we lost too many points, add more
    valid_points = np.array(valid_points)
    if len(valid_points) < n_points:
        # Generate more points and filter again
        n_additional = 2 * (n_points - len(valid_points))  # Generate extra to account for filtering
        x_add = np.random.uniform(x_range[0], x_range[1], n_additional)
        y_add = np.random.uniform(y_range[0], y_range[1], n_additional)
        
        for i in range(len(x_add)):
            y_top = 1.0 - alpha * np.exp(-50.0 * (x_add[i] - 0.5)**2)
            if 0 <= y_add[i] <= y_top:
                valid_points = np.vstack([valid_points, [x_add[i], y_add[i]]])
                if len(valid_points) >= n_points:
                    break
    
    # Trim or pad to exactly n_points
    if len(valid_points) > n_points:
        indices = np.random.choice(len(valid_points), n_points, replace=False)
        valid_points = valid_points[indices]
    elif len(valid_points) < n_points:
        # If still too few points, duplicate some
        n_missing = n_points - len(valid_points)
        indices = np.random.choice(len(valid_points), n_missing, replace=True)
        extra_points = valid_points[indices]
        valid_points = np.vstack([valid_points, extra_points])
    
    return valid_points


def initialize_training(model, seed=0, learning_rate=1e-3):
    """
    Initialize model parameters and optimizer for training.
    
    Args:
        model: PINN model instance
        seed: Random seed for initialization
        learning_rate: Initial learning rate for optimizer
    
    Returns:
        Tuple of (initial_params, optimizer, opt_state)
    """
    # Initialize parameters
    key = jax.random.PRNGKey(seed)
    dummy_input = jnp.ones((1, 2))  # Assuming 2D input (x, y)
    params = model.init(key, dummy_input)
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    return params, optimizer, opt_state


def train_pinn(
    model,
    domain_bounds,
    rho,
    mu,
    n_iterations=5000,
    batch_size=1000,
    learning_rate=1e-3,
    adaptive_sampling=True,
    adaptive_weighting=True,
    n_adapt=100,
    seed=0,
    callback=None
):
    """
    Complete training procedure for physics-constrained neural network.
    
    Args:
        model: PINN model instance
        domain_bounds: List of (min, max) tuples for each dimension
        rho: Fluid density
        mu: Dynamic viscosity
        n_iterations: Total number of training iterations
        batch_size: Number of collocation points per batch
        learning_rate: Initial learning rate for optimizer
        adaptive_sampling: Whether to use adaptive collocation point sampling
        adaptive_weighting: Whether to use adaptive loss weighting
        n_adapt: Interval for weight adaptation if adaptive_weighting is True
        seed: Random seed for initialization
        callback: Optional callback function for monitoring training progress
        
    Returns:
        Trained parameters and training history dictionary
    """
    # Initialize model and optimizer
    params, optimizer, opt_state = initialize_training(model, seed, learning_rate)
    
    # Initialize weights for adaptive training
    momentum_weight = 1.0
    continuity_weight = 1.0
    
    # Training history
    history = {
        'loss': [],
        'momentum_loss': [],
        'continuity_loss': [],
        'momentum_weight': [],
        'continuity_weight': []
    }
    
    # Training loop
    for i in range(n_iterations):
        # Generate collocation points for this iteration
        batch = generate_collocation_points(domain_bounds, batch_size, adaptive=adaptive_sampling)
        batch = jnp.array(batch)
        
        # Training step
        if adaptive_weighting:
            params, opt_state, loss, weights, aux = train_step_adaptive(
                params, opt_state, batch, i, rho, mu, model, optimizer, n_adapt
            )
            momentum_weight, continuity_weight = weights
            momentum_loss, continuity_loss = aux
        else:
            params, opt_state, loss, aux = train_step(
                params, opt_state, batch, rho, mu, model, optimizer, lambda_cont=continuity_weight
            )
            momentum_loss = (aux[0] + aux[1]) / 2.0  # Average of x and y momentum losses
            continuity_loss = aux[2]
        
        # Record history
        history['loss'].append(float(loss))
        history['momentum_loss'].append(float(momentum_loss))
        history['continuity_loss'].append(float(continuity_loss))
        history['momentum_weight'].append(float(momentum_weight))
        history['continuity_weight'].append(float(continuity_weight))
        
        # Optional callback for monitoring
        if callback is not None and i % 100 == 0:
            callback(i, params, history)
    
    return params, history
