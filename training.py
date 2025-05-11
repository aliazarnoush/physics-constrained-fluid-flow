"""
Training utilities for physics-constrained neural networks.

This module contains functions for training physics-constrained neural networks
with adaptive loss weighting and collocation point generation.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial
from typing import Dict, Tuple, List, Callable, Any, Union
import time

from physics import compute_ns_residuals, compute_continuity_residual


def init_model_params(model, key, input_shape=(1, 2)):
    """
    Initialize model parameters.
    
    Args:
        model: Flax model to initialize
        key: Random seed key
        input_shape: Shape of input tensor, default (1, 2) for 2D coordinates
        
    Returns:
        Initialized parameters
    """
    dummy_input = jnp.ones(input_shape)  # Batch size 1, 2D coordinates
    return model.init(key, dummy_input)


def generate_collocation_points(x_domain, y_domain, nx=50, ny=50, method='uniform'):
    """
    Generate collocation points for training.
    
    Args:
        x_domain: Tuple of (x_min, x_max)
        y_domain: Tuple of (y_min, y_max)
        nx: Number of points in x direction
        ny: Number of points in y direction
        method: Sampling method ('uniform', 'random', 'lhs', 'adaptive')
        
    Returns:
        Array of collocation points with shape (nx*ny, 2)
    """
    x_min, x_max = x_domain
    y_min, y_max = y_domain
    
    if method == 'uniform':
        # Uniform grid
        x = jnp.linspace(x_min, x_max, nx)
        y = jnp.linspace(y_min, y_max, ny)
        X, Y = jnp.meshgrid(x, y)
        return jnp.column_stack([X.flatten(), Y.flatten()])
    
    elif method == 'random':
        # Random sampling
        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(key, (nx * ny,), minval=x_min, maxval=x_max)
        y = jax.random.uniform(subkey, (nx * ny,), minval=y_min, maxval=y_max)
        return jnp.column_stack([x, y])
    
    elif method == 'lhs':
        # Latin Hypercube Sampling
        try:
            from scipy.stats import qmc
            
            sampler = qmc.LatinHypercube(d=2, seed=0)
            sample = sampler.random(n=nx * ny)
            
            # Scale samples to the domain
            x = x_min + (x_max - x_min) * sample[:, 0]
            y = y_min + (y_max - y_min) * sample[:, 1]
            
            return jnp.column_stack([x, y])
        except ImportError:
            print("SciPy not available, falling back to uniform sampling")
            return generate_collocation_points(x_domain, y_domain, nx, ny, 'uniform')
    
    else:
        # Default to uniform if method not recognized
        return generate_collocation_points(x_domain, y_domain, nx, ny, 'uniform')


def generate_adaptive_points(model, params, domain_points, residual_fn, n_points=1000):
    """
    Generate adaptive sampling points based on residual magnitude.
    
    Args:
        model: Neural network model
        params: Model parameters
        domain_points: Initial domain sampling points
        residual_fn: Function to compute residuals
        n_points: Number of adaptive points to generate
        
    Returns:
        Array of adaptively sampled points
    """
    # Compute residuals on domain points
    residuals = residual_fn(model, params, domain_points)
    
    # Convert to scalar residual magnitude (if residuals is a tuple)
    if isinstance(residuals, tuple):
        residual_mags = jnp.zeros(domain_points.shape[0])
        for res in residuals:
            residual_mags += jnp.abs(res.flatten())**2
        residual_mags = jnp.sqrt(residual_mags)
    else:
        residual_mags = jnp.abs(residuals.flatten())
    
    # Create distribution based on residual magnitudes
    residual_probs = residual_mags / jnp.sum(residual_mags)
    
    # Sample points based on this distribution
    key = jax.random.PRNGKey(0)
    indices = jax.random.choice(key, len(domain_points), shape=(n_points,), p=residual_probs)
    sampled_points = domain_points[indices]
    
    # Add small random perturbations
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, sampled_points.shape) * 0.01
    perturbed_points = sampled_points + noise
    
    return perturbed_points


def compute_loss(model, params, batch, rho, mu, loss_weights):
    """
    Compute loss function for physics-constrained neural network.
    
    Args:
        model: Neural network model
        params: Model parameters
        batch: Batch of collocation points
        rho: Fluid density
        mu: Dynamic viscosity
        loss_weights: Dictionary of weights for loss components
        
    Returns:
        Tuple of (total_loss, loss_components)
    """
    # Compute residuals
    x_momentum, y_momentum, continuity = compute_ns_residuals(model, params, batch, rho, mu)
    
    # Compute individual loss terms
    x_momentum_loss = jnp.mean(x_momentum**2)
    y_momentum_loss = jnp.mean(y_momentum**2)
    continuity_loss = jnp.mean(continuity**2)
    
    # Apply weights to each loss component
    weighted_x_momentum = loss_weights['x_momentum'] * x_momentum_loss
    weighted_y_momentum = loss_weights['y_momentum'] * y_momentum_loss
    weighted_continuity = loss_weights['continuity'] * continuity_loss
    
    # Compute total loss
    total_loss = weighted_x_momentum + weighted_y_momentum + weighted_continuity
    
    # Return total loss and components for logging
    loss_components = {
        'x_momentum': x_momentum_loss,
        'y_momentum': y_momentum_loss,
        'continuity': continuity_loss,
        'total': total_loss
    }
    
    return total_loss, loss_components


def update_loss_weights(loss_components, current_weights, scale=0.1, method='exp_scaling'):
    """
    Update loss weights adaptively based on current loss components.
    
    Args:
        loss_components: Dictionary of current loss values
        current_weights: Dictionary of current weights
        scale: Scaling factor for weight updates
        method: Method for weight adaptation ('exp_scaling' or 'inverse')
        
    Returns:
        Updated weights dictionary
    """
    if method == 'exp_scaling':
        # Exponential scaling based on loss magnitudes
        max_loss = max(loss_components['x_momentum'],
                       loss_components['y_momentum'],
                       loss_components['continuity'])
        
        new_weights = {
            'x_momentum': current_weights['x_momentum'] * jnp.exp(scale * (max_loss / (loss_components['x_momentum'] + 1e-8) - 1)),
            'y_momentum': current_weights['y_momentum'] * jnp.exp(scale * (max_loss / (loss_components['y_momentum'] + 1e-8) - 1)),
            'continuity': current_weights['continuity'] * jnp.exp(scale * (max_loss / (loss_components['continuity'] + 1e-8) - 1))
        }
        
    elif method == 'inverse':
        # Inverse proportional scaling
        total_loss = loss_components['x_momentum'] + loss_components['y_momentum'] + loss_components['continuity']
        
        new_weights = {
            'x_momentum': 1.0 / (loss_components['x_momentum'] / total_loss + 1e-8),
            'y_momentum': 1.0 / (loss_components['y_momentum'] / total_loss + 1e-8),
            'continuity': 1.0 / (loss_components['continuity'] / total_loss + 1e-8)
        }
        
        # Normalize weights
        sum_weights = sum(new_weights.values())
        for k in new_weights:
            new_weights[k] = new_weights[k] / sum_weights * 3.0  # Scale to have average of 1.0
    
    else:
        # Default: keep weights the same
        new_weights = current_weights.copy()
    
    return new_weights


@partial(jax.jit, static_argnums=(0,))
def train_step(model, params, batch, rho, mu, loss_weights, optimizer, opt_state):
    """
    Execute a single training step.
    
    Args:
        model: Neural network model
        params: Current model parameters
        batch: Batch of training points
        rho: Fluid density
        mu: Dynamic viscosity
        loss_weights: Dictionary of weights for loss components
        optimizer: Optax optimizer
        opt_state: Optimizer state
        
    Returns:
        Tuple of (new_params, new_opt_state, loss, loss_components)
    """
    # Define loss function for this step
    def loss_fn(p):
        loss, components = compute_loss(model, p, batch, rho, mu, loss_weights)
        return loss, components
    
    # Compute gradients
    (loss, loss_components), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Update parameters
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss, loss_components


def train_pinn(model, params, collocation_points, rho, mu, num_epochs=10000, 
              batch_size=1000, learning_rate=1e-3, adaptive_weights=True,
              adaptive_sampling=False, adaptive_frequency=500,
              log_frequency=100, callback=None):
    """
    Train a physics-constrained neural network.
    
    Args:
        model: Neural network model
        params: Initial model parameters
        collocation_points: Training collocation points
        rho: Fluid density
        mu: Dynamic viscosity
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        adaptive_weights: Whether to use adaptive loss weights
        adaptive_sampling: Whether to use adaptive sampling
        adaptive_frequency: Frequency of adaptive updates
        log_frequency: How often to log progress
        callback: Optional callback function for monitoring
        
    Returns:
        Tuple of (trained_params, loss_history, weights_history)
    """
    # Set up optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Initialize loss weights
    loss_weights = {
        'x_momentum': 1.0,
        'y_momentum': 1.0,
        'continuity': 1.0
    }
    
    # Prepare training data
    n_samples = collocation_points.shape[0]
    n_batches = n_samples // batch_size
    
    # Initialize tracking variables
    loss_history = []
    weights_history = []
    
    print(f"Starting training with {n_samples} points, {n_batches} batches per epoch")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Shuffle points for this epoch
        key = jax.random.PRNGKey(epoch)
        perm = jax.random.permutation(key, n_samples)
        shuffled_points = collocation_points[perm]
        
        # Process each batch
        epoch_loss = 0.0
        for batch_idx in range(n_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch = shuffled_points[start_idx:end_idx]
            
            # Train on this batch
            params, opt_state, loss, loss_components = train_step(
                model, params, batch, rho, mu, loss_weights, optimizer, opt_state
            )
            
            epoch_loss += loss / n_batches
        
        # Adaptive weight updating
        if adaptive_weights and epoch % adaptive_frequency == 0 and epoch > 0:
            # Compute loss components on full dataset
            _, full_loss_components = compute_loss(model, params, collocation_points, rho, mu, loss_weights)
            
            # Update weights
            loss_weights = update_loss_weights(full_loss_components, loss_weights)
            weights_history.append((epoch, loss_weights.copy()))
        
        # Adaptive sampling
        if adaptive_sampling and epoch % adaptive_frequency == 0 and epoch > 0:
            # Generate new points with higher density in regions with large residuals
            new_points = generate_adaptive_points(
                model, params, collocation_points, 
                lambda m, p, x: compute_ns_residuals(m, p, x, rho, mu),
                n_points=n_samples // 2
            )
            
            # Combine with uniform sampling
            uniform_points = generate_collocation_points(
                (jnp.min(collocation_points[:, 0]), jnp.max(collocation_points[:, 0])),
                (jnp.min(collocation_points[:, 1]), jnp.max(collocation_points[:, 1])),
                nx=int(jnp.sqrt(n_samples // 2)),
                ny=int(jnp.sqrt(n_samples // 2))
            )
            
            collocation_points = jnp.vstack([new_points, uniform_points])
            print(f"Updated collocation points at epoch {epoch}")
        
        # Logging
        loss_history.append((epoch, float(epoch_loss)))
        if epoch % log_frequency == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.6e}, Time: {elapsed:.2f}s")
            
            if callback is not None:
                callback(epoch, params, loss_history, weights_history)
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")
    
    return params, loss_history, weights_history


def train_parameterized_pinn(model, params, collocation_points, parameter_values, 
                           rho, mu, num_epochs=10000, batch_size=1000, 
                           learning_rate=1e-3, **kwargs):
    """
    Train a parameterized physics-constrained neural network.
    
    Args:
        model: Parameterized neural network model
        params: Initial model parameters
        collocation_points: Training collocation points
        parameter_values: List of parameter values to train on
        rho: Fluid density
        mu: Dynamic viscosity
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        **kwargs: Additional arguments for train_pinn
        
    Returns:
        Tuple of (trained_params, loss_history, weights_history)
    """
    # Set up optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    # Initialize loss weights
    loss_weights = {
        'x_momentum': 1.0,
        'y_momentum': 1.0,
        'continuity': 1.0
    }
    
    # Prepare training data
    n_samples = collocation_points.shape[0]
    n_param_values = len(parameter_values)
    samples_per_param = n_samples // n_param_values
    
    # Initialize tracking variables
    loss_history = []
    weights_history = []
    
    print(f"Starting parameterized training with {n_samples} points, {n_param_values} parameter values")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Shuffle points and parameters for this epoch
        key = jax.random.PRNGKey(epoch)
        key, subkey = jax.random.split(key)
        perm_points = jax.random.permutation(key, n_samples)
        perm_params = jax.random.permutation(subkey, n_param_values)
        
        shuffled_points = collocation_points[perm_points]
        shuffled_params = jnp.array(parameter_values)[perm_params]
        
        # Process batches
        n_batches = n_samples // batch_size
        epoch_loss = 0.0
        
        for batch_idx in range(n_batches):
            # Get batch of points
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_points = shuffled_points[start_idx:end_idx]
            
            # Get batch of parameters (by cycling through them)
            batch_size_params = batch_size // n_param_values + 1
            param_indices = jnp.tile(jnp.arange(n_param_values), batch_size_params)[:batch_size]
            batch_params = shuffled_params[param_indices]
            
            # Define a training step for parameterized model
            def param_train_step(p):
                total_loss = 0.0
                components_sum = {'x_momentum': 0.0, 'y_momentum': 0.0, 'continuity': 0.0, 'total': 0.0}
                
                for i in range(batch_points.shape[0]):
                    pt = batch_points[i:i+1]  # Single point
                    param = batch_params[i]   # Corresponding parameter
                    
                    # Define parameterized loss function
                    def loss_fn(pt, param):
                        x_momentum, y_momentum, continuity = compute_ns_residuals(
                            model, p, pt, rho, mu, param
                        )
                        
                        # Compute loss components
                        x_momentum_loss = jnp.mean(x_momentum**2)
                        y_momentum_loss = jnp.mean(y_momentum**2)
                        continuity_loss = jnp.mean(continuity**2)
                        
                        # Apply weights
                        weighted_x_momentum = loss_weights['x_momentum'] * x_momentum_loss
                        weighted_y_momentum = loss_weights['y_momentum'] * y_momentum_loss
                        weighted_continuity = loss_weights['continuity'] * continuity_loss
                        
                        # Total loss
                        loss = weighted_x_momentum + weighted_y_momentum + weighted_continuity
                        
                        return loss, {
                            'x_momentum': x_momentum_loss,
                            'y_momentum': y_momentum_loss,
                            'continuity': continuity_loss,
                            'total': loss
                        }
                    
                    loss, components = loss_fn(pt, param)
                    total_loss += loss / batch_points.shape[0]
                    
                    # Accumulate components
                    for k in components:
                        components_sum[k] += components[k] / batch_points.shape[0]
                
                return total_loss, components_sum
            
            # Compute gradients
            (loss, components), grads = jax.value_and_grad(param_train_step, has_aux=True)(params)
            
            # Update parameters
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            epoch_loss += loss / n_batches
        
        # Update weights adaptively if needed
        if kwargs.get('adaptive_weights', False) and epoch % kwargs.get('adaptive_frequency', 500) == 0 and epoch > 0:
            # Implement adaptive weight updates (similar to train_pinn)
            pass
        
        # Logging
        loss_history.append((epoch, float(epoch_loss)))
        if epoch % kwargs.get('log_frequency', 100) == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.6e}, Time: {elapsed:.2f}s")
    
    total_time = time.time() - start_time
    print(f"Parameterized training completed in {total_time:.2f}s")
    
    return params, loss_history, weights_history