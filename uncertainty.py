"""
Uncertainty quantification methods for physics-constrained neural networks.

This module provides functions for propagating uncertainties through
physics-constrained neural networks using Monte Carlo sampling and other methods.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Dict, Any, Union, Optional
from functools import partial


def monte_carlo_uq(model, params, domain_points, alpha_mean=0.5, alpha_std=0.1, num_samples=100, seed=0):
    """
    Propagate uncertainty in stenosis parameter through the model using Monte Carlo sampling.
    
    Args:
        model: Trained PINN model
        params: Trained parameters
        domain_points: Grid points in the domain for prediction
        alpha_mean: Mean stenosis severity parameter
        alpha_std: Standard deviation of stenosis parameter
        num_samples: Number of Monte Carlo samples
        seed: Random seed for reproducibility
    
    Returns:
        Mean and standard deviation of predictions
    """
    # Generate random samples of the stenosis parameter
    key = jax.random.PRNGKey(seed)
    alpha_samples = jax.random.normal(key, (num_samples,)) * alpha_std + alpha_mean
    
    # Vectorize prediction function over alpha
    def predict_single(alpha):
        return jax.vmap(lambda x: model.apply(params, x, alpha))(domain_points)
    
    # Map over all parameter samples
    predictions = jax.vmap(predict_single)(alpha_samples)
    
    # Compute statistics
    mean_prediction = jnp.mean(predictions, axis=0)
    std_prediction = jnp.std(predictions, axis=0)
    
    return mean_prediction, std_prediction


def confidence_interval_uq(model, params, domain_points, alpha_mean=0.5, alpha_std=0.1, 
                          num_samples=100, confidence=0.95, seed=0):
    """
    Compute confidence intervals for model predictions using Monte Carlo sampling.
    
    Args:
        model: Trained PINN model
        params: Trained parameters
        domain_points: Grid points in the domain for prediction
        alpha_mean: Mean stenosis severity parameter
        alpha_std: Standard deviation of stenosis parameter
        num_samples: Number of Monte Carlo samples
        confidence: Confidence level (e.g., 0.95 for 95% confidence)
        seed: Random seed for reproducibility
    
    Returns:
        Mean predictions and lower/upper confidence bounds
    """
    # Generate random samples of the stenosis parameter
    key = jax.random.PRNGKey(seed)
    alpha_samples = jax.random.normal(key, (num_samples,)) * alpha_std + alpha_mean
    
    # Vectorize prediction function over alpha
    def predict_single(alpha):
        return jax.vmap(lambda x: model.apply(params, x, alpha))(domain_points)
    
    # Map over all parameter samples
    predictions = jax.vmap(predict_single)(alpha_samples)
    
    # Compute statistics
    mean_prediction = jnp.mean(predictions, axis=0)
    
    # Compute percentiles for confidence interval
    lower_percentile = 100 * (1 - confidence) / 2
    upper_percentile = 100 * (1 - (1 - confidence) / 2)
    
    lower_bound = jnp.percentile(predictions, lower_percentile, axis=0)
    upper_bound = jnp.percentile(predictions, upper_percentile, axis=0)
    
    return mean_prediction, lower_bound, upper_bound


def sensitivity_analysis(model, params, domain_points, parameter_ranges, num_samples=20, seed=0):
    """
    Perform sensitivity analysis by varying input parameters and observing effects on output.
    
    Args:
        model: Trained PINN model
        params: Trained parameters
        domain_points: Grid points in the domain for prediction
        parameter_ranges: Dictionary with parameter names as keys and (min, max) tuples as values
        num_samples: Number of samples for each parameter
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with parameter names as keys and sensitivity metrics as values
    """
    key = jax.random.PRNGKey(seed)
    
    # Initialize results dictionary
    results = {}
    
    # Baseline prediction with mean parameter values
    baseline_params = {name: (min_val + max_val) / 2 for name, (min_val, max_val) in parameter_ranges.items()}
    baseline_pred = predict_with_params(model, params, domain_points, baseline_params)
    
    # For each parameter, vary it and measure the effect
    for param_name, (min_val, max_val) in parameter_ranges.items():
        # Create samples for this parameter
        key, subkey = jax.random.split(key)
        param_samples = jnp.linspace(min_val, max_val, num_samples)
        
        # Initialize array to store predictions
        all_predictions = []
        
        # For each sample, make prediction
        for sample_val in param_samples:
            # Create parameter dict with current sample
            current_params = baseline_params.copy()
            current_params[param_name] = sample_val
            
            # Make prediction
            pred = predict_with_params(model, params, domain_points, current_params)
            all_predictions.append(pred)
        
        # Convert to array
        all_predictions = jnp.stack(all_predictions)
        
        # Compute sensitivity metrics
        param_range = max_val - min_val
        output_range = jnp.max(all_predictions, axis=0) - jnp.min(all_predictions, axis=0)
        sensitivity = output_range / param_range
        
        # Store results
        results[param_name] = {
            'sensitivity': sensitivity,
            'predictions': all_predictions,
            'param_samples': param_samples
        }
    
    return results


def predict_with_params(model, params, domain_points, param_dict):
    """
    Helper function to make predictions with a dictionary of parameters.
    
    Args:
        model: Trained PINN model
        params: Trained parameters
        domain_points: Grid points in the domain for prediction
        param_dict: Dictionary of parameter names and values
    
    Returns:
        Model predictions
    """
    # Extract alpha (assuming it's the main parameter)
    alpha = param_dict.get('alpha', 0.5)
    
    # Make predictions
    predictions = jax.vmap(lambda x: model.apply(params, x, alpha))(domain_points)
    
    return predictions


def plot_uncertainty(domain_points, mean_pred, std_pred, output_idx=0, save_path=None):
    """
    Plot uncertainty quantification results.
    
    Args:
        domain_points: Grid points in the domain
        mean_pred: Mean predictions
        std_pred: Standard deviation of predictions
        output_idx: Index of output to plot (0=u, 1=v, 2=p)
        save_path: Path to save the figure
    
    Returns:
        Figure handle
    """
    # Extract x and y coordinates
    x = domain_points[:, 0]
    y = domain_points[:, 1]
    
    # Extract mean and std for selected output
    mean_output = mean_pred[:, output_idx]
    std_output = std_pred[:, output_idx]
    
    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Determine unique x and y values for reshaping
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    
    # Reshape for contour plotting
    X, Y = np.meshgrid(unique_x, unique_y)
    Z_mean = mean_output.reshape(len(unique_y), len(unique_x))
    Z_std = std_output.reshape(len(unique_y), len(unique_x))
    
    # Plot mean field
    cf1 = axs[0].contourf(X, Y, Z_mean, 50, cmap='viridis')
    axs[0].set_title('Mean Field')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    plt.colorbar(cf1, ax=axs[0])
    
    # Plot standard deviation field
    cf2 = axs[1].contourf(X, Y, Z_std, 50, cmap='plasma')
    axs[1].set_title('Standard Deviation')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    plt.colorbar(cf2, ax=axs[1])
    
    # Add overall title based on output index
    output_names = ['Velocity (u)', 'Velocity (v)', 'Pressure (p)']
    plt.suptitle(f'Uncertainty Quantification: {output_names[output_idx]}', fontsize=16)
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def covariance_analysis(model, params, domain_points, alpha_mean=0.5, alpha_std=0.1, 
                        num_samples=100, seed=0):
    """
    Compute covariance matrix to analyze correlations between outputs at different locations.
    
    Args:
        model: Trained PINN model
        params: Trained parameters
        domain_points: Grid points in the domain for prediction
        alpha_mean: Mean stenosis severity parameter
        alpha_std: Standard deviation of stenosis parameter
        num_samples: Number of Monte Carlo samples
        seed: Random seed for reproducibility
    
    Returns:
        Mean predictions and covariance matrix
    """
    # Generate random samples of the stenosis parameter
    key = jax.random.PRNGKey(seed)
    alpha_samples = jax.random.normal(key, (num_samples,)) * alpha_std + alpha_mean
    
    # Vectorize prediction function over alpha
    def predict_single(alpha):
        return jax.vmap(lambda x: model.apply(params, x, alpha))(domain_points)
    
    # Map over all parameter samples
    predictions = jax.vmap(predict_single)(alpha_samples)
    
    # Compute mean
    mean_prediction = jnp.mean(predictions, axis=0)
    
    # Reshape predictions for covariance computation
    # (num_samples, num_points, num_outputs) -> (num_samples, num_points * num_outputs)
    num_samples, num_points, num_outputs = predictions.shape
    predictions_reshaped = predictions.reshape(num_samples, -1)
    
    # Compute covariance matrix
    covariance = jnp.cov(predictions_reshaped, rowvar=False)
    
    return mean_prediction, covariance


def analyze_extreme_cases(model, params, domain_points, alpha_values, output_idx=0):
    """
    Analyze extreme cases by evaluating model at specific parameter values.
    
    Args:
        model: Trained PINN model
        params: Trained parameters
        domain_points: Grid points in the domain for prediction
        alpha_values: List of alpha values to evaluate
        output_idx: Index of output to analyze (0=u, 1=v, 2=p)
    
    Returns:
        Dictionary of predictions for each alpha value
    """
    results = {}
    
    # For each alpha value, make prediction
    for alpha in alpha_values:
        pred = jax.vmap(lambda x: model.apply(params, x, alpha))(domain_points)
        results[alpha] = pred
    
    # Compute statistics
    all_preds = jnp.stack([results[alpha][:, output_idx] for alpha in alpha_values])
    min_val = jnp.min(all_preds)
    max_val = jnp.max(all_preds)
    range_val = max_val - min_val
    
    # Add statistics to results
    results['statistics'] = {
        'min': min_val,
        'max': max_val,
        'range': range_val
    }
    
    return results


def plot_extreme_cases(domain_points, predictions_dict, output_idx=0, save_path=None):
    """
    Plot results from extreme case analysis.
    
    Args:
        domain_points: Grid points in the domain
        predictions_dict: Dictionary of predictions for each alpha value
        output_idx: Index of output to plot (0=u, 1=v, 2=p)
        save_path: Path to save the figure
    
    Returns:
        Figure handle
    """
    # Extract alpha values
    alpha_values = [alpha for alpha in predictions_dict.keys() if alpha != 'statistics']
    
    # Create figure
    fig, axs = plt.subplots(1, len(alpha_values), figsize=(5*len(alpha_values), 5))
    
    # Extract x and y coordinates
    x = domain_points[:, 0]
    y = domain_points[:, 1]
    
    # Determine unique x and y values for reshaping
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    
    # Plot for each alpha value
    for i, alpha in enumerate(alpha_values):
        # Extract output
        output = predictions_dict[alpha][:, output_idx]
        
        # Reshape for contour plotting
        X, Y = np.meshgrid(unique_x, unique_y)
        Z = output.reshape(len(unique_y), len(unique_x))
        
        # Plot
        ax = axs[i] if len(alpha_values) > 1 else axs
        cf = ax.contourf(X, Y, Z, 50, cmap='viridis')
        ax.set_title(f'α = {alpha:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(cf, ax=ax)
    
    # Add overall title based on output index
    output_names = ['Velocity (u)', 'Velocity (v)', 'Pressure (p)']
    plt.suptitle(f'Extreme Case Analysis: {output_names[output_idx]}', fontsize=16)
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
