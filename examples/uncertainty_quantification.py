"""
Uncertainty quantification for physics-constrained neural networks.

This script demonstrates how to perform Monte Carlo uncertainty quantification
for stenotic flow with varying stenosis parameters.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map  # Updated import for JAX compatibility
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import pickle

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import StenoticPINN
from uncertainty import monte_carlo_uq, plot_uncertainty, sensitivity_analysis

def run_uncertainty_quantification(model_path, alpha_mean=0.5, alpha_std=0.1, 
                                  num_samples=100, save_dir="results"):
    """
    Run uncertainty quantification analysis for stenotic flow.
    
    Args:
        model_path: Path to the pre-trained stenotic flow model
        alpha_mean: Mean value of stenosis parameter
        alpha_std: Standard deviation of stenosis parameter
        num_samples: Number of Monte Carlo samples
        save_dir: Directory to save results
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Loading model from {model_path}...")
    try:
        # Try to load model with three components (model, params, alpha)
        with open(model_path, 'rb') as f:
            model, params, _ = pickle.load(f)
    except ValueError:
        # If it's an older format with just (model, params)
        with open(model_path, 'rb') as f:
            model, params = pickle.load(f)
    
    print(f"Running Monte Carlo uncertainty quantification with {num_samples} samples...")
    
    # Create a grid for prediction
    nx, ny = 100, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.flatten(), Y.flatten()], axis=1)
    points = jnp.array(points)
    
    # Get start time
    start_time = time.time()
    
    # Define a safe prediction function that handles shape issues
    def safe_predict_with_alpha(x, alpha):
        # Add batch dimension if needed
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        try:
            result = model.apply(params, x, alpha)
            
            # Handle different output shapes
            if len(result.shape) == 1:
                # If output is just [u, v, p] without batch dimension
                if result.shape[0] == 3:
                    return result.reshape(1, 3)
                else:
                    # Unexpected shape
                    print(f"Warning: Unexpected output shape: {result.shape}")
                    return result
            return result
        except Exception as e:
            print(f"Error in prediction: {e}")
            return jnp.zeros((1, 3))  # Return zeros as fallback
    
    # Modified Monte Carlo UQ function with batching and error handling
    def robust_monte_carlo_uq(model, params, domain_points, alpha_mean, alpha_std, num_samples, batch_size=500):
        """Monte Carlo uncertainty quantification with batching and error handling"""
        # Generate random samples of the stenosis parameter
        key = jax.random.PRNGKey(0)
        alpha_samples = jax.random.normal(key, (num_samples,)) * alpha_std + alpha_mean
        
        # Initialize arrays to store all predictions
        all_predictions = []
        
        # Process each alpha sample
        for i, alpha in enumerate(alpha_samples):
            print(f"Processing sample {i+1}/{num_samples}, alpha = {alpha:.4f}")
            
            # Process points in batches
            sample_predictions = []
            
            for j in range(0, len(domain_points), batch_size):
                batch_end = min(j + batch_size, len(domain_points))
                batch_points = domain_points[j:batch_end]
                
                try:
                    # Try using vectorized prediction
                    batch_preds = jax.vmap(lambda x: safe_predict_with_alpha(x, alpha))(batch_points)
                    sample_predictions.append(batch_preds)
                except Exception as e:
                    print(f"Error in batch {j//batch_size + 1}: {e}")
                    # Fall back to non-vectorized prediction
                    batch_results = []
                    for k in range(len(batch_points)):
                        try:
                            pred = safe_predict_with_alpha(batch_points[k], alpha)
                            batch_results.append(pred)
                        except Exception as inner_e:
                            print(f"Error on point {k}: {inner_e}")
                            batch_results.append(jnp.zeros((1, 3)))
                    
                    sample_predictions.append(jnp.vstack(batch_results))
            
            # Combine all batches for this sample
            try:
                full_prediction = jnp.vstack(sample_predictions)
                all_predictions.append(full_prediction)
            except Exception as e:
                print(f"Error combining batches for sample {i}: {e}")
                # Create a fallback prediction of zeros
                full_prediction = jnp.zeros((len(domain_points), 3))
                all_predictions.append(full_prediction)
        
        # Stack all predictions into a single array: [num_samples, num_points, 3]
        try:
            predictions_array = jnp.stack(all_predictions)
            
            # Compute statistics
            mean_prediction = jnp.mean(predictions_array, axis=0)
            std_prediction = jnp.std(predictions_array, axis=0)
            
            return mean_prediction, std_prediction
        except Exception as e:
            print(f"Error computing statistics: {e}")
            # Return zeros as fallback
            return jnp.zeros((len(domain_points), 3)), jnp.zeros((len(domain_points), 3))
    
    # Run Monte Carlo uncertainty quantification with the robust version
    mean_pred, std_pred = robust_monte_carlo_uq(
        model=model, 
        params=params, 
        domain_points=points,
        alpha_mean=alpha_mean,
        alpha_std=alpha_std,
        num_samples=num_samples
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    print(f"Uncertainty quantification completed in {elapsed_time:.2f} seconds")
    
    # Plot uncertainty results for velocity components and pressure
    print("Generating uncertainty visualization plots...")
    
    # Create a function to visualize and save uncertainty results with stenosis boundary
    def visualize_uncertainty(output_idx, output_name):
        plt.figure(figsize=(15, 6))
        
        # Extract mean and std for selected output
        mean_output = mean_pred[:, output_idx].reshape(ny, nx)
        std_output = std_pred[:, output_idx].reshape(ny, nx)
        
        # Plot mean field
        plt.subplot(1, 2, 1)
        cf1 = plt.contourf(X, Y, mean_output, 50, cmap='viridis')
        # Draw the stenosis boundary for both mean and upper/lower alpha values
        y_top_mean = 1.0 - alpha_mean * np.exp(-50.0 * (X[0, :] - 0.5)**2)
        plt.plot(X[0, :], y_top_mean, 'k-', linewidth=2)
        plt.plot(X[0, :], np.zeros_like(X[0, :]), 'k-', linewidth=2)
        plt.title(f'Mean {output_name}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(cf1)
        
        # Plot standard deviation field
        plt.subplot(1, 2, 2)
        cf2 = plt.contourf(X, Y, std_output, 50, cmap='plasma')
        # Draw the stenosis boundary
        plt.plot(X[0, :], y_top_mean, 'k-', linewidth=2)
        plt.plot(X[0, :], np.zeros_like(X[0, :]), 'k-', linewidth=2)
        plt.title(f'Standard Deviation of {output_name}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(cf2)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/uncertainty_{output_name.lower().replace(" ", "_")}.png', dpi=300)
        plt.close()
        
        print(f"Saved uncertainty plot for {output_name} to {save_dir}/uncertainty_{output_name.lower().replace(' ', '_')}.png")
    
    # Visualize uncertainty for each output
    try:
        visualize_uncertainty(0, 'Velocity (u)')
        visualize_uncertainty(1, 'Velocity (v)')
        visualize_uncertainty(2, 'Pressure (p)')
    except Exception as e:
        print(f"Error visualizing uncertainty: {e}")
    
    # Calculate and visualize the coefficient of variation (CoV) for velocity magnitude
    try:
        print("Generating coefficient of variation plot...")
        
        plt.figure(figsize=(8, 6))
        
        # Extract mean and std for velocity components
        mean_u = mean_pred[:, 0].reshape(ny, nx)
        mean_v = mean_pred[:, 1].reshape(ny, nx)
        std_u = std_pred[:, 0].reshape(ny, nx)
        std_v = std_pred[:, 1].reshape(ny, nx)
        
        # Calculate velocity magnitude mean and std
        mean_vel_mag = np.sqrt(mean_u**2 + mean_v**2)
        
        # Approximate the std of velocity magnitude
        # Using error propagation formula: std_|v| ≈ √((u·std_u)² + (v·std_v)²) / |v|
        std_vel_mag = np.sqrt((mean_u * std_u)**2 + (mean_v * std_v)**2) / np.maximum(mean_vel_mag, 1e-6)
        
        # Calculate coefficient of variation (CoV = std / mean)
        cov = std_vel_mag / np.maximum(mean_vel_mag, 1e-6)
        
        # Plot CoV
        cf = plt.contourf(X, Y, cov, 50, cmap='hot_r')
        # Draw the stenosis boundary
        y_top_mean = 1.0 - alpha_mean * np.exp(-50.0 * (X[0, :] - 0.5)**2)
        plt.plot(X[0, :], y_top_mean, 'k-', linewidth=2)
        plt.plot(X[0, :], np.zeros_like(X[0, :]), 'k-', linewidth=2)
        plt.title('Coefficient of Variation for Velocity Magnitude')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.colorbar(cf, label='CoV')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/velocity_cov.png', dpi=300)
        plt.close()
        
        print(f"Saved coefficient of variation plot to {save_dir}/velocity_cov.png")
    except Exception as e:
        print(f"Error generating CoV plot: {e}")
    
    # Run sensitivity analysis to compare different alpha values
    try:
        print("Running sensitivity analysis for different stenosis parameters...")
        
        # Create several sample stenosis values
        alpha_values = [0.3, 0.5, 0.7]
        
        # Initialize arrays to store predictions for each alpha
        all_u = []
        all_v = []
        all_p = []
        all_vel_mag = []
        
        # Function to predict with a specific alpha (with batching and error handling)
        def predict_with_alpha(alpha, batch_size=500):
            all_batch_predictions = []
            
            for i in range(0, len(points), batch_size):
                batch_end = min(i + batch_size, len(points))
                batch_points = points[i:batch_end]
                
                try:
                    # Try using vectorized prediction
                    batch_preds = jax.vmap(lambda x: safe_predict_with_alpha(x, alpha))(batch_points)
                except Exception as e:
                    print(f"Error in batch {i//batch_size + 1}: {e}")
                    # Fall back to non-vectorized prediction
                    batch_results = []
                    for j in range(len(batch_points)):
                        try:
                            pred = safe_predict_with_alpha(batch_points[j], alpha)
                            batch_results.append(pred)
                        except Exception as inner_e:
                            print(f"Error on point {j}: {inner_e}")
                            batch_results.append(jnp.zeros((1, 3)))
                    
                    batch_preds = jnp.vstack(batch_results)
                
                all_batch_predictions.append(batch_preds)
            
            # Combine all batches
            try:
                predictions = jnp.vstack(all_batch_predictions)
                u = predictions[:, 0].reshape(ny, nx)
                v = predictions[:, 1].reshape(ny, nx)
                p = predictions[:, 2].reshape(ny, nx)
                vel_mag = np.sqrt(u**2 + v**2)
                return u, v, p, vel_mag
            except Exception as e:
                print(f"Error combining batches: {e}")
                # Return zeros as fallback
                zeros = np.zeros((ny, nx))
                return zeros, zeros, zeros, zeros
        
        # Get predictions for each alpha value
        for alpha in alpha_values:
            print(f"Processing alpha = {alpha}")
            u, v, p, vel_mag = predict_with_alpha(alpha)
            all_u.append(u)
            all_v.append(v)
            all_p.append(p)
            all_vel_mag.append(vel_mag)
        
        # Create comparison plots
        fig, axs = plt.subplots(2, len(alpha_values), figsize=(15, 10))
        
        # Plot velocity magnitude for each alpha
        for i, alpha in enumerate(alpha_values):
            # Get the correct subplot
            ax = axs[0, i]
            
            # Plot velocity magnitude
            cf = ax.contourf(X, Y, all_vel_mag[i], 50, cmap='viridis')
            
            # Draw stenosis boundary
            y_top = 1.0 - alpha * np.exp(-50.0 * (X[0, :] - 0.5)**2)
            ax.plot(X[0, :], y_top, 'k-', linewidth=2)
            ax.plot(X[0, :], np.zeros_like(X[0, :]), 'k-', linewidth=2)
            
            ax.set_title(f'Velocity Magnitude (α = {alpha:.1f})')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            # Add colorbar
            plt.colorbar(cf, ax=ax)
        
        # Plot pressure for each alpha
        for i, alpha in enumerate(alpha_values):
            # Get the correct subplot
            ax = axs[1, i]
            
            # Plot pressure
            cf = ax.contourf(X, Y, all_p[i], 50, cmap='plasma')
            
            # Draw stenosis boundary
            y_top = 1.0 - alpha * np.exp(-50.0 * (X[0, :] - 0.5)**2)
            ax.plot(X[0, :], y_top, 'k-', linewidth=2)
            ax.plot(X[0, :], np.zeros_like(X[0, :]), 'k-', linewidth=2)
            
            ax.set_title(f'Pressure (α = {alpha:.1f})')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            # Add colorbar
            plt.colorbar(cf, ax=ax)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/alpha_comparison.png', dpi=300)
        plt.close()
        
        print(f"Saved parameter comparison plot to {save_dir}/alpha_comparison.png")
    except Exception as e:
        print(f"Error in sensitivity analysis: {e}")
    
    # Create velocity profiles at different x-locations
    try:
        print("Generating velocity profiles with uncertainty bands...")
        
        # Select x-locations for profiles
        x_locations = [0.25, 0.5, 0.75]  # Before, at, and after stenosis
        
        plt.figure(figsize=(15, 5))
        
        for i, x_loc in enumerate(x_locations):
            # Find the nearest x-index
            x_idx = np.abs(x - x_loc).argmin()
            
            # Get the mean and std of u-velocity at this x-location
            mean_u_profile = mean_u[:, x_idx]
            std_u_profile = std_u[:, x_idx]
            
            # Plot profile with uncertainty
            plt.subplot(1, 3, i+1)
            plt.plot(mean_u_profile, y, 'b-', linewidth=2, label='Mean')
            plt.fill_betweenx(y, mean_u_profile - 2*std_u_profile, mean_u_profile + 2*std_u_profile, 
                             color='b', alpha=0.2, label='95% CI')
            
            # Draw the stenosis location
            y_loc = 1.0 - alpha_mean * np.exp(-50.0 * (x_loc - 0.5)**2)
            plt.axhline(y=y_loc, color='r', linestyle='--', label='Stenosis')
            plt.axhline(y=0, color='r', linestyle='--')
            
            plt.title(f'u-Velocity Profile at x = {x_loc:.2f}')
            plt.xlabel('u-Velocity')
            plt.ylabel('y')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/velocity_profiles.png', dpi=300)
        plt.close()
        
        print(f"Saved velocity profiles plot to {save_dir}/velocity_profiles.png")
    except Exception as e:
        print(f"Error generating velocity profiles: {e}")
    
    return mean_pred, std_pred

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run uncertainty quantification')
    parser.add_argument('--model_path', type=str, default='results/stenotic_model_alpha_0.50.pkl', 
                      help='Path to pre-trained model')
    parser.add_argument('--alpha_mean', type=float, default=0.5, help='Mean stenosis parameter')
    parser.add_argument('--alpha_std', type=float, default=0.1, help='Std dev of stenosis parameter')
    parser.add_argument('--samples', type=int, default=100, help='Number of Monte Carlo samples')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    run_uncertainty_quantification(
        model_path=args.model_path,
        alpha_mean=args.alpha_mean,
        alpha_std=args.alpha_std,
        num_samples=args.samples,
        save_dir=args.save_dir
    )