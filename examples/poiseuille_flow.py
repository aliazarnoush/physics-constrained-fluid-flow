"""
Poiseuille flow example for physics-constrained neural networks.

This script demonstrates the application of PINNs to Poiseuille flow between
parallel plates, which has a known analytical solution for validation.
"""

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map  # Updated import for JAX compatibility
import numpy as np
import matplotlib.pyplot as plt
import optax
import time
import os
import sys

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import PoiseuillePINN
from training import train_step, initialize_training, generate_collocation_points

def run_poiseuille_example(height=1.0, dp_dx=-1.0, rho=1.0, mu=0.01, 
                          n_iterations=5000, batch_size=200, learning_rate=1e-3,
                          features=[20, 20, 20, 3], save_dir="results"):
    """
    Run the Poiseuille flow example with physics-constrained neural networks.
    
    Args:
        height: Channel height
        dp_dx: Pressure gradient
        rho: Fluid density
        mu: Dynamic viscosity
        n_iterations: Number of training iterations
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        features: Neural network architecture
        save_dir: Directory to save results
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print("Initializing model...")
    # Set up the model
    key = jax.random.PRNGKey(0)
    model = PoiseuillePINN(features=features, height=height, dp_dx=dp_dx)

    # Domain bounds for Poiseuille flow
    domain_bounds = [(0.0, 1.0), (0.0, height)]

    # Initialize model parameters and optimizer
    params, optimizer, opt_state = initialize_training(model, seed=0, learning_rate=learning_rate)

    # Initialize loss history
    loss_history = []
    x_momentum_history = []
    y_momentum_history = []
    continuity_history = []

    # Get apply function
    apply_fn = model.apply

    # Define residual computation functions that take apply_fn directly
    def compute_momentum_residual(params, x, rho, mu, dim='x'):
        """Compute residuals of momentum equation without model reference"""
        
        # Define helper functions for extracting velocity and pressure components
        def u_fn(x): return apply_fn(params, x)[:, 0:1]
        def v_fn(x): return apply_fn(params, x)[:, 1:2]
        def p_fn(x): return apply_fn(params, x)[:, 2:3]
        
        # Compute first-order derivatives
        u_x = jax.jacfwd(u_fn, 0)(x)[:, :, 0]
        u_y = jax.jacfwd(u_fn, 0)(x)[:, :, 1]
        v_x = jax.jacfwd(v_fn, 0)(x)[:, :, 0]
        v_y = jax.jacfwd(v_fn, 0)(x)[:, :, 1]
        p_x = jax.jacfwd(p_fn, 0)(x)[:, :, 0]
        p_y = jax.jacfwd(p_fn, 0)(x)[:, :, 1]
        
        # Compute second-order derivatives based on dimension
        if dim == 'x':
            # X-momentum equation: rho * (u * u_x + v * u_y) = -p_x + mu * (u_xx + u_yy)
            u_xx = jax.jacfwd(lambda x: jax.jacfwd(u_fn, 0)(x)[:, :, 0], 0)(x)[:, :, 0]
            u_yy = jax.jacfwd(lambda x: jax.jacfwd(u_fn, 0)(x)[:, :, 1], 0)(x)[:, :, 1]
            
            # Compute residual: difference between LHS and RHS should be zero
            momentum = rho * (u_fn(x) * u_x + v_fn(x) * u_y) + p_x - mu * (u_xx + u_yy)
        else:
            # Y-momentum equation: rho * (u * v_x + v * v_y) = -p_y + mu * (v_xx + v_yy)
            v_xx = jax.jacfwd(lambda x: jax.jacfwd(v_fn, 0)(x)[:, :, 0], 0)(x)[:, :, 0]
            v_yy = jax.jacfwd(lambda x: jax.jacfwd(v_fn, 0)(x)[:, :, 1], 0)(x)[:, :, 1]
            
            # Compute residual: difference between LHS and RHS should be zero
            momentum = rho * (u_fn(x) * v_x + v_fn(x) * v_y) + p_y - mu * (v_xx + v_yy)
        
        return momentum

    def compute_continuity_residual(params, x):
        """Compute residuals of continuity equation without model reference"""
        
        # Define helper functions for extracting velocity components
        def u_fn(x): return apply_fn(params, x)[:, 0:1]
        def v_fn(x): return apply_fn(params, x)[:, 1:2]
        
        # Compute derivatives
        u_x = jax.jacfwd(u_fn, 0)(x)[:, :, 0]
        v_y = jax.jacfwd(v_fn, 0)(x)[:, :, 1]
        
        # Continuity equation for incompressible flow: u_x + v_y = 0
        continuity = u_x + v_y
        
        return continuity

    def compute_ns_residuals(params, x, rho, mu):
        """Compute all Navier-Stokes residuals for a batch of points."""
        x_momentum = compute_momentum_residual(params, x, rho, mu, dim='x')
        y_momentum = compute_momentum_residual(params, x, rho, mu, dim='y')
        continuity = compute_continuity_residual(params, x)
        
        return x_momentum, y_momentum, continuity

    # Define chunked training step that computes gradients in smaller batches
    def chunked_train_step(params, opt_state, full_batch, rho, mu, chunk_size=50, lambda_cont=1.0):
        """Training step that processes data in smaller chunks to reduce memory usage."""
        n_points = full_batch.shape[0]
        n_chunks = (n_points + chunk_size - 1) // chunk_size  # Ceiling division
        
        # Initialize average loss values
        avg_loss = 0.0
        avg_x_momentum_loss = 0.0
        avg_y_momentum_loss = 0.0
        avg_continuity_loss = 0.0
        
        # Initialize gradient accumulator
        grads_acc = None
        
        # Process each chunk separately
        for i in range(n_chunks):
            # Extract chunk
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, n_points)
            chunk = full_batch[start_idx:end_idx]
            
            # Define loss function for this chunk
            def chunk_loss_fn(p):
                # Compute residuals for this chunk
                x_momentum, y_momentum, continuity = compute_ns_residuals(p, chunk, rho, mu)
                
                # Compute mean squared residuals
                x_momentum_loss = jnp.mean(x_momentum**2)
                y_momentum_loss = jnp.mean(y_momentum**2)
                continuity_loss = jnp.mean(continuity**2)
                
                # Total loss with weighting
                total_loss = x_momentum_loss + y_momentum_loss + lambda_cont * continuity_loss
                
                return total_loss, (x_momentum_loss, y_momentum_loss, continuity_loss)
            
            # Compute loss and gradients for this chunk
            (chunk_loss, chunk_aux), chunk_grads = jax.value_and_grad(chunk_loss_fn, has_aux=True)(params)
            
            # Weight by chunk size for proper averaging
            chunk_weight = (end_idx - start_idx) / n_points
            
            # Accumulate weighted losses
            avg_loss += chunk_loss * chunk_weight
            avg_x_momentum_loss += chunk_aux[0] * chunk_weight
            avg_y_momentum_loss += chunk_aux[1] * chunk_weight
            avg_continuity_loss += chunk_aux[2] * chunk_weight
            
            # Accumulate gradients (weighted by chunk size)
            if grads_acc is None:
                grads_acc = tree_map(lambda g: g * chunk_weight, chunk_grads)
            else:
                grads_acc = tree_map(lambda g1, g2: g1 + g2 * chunk_weight, grads_acc, chunk_grads)
        
        # Apply accumulated gradients
        updates, new_opt_state = optimizer.update(grads_acc, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, avg_loss, (avg_x_momentum_loss, avg_y_momentum_loss, avg_continuity_loss)

    # For better time estimation
    recent_times = []
    max_times_to_track = 100  # Only use last 100 iterations for time estimation

    # Training loop with progress tracking
    print("Training for", n_iterations, "iterations...")
    start_time = time.time()
    for i in range(n_iterations):
        # Generate collocation points
        batch = generate_collocation_points(domain_bounds, batch_size)
        batch = jnp.array(batch)
        
        # Training step using chunked implementation for memory efficiency
        params, opt_state, loss, aux = chunked_train_step(params, opt_state, batch, rho, mu)
        
        # Record losses
        loss_history.append(float(loss))
        x_momentum_history.append(float(aux[0]))
        y_momentum_history.append(float(aux[1]))
        continuity_history.append(float(aux[2]))
        
        # Print progress with time estimation
        if (i+1) % 50 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            iter_time = elapsed_time / (i+1)  # Average time per iteration
            
            # Track recent iteration times
            if len(recent_times) >= max_times_to_track:
                recent_times.pop(0)  # Remove oldest time
            recent_times.append(iter_time)
            
            # Use recent average for estimation
            recent_avg_time = sum(recent_times) / len(recent_times) if recent_times else iter_time
            iterations_remaining = n_iterations - (i+1)
            estimated_remaining_time = recent_avg_time * iterations_remaining
            
            # Format as hours:minutes:seconds
            hours, remainder = divmod(estimated_remaining_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            print(f"Iteration {i+1}/{n_iterations}, Loss: {loss:.6f}")
            print(f"Elapsed: {elapsed_time:.1f}s, Estimated remaining: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            
            # Plot loss curves every 500 iterations
            if (i+1) % 500 == 0:
                plt.figure(figsize=(12, 8))
                
                # Total loss
                plt.subplot(2, 2, 1)
                plt.semilogy(loss_history)
                plt.title('Total Loss')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.grid(True)
                
                # Component losses
                plt.subplot(2, 2, 2)
                plt.semilogy(x_momentum_history, label='X-Momentum')
                plt.semilogy(y_momentum_history, label='Y-Momentum')
                plt.semilogy(continuity_history, label='Continuity')
                plt.title('Component Losses')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True)
                
                # Save the plot
                plt.tight_layout()
                plt.savefig(f'{save_dir}/training_progress_iter_{i+1}.png')
                plt.close()
                
                print(f"Loss plot saved to {save_dir}/training_progress_iter_{i+1}.png")
        
        # Print a simple progress dot every 10 iterations
        elif (i+1) % 10 == 0:
            print(".", end="", flush=True)

    # Final loss plot after training
    plt.figure(figsize=(12, 8))

    # Total loss
    plt.subplot(2, 2, 1)
    plt.semilogy(loss_history)
    plt.title('Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)

    # Component losses
    plt.subplot(2, 2, 2)
    plt.semilogy(x_momentum_history, label='X-Momentum')
    plt.semilogy(y_momentum_history, label='Y-Momentum')
    plt.semilogy(continuity_history, label='Continuity')
    plt.title('Component Losses')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the final plot
    plt.tight_layout()
    plt.savefig(f'{save_dir}/final_training_progress.png')
    plt.close()

    print(f"Final loss plot saved to {save_dir}/final_training_progress.png")
    print("Training complete!")

    # Create a grid for visualization
    print("Generating predictions...")
    nx, ny = 100, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, height, ny)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.flatten(), Y.flatten()], axis=1)
    points = jnp.array(points)

    # Make predictions with error handling for shape issues
    def safe_predict(x):
        # Add batch dimension if needed
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        result = apply_fn(params, x)
        
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

    # Process points in smaller batches to avoid memory issues
    print("Processing predictions in batches...")
    pred_batch_size = 500
    all_predictions = []
    
    for i in range(0, len(points), pred_batch_size):
        batch_end = min(i + pred_batch_size, len(points))
        batch_points = points[i:batch_end]
        print(f"Processing batch {i//pred_batch_size + 1}/{(len(points) + pred_batch_size - 1)//pred_batch_size}...")
        
        try:
            batch_preds = jax.vmap(safe_predict)(batch_points)
            all_predictions.append(batch_preds)
        except Exception as e:
            print(f"Error processing batch: {e}")
            # Try with non-vectorized version as fallback
            batch_results = []
            for j in range(len(batch_points)):
                try:
                    pred = safe_predict(batch_points[j])
                    batch_results.append(pred)
                except Exception as inner_e:
                    print(f"Error on point {j}: {inner_e}")
                    # Use a placeholder with zeros
                    batch_results.append(jnp.zeros(3))
            
            all_predictions.append(jnp.array(batch_results))
    
    try:
        predictions = jnp.vstack(all_predictions)
        
        # Extract velocity and pressure
        u = predictions[:, 0].reshape(ny, nx)
        v = predictions[:, 1].reshape(ny, nx)
        p = predictions[:, 2].reshape(ny, nx)
        
        # Compute analytical solution for comparison
        y_vals = np.linspace(0, height, ny)
        u_analytical = (dp_dx / (2 * mu)) * y_vals * (height - y_vals)
        
        # Create plots
        print("Creating plots...")
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot velocity field
        cf1 = axs[0, 0].contourf(X, Y, u, 50, cmap='viridis')
        axs[0, 0].set_title('Velocity u (PINN)')
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('y')
        plt.colorbar(cf1, ax=axs[0, 0])
        
        # Plot pressure field
        cf2 = axs[0, 1].contourf(X, Y, p, 50, cmap='plasma')
        axs[0, 1].set_title('Pressure (PINN)')
        axs[0, 1].set_xlabel('x')
        axs[0, 1].set_ylabel('y')
        plt.colorbar(cf2, ax=axs[0, 1])
        
        # Plot velocity profile at x = 0.5
        x_idx = nx // 2
        axs[1, 0].plot(u[:, x_idx], y, 'r-', label='PINN')
        axs[1, 0].plot(u_analytical, y_vals, 'b--', label='Analytical')
        axs[1, 0].set_title('Velocity Profile at x = 0.5')
        axs[1, 0].set_xlabel('u')
        axs[1, 0].set_ylabel('y')
        axs[1, 0].legend()
        
        # Plot error
        u_error = np.abs(u[:, x_idx] - u_analytical)
        axs[1, 1].semilogy(y, u_error, 'k-')
        axs[1, 1].set_title('Absolute Error')
        axs[1, 1].set_xlabel('y')
        axs[1, 1].set_ylabel('|Error|')
        axs[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/poiseuille_results.png', dpi=300)
        plt.show()
        
        print(f"Done! Results saved to {save_dir}/poiseuille_results.png")
        
        # Print final MSE
        mse = np.mean((u[:, x_idx] - u_analytical)**2)
        print(f"Mean Squared Error: {mse:.6e}")
        
        # Save trained model
        import pickle
        with open(f'{save_dir}/poiseuille_model.pkl', 'wb') as f:
            pickle.dump((model, params), f)
        print(f"Model saved to {save_dir}/poiseuille_model.pkl")
        
        return model, params, mse
    
    except Exception as e:
        print(f"Error in generating results: {e}")
        # Save the trained model anyway so we don't lose the training
        import pickle
        with open(f'{save_dir}/poiseuille_model.pkl', 'wb') as f:
            pickle.dump((model, params), f)
        print(f"Model saved to {save_dir}/poiseuille_model.pkl despite visualization error")
        
        return model, params, None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Poiseuille flow example')
    parser.add_argument('--height', type=float, default=1.0, help='Channel height')
    parser.add_argument('--dp_dx', type=float, default=-1.0, help='Pressure gradient')
    parser.add_argument('--rho', type=float, default=1.0, help='Fluid density')
    parser.add_argument('--mu', type=float, default=0.01, help='Dynamic viscosity')
    parser.add_argument('--iterations', type=int, default=5000, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    run_poiseuille_example(
        height=args.height,
        dp_dx=args.dp_dx,
        rho=args.rho,
        mu=args.mu,
        n_iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_dir=args.save_dir
    )