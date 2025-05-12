"""
Stenotic flow example for physics-constrained neural networks.

This script demonstrates the application of PINNs to flow through
a stenotic channel, which has a constriction that creates complex flow patterns.
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

from models import StenoticPINN
from training import train_step, initialize_training, generate_collocation_points_stenotic

def run_stenotic_example(alpha=0.5, rho=1.0, mu=0.01, 
                         n_iterations=5000, batch_size=200, learning_rate=1e-3,
                         features=[20, 20, 20, 3], adaptive=True, save_dir="results"):
    """
    Run the stenotic flow example with physics-constrained neural networks.
    
    Args:
        alpha: Stenosis severity parameter (0 to 1)
        rho: Fluid density
        mu: Dynamic viscosity
        n_iterations: Number of training iterations
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        features: Neural network architecture
        adaptive: Whether to use adaptive sampling
        save_dir: Directory to save results
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    print("Initializing model...")
    # Set up the model
    key = jax.random.PRNGKey(0)
    model = StenoticPINN(features=features)

    # Initialize model parameters with a dummy input
    dummy_input = jnp.ones((1, 2))
    params = model.init(key, dummy_input, alpha=alpha)

    # Create optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    # Initialize loss history
    loss_history = []
    x_momentum_history = []
    y_momentum_history = []
    continuity_history = []

    # Get apply function
    apply_fn = model.apply

    # Define residual computation functions
    def compute_momentum_residual(params, x, rho, mu, dim='x'):
        """Compute residuals of momentum equation without model reference"""
        
        # Define helper functions for extracting velocity and pressure components
        def u_fn(x): return apply_fn(params, x, alpha)[:, 0:1]
        def v_fn(x): return apply_fn(params, x, alpha)[:, 1:2]
        def p_fn(x): return apply_fn(params, x, alpha)[:, 2:3]
        
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
        def u_fn(x): return apply_fn(params, x, alpha)[:, 0:1]
        def v_fn(x): return apply_fn(params, x, alpha)[:, 1:2]
        
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
        # Generate collocation points for stenotic channel
        batch = generate_collocation_points_stenotic(batch_size, alpha=alpha, adaptive=adaptive)
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
                plt.savefig(f'{save_dir}/stenotic_training_progress_iter_{i+1}.png')
                plt.close()
                
                print(f"Loss plot saved to {save_dir}/stenotic_training_progress_iter_{i+1}.png")
        
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
    plt.savefig(f'{save_dir}/stenotic_final_training_progress.png')
    plt.close()

    print(f"Final loss plot saved to {save_dir}/stenotic_final_training_progress.png")
    print("Training complete!")

    # Create a grid for visualization
    print("Generating predictions...")
    nx, ny = 100, 50
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    points = np.stack([X.flatten(), Y.flatten()], axis=1)
    points = jnp.array(points)

    # Make predictions
    predictions = jax.vmap(lambda x: apply_fn(params, x, alpha))(points)

    # Extract velocity and pressure
    u = predictions[:, 0].reshape(ny, nx)
    v = predictions[:, 1].reshape(ny, nx)
    p = predictions[:, 2].reshape(ny, nx)

    # Compute velocity magnitude
    vel_mag = np.sqrt(u**2 + v**2)

    # Create plots
    print("Creating plots...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot velocity magnitude
    cf1 = axs[0, 0].contourf(X, Y, vel_mag, 50, cmap='viridis')
    # Draw the stenosis boundary
    y_top = 1.0 - alpha * np.exp(-50.0 * (X[0, :] - 0.5)**2)
    axs[0, 0].plot(X[0, :], y_top, 'k-', linewidth=2)
    axs[0, 0].plot(X[0, :], np.zeros_like(X[0, :]), 'k-', linewidth=2)
    axs[0, 0].set_title('Velocity Magnitude')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('y')
    plt.colorbar(cf1, ax=axs[0, 0])

    # Plot pressure field
    cf2 = axs[0, 1].contourf(X, Y, p, 50, cmap='plasma')
    # Draw the stenosis boundary
    axs[0, 1].plot(X[0, :], y_top, 'k-', linewidth=2)
    axs[0, 1].plot(X[0, :], np.zeros_like(X[0, :]), 'k-', linewidth=2)
    axs[0, 1].set_title('Pressure')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
    plt.colorbar(cf2, ax=axs[0, 1])

    # Plot velocity streamlines
    axs[1, 0].streamplot(X, Y, u, v, density=1, color='k')
    # Draw the stenosis boundary
    axs[1, 0].plot(X[0, :], y_top, 'r-', linewidth=2)
    axs[1, 0].plot(X[0, :], np.zeros_like(X[0, :]), 'r-', linewidth=2)
    axs[1, 0].set_title('Streamlines')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('y')

    # Plot vorticity
    # Compute vorticity (∂v/∂x - ∂u/∂y)
    u_reshaped = u.reshape(-1, 1)
    v_reshaped = v.reshape(-1, 1)
    
    def u_fn(x): return apply_fn(params, x, alpha)[:, 0:1]
    def v_fn(x): return apply_fn(params, x, alpha)[:, 1:2]
    
    # Compute derivatives for vorticity
    u_y = jax.vmap(lambda x: jax.jacfwd(u_fn, 0)(x[None, :])[:, :, 1])(points).reshape(ny, nx)
    v_x = jax.vmap(lambda x: jax.jacfwd(v_fn, 0)(x[None, :])[:, :, 0])(points).reshape(ny, nx)
    
    vorticity = v_x - u_y
    
    cf3 = axs[1, 1].contourf(X, Y, vorticity, 50, cmap='RdBu_r')
    # Draw the stenosis boundary
    axs[1, 1].plot(X[0, :], y_top, 'k-', linewidth=2)
    axs[1, 1].plot(X[0, :], np.zeros_like(X[0, :]), 'k-', linewidth=2)
    axs[1, 1].set_title('Vorticity')
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')
    plt.colorbar(cf3, ax=axs[1, 1])

    plt.tight_layout()
    plt.savefig(f'{save_dir}/stenotic_results_alpha_{alpha:.2f}.png', dpi=300)
    plt.show()

    print(f"Done! Results saved to {save_dir}/stenotic_results_alpha_{alpha:.2f}.png")
    
    # Save trained model
    import pickle
    with open(f'{save_dir}/stenotic_model_alpha_{alpha:.2f}.pkl', 'wb') as f:
        pickle.dump((model, params, alpha), f)
    print(f"Model saved to {save_dir}/stenotic_model_alpha_{alpha:.2f}.pkl")
    
    return model, params, alpha

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Stenotic flow example')
    parser.add_argument('--alpha', type=float, default=0.5, help='Stenosis severity parameter (0 to 1)')
    parser.add_argument('--rho', type=float, default=1.0, help='Fluid density')
    parser.add_argument('--mu', type=float, default=0.01, help='Dynamic viscosity')
    parser.add_argument('--iterations', type=int, default=5000, help='Number of training iterations')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--adaptive', type=bool, default=True, help='Use adaptive sampling')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    run_stenotic_example(
        alpha=args.alpha,
        rho=args.rho,
        mu=args.mu,
        n_iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        adaptive=args.adaptive,
        save_dir=args.save_dir
    )