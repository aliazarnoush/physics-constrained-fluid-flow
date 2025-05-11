import jax
import jax.numpy as jnp
import time
import numpy as np
import matplotlib.pyplot as plt
import optax
from models import PoiseuillePINN
from functools import partial

# Parameters
height = 1.0
dp_dx = -1.0
rho = 1.0
mu = 0.01
n_iterations = 5000
batch_size = 200 #1000
learning_rate = 1e-3

print("Initializing model...")
# Set up the model
key = jax.random.PRNGKey(0)
model = PoiseuillePINN(features=[20, 20, 20, 3], height=height, dp_dx=dp_dx)

# Domain bounds for Poiseuille flow
domain_bounds = [(0.0, 1.0), (0.0, height)]

# Initialize model parameters with a dummy input
dummy_input = jnp.ones((1, 2))
params = model.init(key, dummy_input)

# Create optimizer
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Define a function to compute residuals
def compute_momentum_residual(params, x, rho, mu, dim='x', apply_fn=None):
    """Compute residuals of momentum equation using automatic differentiation"""
    
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

def compute_continuity_residual(params, x, apply_fn=None):
    """Compute residuals of continuity equation (mass conservation)"""
    
    # Define helper functions for extracting velocity components
    def u_fn(x): return apply_fn(params, x)[:, 0:1]
    def v_fn(x): return apply_fn(params, x)[:, 1:2]
    
    # Compute derivatives
    u_x = jax.jacfwd(u_fn, 0)(x)[:, :, 0]
    v_y = jax.jacfwd(v_fn, 0)(x)[:, :, 1]
    
    # Continuity equation for incompressible flow: u_x + v_y = 0
    continuity = u_x + v_y
    
    return continuity

def compute_ns_residuals(params, x, rho, mu, apply_fn=None):
    """Compute all Navier-Stokes residuals for a batch of points."""
    x_momentum = compute_momentum_residual(params, x, rho, mu, dim='x', apply_fn=apply_fn)
    y_momentum = compute_momentum_residual(params, x, rho, mu, dim='y', apply_fn=apply_fn)
    continuity = compute_continuity_residual(params, x, apply_fn=apply_fn)
    
    return x_momentum, y_momentum, continuity

# Get a reference to the model.apply function
apply_fn = model.apply

# Define training step without jit for simplicity with apply_fn passed directly
@jax.jit
def train_step(params, opt_state, batch, rho, mu, lambda_cont=1.0):
    def loss_fn(params):
        """Loss function combining momentum and continuity residuals"""
        # Compute residuals for current batch 
        x_momentum, y_momentum, continuity = compute_ns_residuals(
            params, batch, rho, mu, apply_fn=apply_fn)
        
        # Compute mean squared residuals
        x_momentum_loss = jnp.mean(x_momentum**2)
        y_momentum_loss = jnp.mean(y_momentum**2)
        continuity_loss = jnp.mean(continuity**2)
        
        # Total loss with weighting
        total_loss = x_momentum_loss + y_momentum_loss + lambda_cont * continuity_loss
        
        return total_loss, (x_momentum_loss, y_momentum_loss, continuity_loss)
    
    # Compute loss and gradients
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    
    # Update parameters using optimizer
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss, aux

# Generate collocation points
def generate_collocation_points(domain_bounds, n_points):
    """Generate collocation points for training within the specified domain bounds."""
    # Uniform sampling across domain
    points = []
    for dim_bounds in domain_bounds:
        dim_min, dim_max = dim_bounds
        points.append(np.random.uniform(dim_min, dim_max, n_points))
    
    return np.stack(points, axis=1)


# Training loop with time estimation
print("Training for", n_iterations, "iterations...")
start_time = time.time()
for i in range(n_iterations):
    # Generate collocation points
    batch = generate_collocation_points(domain_bounds, batch_size)
    batch = jnp.array(batch)
    
    # Training step
    params, opt_state, loss, aux = train_step(params, opt_state, batch, rho, mu)
    
    # Print progress with time estimation
    if (i+1) % 50 == 0:
        elapsed_time = time.time() - start_time
        iterations_completed = i + 1
        iterations_remaining = n_iterations - iterations_completed
        
        # Estimate remaining time
        time_per_iteration = elapsed_time / iterations_completed
        estimated_remaining_time = time_per_iteration * iterations_remaining
        
        # Format as hours:minutes:seconds
        hours, remainder = divmod(estimated_remaining_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"Iteration {i+1}/{n_iterations}, Loss: {loss:.6f}")
        print(f"Elapsed: {elapsed_time:.1f}s, Estimated remaining: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
    # Print a simple progress dot every 10 iterations
    elif (i+1) % 10 == 0:
        print(".", end="", flush=True)

# Create a grid for visualization
print("Generating predictions...")
nx, ny = 100, 50
x = np.linspace(0, 1, nx)
y = np.linspace(0, height, ny)
X, Y = np.meshgrid(x, y)
points = np.stack([X.flatten(), Y.flatten()], axis=1)
points = jnp.array(points)

# Make predictions
predictions = jax.vmap(lambda x: apply_fn(params, x))(points)

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
plt.savefig('poiseuille_results.png', dpi=300)
plt.show()

print("Done! Results saved to poiseuille_results.png")

# Print final MSE
mse = np.mean((u[:, x_idx] - u_analytical)**2)
print(f"Mean Squared Error: {mse:.6e}")