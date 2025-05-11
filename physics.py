"""
Functions for computing Navier-Stokes residuals using automatic differentiation in JAX.

These functions compute the residuals of the momentum and continuity equations
to enforce physics constraints during training without simulation data.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Dict, Any, Union


def compute_momentum_residual(model, params, x, rho, mu, dim='x'):
    """
    Compute residuals of momentum equation using automatic differentiation.
    
    Args:
        model: PINN model instance
        params: Model parameters
        x: Input coordinates
        rho: Fluid density
        mu: Dynamic viscosity
        dim: Dimension of momentum equation ('x' or 'y')
    
    Returns:
        Residual of momentum equation in the specified dimension
    """
    
    # Define helper functions for extracting velocity and pressure components
    def u_fn(x): return model.apply(params, x)[:, 0:1]
    def v_fn(x): return model.apply(params, x)[:, 1:2]
    def p_fn(x): return model.apply(params, x)[:, 2:3]
    
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


def compute_continuity_residual(model, params, x):
    """
    Compute residuals of continuity equation (mass conservation).
    
    Args:
        model: PINN model instance
        params: Model parameters
        x: Input coordinates
    
    Returns:
        Residual of continuity equation
    """
    
    # Define helper functions for extracting velocity components
    def u_fn(x): return model.apply(params, x)[:, 0:1]
    def v_fn(x): return model.apply(params, x)[:, 1:2]
    
    # Compute derivatives
    u_x = jax.jacfwd(u_fn, 0)(x)[:, :, 0]
    v_y = jax.jacfwd(v_fn, 0)(x)[:, :, 1]
    
    # Continuity equation for incompressible flow: u_x + v_y = 0
    continuity = u_x + v_y
    
    return continuity


def compute_ns_residuals(model, params, x, rho, mu):
    """
    Compute all Navier-Stokes residuals for a batch of points.
    
    Args:
        model: PINN model instance
        params: Model parameters
        x: Batch of input coordinates
        rho: Fluid density
        mu: Dynamic viscosity
    
    Returns:
        Tuple of (x_momentum_residual, y_momentum_residual, continuity_residual)
    """
    x_momentum = compute_momentum_residual(model, params, x, rho, mu, dim='x')
    y_momentum = compute_momentum_residual(model, params, x, rho, mu, dim='y')
    continuity = compute_continuity_residual(model, params, x)
    
    return x_momentum, y_momentum, continuity


def compute_second_derivatives(model, params, x):
    """
    Compute second-order spatial derivatives efficiently.
    
    This function optimizes the computation of higher-order derivatives
    using JAX's automatic differentiation.
    
    Args:
        model: PINN model instance
        params: Model parameters
        x: Input coordinates
    
    Returns:
        Tuple of second derivatives (u_xx, u_yy, v_xx, v_yy)
    """
    # First-order derivative functions
    def u_x_fn(x): 
        return jax.jacfwd(lambda x: model.apply(params, x)[:, 0], 0)(x)
    
    def u_y_fn(x): 
        return jax.jacfwd(lambda x: model.apply(params, x)[:, 1], 0)(x)
    
    def v_x_fn(x): 
        return jax.jacfwd(lambda x: model.apply(params, x)[:, 1], 0)(x)
    
    def v_y_fn(x): 
        return jax.jacfwd(lambda x: model.apply(params, x)[:, 1], 0)(x)
    
    # Second-order derivatives using forward-over-forward mode
    u_xx = jax.jacfwd(u_x_fn, 0)(x)
    u_yy = jax.jacfwd(u_y_fn, 0)(x)
    v_xx = jax.jacfwd(v_x_fn, 0)(x)
    v_yy = jax.jacfwd(v_y_fn, 0)(x)
    
    return u_xx, u_yy, v_xx, v_yy


def compute_vorticity(model, params, x):
    """
    Compute vorticity from velocity field.
    
    Args:
        model: PINN model instance
        params: Model parameters
        x: Input coordinates
    
    Returns:
        Vorticity field (∂v/∂x - ∂u/∂y)
    """
    # Define helper functions
    def u_fn(x): return model.apply(params, x)[:, 0:1]
    def v_fn(x): return model.apply(params, x)[:, 1:2]
    
    # Compute derivatives
    u_y = jax.jacfwd(u_fn, 0)(x)[:, :, 1]
    v_x = jax.jacfwd(v_fn, 0)(x)[:, :, 0]
    
    # Vorticity = ∂v/∂x - ∂u/∂y
    vorticity = v_x - u_y
    
    return vorticity


def compute_strain_rate(model, params, x):
    """
    Compute strain rate tensor from velocity field.
    
    Args:
        model: PINN model instance
        params: Model parameters
        x: Input coordinates
    
    Returns:
        Components of the strain rate tensor (ε_xx, ε_xy, ε_yx, ε_yy)
    """
    # Define helper functions
    def u_fn(x): return model.apply(params, x)[:, 0:1]
    def v_fn(x): return model.apply(params, x)[:, 1:2]
    
    # Compute derivatives
    u_x = jax.jacfwd(u_fn, 0)(x)[:, :, 0]
    u_y = jax.jacfwd(u_fn, 0)(x)[:, :, 1]
    v_x = jax.jacfwd(v_fn, 0)(x)[:, :, 0]
    v_y = jax.jacfwd(v_fn, 0)(x)[:, :, 1]
    
    # Strain rate components
    e_xx = u_x
    e_xy = 0.5 * (u_y + v_x)
    e_yx = e_xy  # Symmetric
    e_yy = v_y
    
    return e_xx, e_xy, e_yx, e_yy


def compute_wall_shear_stress(model, params, x, mu, geometry_fn):
    """
    Compute wall shear stress along boundaries.
    
    Args:
        model: PINN model instance
        params: Model parameters
        x: Input coordinates along the wall
        mu: Dynamic viscosity
        geometry_fn: Function that returns the wall normal directions
    
    Returns:
        Wall shear stress values
    """
    # Get wall normal directions
    normals = geometry_fn(x)  # Should return (nx, ny) components
    
    # Define helper functions
    def u_fn(x): return model.apply(params, x)[:, 0:1]
    def v_fn(x): return model.apply(params, x)[:, 1:2]
    
    # Compute velocity gradients
    grad_u = jax.jacfwd(u_fn, 0)(x)
    grad_v = jax.jacfwd(v_fn, 0)(x)
    
    # Extract components
    u_x, u_y = grad_u[:, :, 0], grad_u[:, :, 1]
    v_x, v_y = grad_v[:, :, 0], grad_v[:, :, 1]
    
    # Compute strain rate tensor
    e_xx = u_x
    e_xy = 0.5 * (u_y + v_x)
    e_yx = e_xy
    e_yy = v_y
    
    # Wall normal and tangential directions
    nx, ny = normals[:, 0], normals[:, 1]
    tx, ty = -ny, nx  # Tangent is perpendicular to normal
    
    # Shear stress is viscosity times the dot product of strain rate tensor with normal,
    # projected onto the tangential direction
    tau_w = mu * ((e_xx * nx + e_xy * ny) * tx + (e_yx * nx + e_yy * ny) * ty)
    
    return tau_w