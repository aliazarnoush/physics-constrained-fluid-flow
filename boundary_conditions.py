"""
Boundary condition implementations for physics-constrained neural networks.

This module provides specialized functions for enforcing various boundary conditions
directly in the network architecture for fluid flow problems.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, List, Dict


def create_channel_boundary_condition(height):
    """
    Create a boundary condition function for a straight channel.
    
    Args:
        height: Channel height
    
    Returns:
        Function that computes boundary distance modulation
    """
    def boundary_fn(x):
        # Extract y-coordinate
        y_coord = x[:, 1:2]
        
        # Distance to nearest wall (top or bottom)
        dist_to_bottom = y_coord
        dist_to_top = height - y_coord
        dist_to_wall = jnp.minimum(dist_to_bottom, dist_to_top)
        
        # Normalize distance
        normalized_dist = dist_to_wall / (height/2) * jnp.pi/2
        
        # Apply sine modulation for smooth transition
        # This ensures zero value at walls and smooth transition
        return jnp.sin(normalized_dist)
    
    return boundary_fn


def create_poiseuille_modulation(height):
    """
    Create a parabolic modulation for Poiseuille flow boundary conditions.
    
    This function creates a parabolic profile that is zero at both walls
    and reaches maximum in the center of the channel.
    
    Args:
        height: Channel height
    
    Returns:
        Function that computes parabolic modulation
    """
    def modulation_fn(x):
        # Extract y-coordinate
        y_coord = x[:, 1:2]
        
        # Parabolic profile: y * (height - y)
        # Normalized to reach maximum of 1 at center
        parabolic = y_coord * (height - y_coord)
        normalization = (height/2)**2
        
        return parabolic / normalization
    
    return modulation_fn


def create_stenotic_boundary_condition(alpha=0.5):
    """
    Create a boundary condition function for a stenotic channel.
    
    Args:
        alpha: Stenosis severity parameter (0 to 1)
    
    Returns:
        Function that computes boundary distance modulation
    """
    def boundary_fn(x):
        # Extract coordinates
        x_coord, y_coord = x[:, 0:1], x[:, 1:2]
        
        # Upper boundary with Gaussian constriction
        y_upper = 1.0 - alpha * jnp.exp(-50.0 * (x_coord - 0.5)**2)
        y_lower = jnp.zeros_like(y_coord)
        
        # Distance to nearest wall
        dist_to_upper = jnp.abs(y_coord - y_upper)
        dist_to_lower = jnp.abs(y_coord - y_lower)
        dist_to_wall = jnp.minimum(dist_to_upper, dist_to_lower)
        
        # Channel height at this x position
        channel_height = y_upper - y_lower
        
        # Normalize distance by local channel height
        normalized_dist = dist_to_wall / channel_height * jnp.pi
        
        # Apply sine modulation for smooth transition
        return jnp.sin(normalized_dist)
    
    return boundary_fn


def create_inflow_condition(profile_fn, transition_width=0.1):
    """
    Create an inflow boundary condition with smooth transition.
    
    Args:
        profile_fn: Function that defines the inflow velocity profile
        transition_width: Width of the transition region
    
    Returns:
        Function that applies the inflow condition
    """
    def inflow_fn(x, u):
        # Extract x-coordinate
        x_coord = x[:, 0:1]
        
        # Compute inflow profile
        inflow_profile = profile_fn(x)
        
        # Smooth transition: 1 at inflow (x=0), 0 elsewhere
        transition = jnp.exp(-(x_coord / transition_width)**2)
        
        # Apply transition: inflow at boundary, neural network elsewhere
        return transition * inflow_profile + (1 - transition) * u
    
    return inflow_fn


def create_parabolic_inflow_profile(height=1.0, max_velocity=1.0):
    """
    Create a parabolic inflow velocity profile function.
    
    Args:
        height: Channel height
        max_velocity: Maximum velocity at the center
    
    Returns:
        Function that computes parabolic velocity profile
    """
    def profile_fn(x):
        # Extract y-coordinate
        y_coord = x[:, 1:2]
        
        # Parabolic profile: 4 * y * (height - y) / height^2 * max_velocity
        # This gives 0 at y=0 and y=height, and max_velocity at y=height/2
        return 4.0 * max_velocity * y_coord * (height - y_coord) / (height**2)
    
    return profile_fn


def create_outflow_condition(domain_length, transition_width=0.1):
    """
    Create an outflow boundary condition with smooth transition.
    
    Args:
        domain_length: Length of the domain
        transition_width: Width of the transition region
    
    Returns:
        Function that applies the outflow condition
    """
    def outflow_fn(x, grad_p):
        # Extract x-coordinate
        x_coord = x[:, 0:1]
        
        # Smooth transition: 1 at outflow (x=domain_length), 0 elsewhere
        distance_to_outflow = domain_length - x_coord
        transition = jnp.exp(-(distance_to_outflow / transition_width)**2)
        
        # Apply transition: zero pressure gradient at outflow
        return (1 - transition) * grad_p
    
    return outflow_fn


def create_no_slip_condition(geometry_fn):
    """
    Create a no-slip boundary condition for arbitrary geometry.
    
    Args:
        geometry_fn: Function that returns boundary indicator (0 on boundary, >0 elsewhere)
    
    Returns:
        Function that enforces no-slip condition
    """
    def no_slip_fn(x, u, v):
        # Get boundary indicator
        boundary_indicator = geometry_fn(x)
        
        # Apply no-slip condition: zero velocity at boundary
        u_no_slip = u * boundary_indicator
        v_no_slip = v * boundary_indicator
        
        return u_no_slip, v_no_slip
    
    return no_slip_fn


def create_cylinder_boundary_condition(center_x, center_y, radius):
    """
    Create a boundary condition function for flow around a cylinder.
    
    Args:
        center_x: x-coordinate of cylinder center
        center_y: y-coordinate of cylinder center
        radius: Cylinder radius
    
    Returns:
        Function that computes boundary distance modulation
    """
    def boundary_fn(x):
        # Extract coordinates
        x_coord, y_coord = x[:, 0:1], x[:, 1:2]
        
        # Distance from point to cylinder center
        dist_to_center = jnp.sqrt((x_coord - center_x)**2 + (y_coord - center_y)**2)
        
        # Distance to cylinder surface
        dist_to_surface = jnp.abs(dist_to_center - radius)
        
        # For points inside the cylinder, set distance to 0
        inside_cylinder = dist_to_center < radius
        dist_to_surface = jnp.where(inside_cylinder, 0.0, dist_to_surface)
        
        # Normalize and apply modulation
        max_distance = 5.0 * radius  # Maximum distance for normalization
        normalized_dist = jnp.minimum(dist_to_surface / max_distance, 1.0) * jnp.pi/2
        
        return jnp.sin(normalized_dist)
    
    return boundary_fn


def create_aneurysm_boundary_condition(params):
    """
    Create a boundary condition function for aneurysmal flow.
    
    Args:
        params: Dictionary of aneurysm geometry parameters
            - vessel_radius: Base vessel radius
            - aneurysm_center_x: x-coordinate of aneurysm center
            - aneurysm_center_y: y-coordinate of aneurysm center
            - aneurysm_width: Width of the aneurysm
            - aneurysm_height: Height of the aneurysm
    
    Returns:
        Function that computes boundary distance modulation
    """
    def boundary_fn(x):
        # Extract parameters
        vessel_radius = params['vessel_radius']
        aneurysm_center_x = params['aneurysm_center_x']
        aneurysm_center_y = params['aneurysm_center_y']
        aneurysm_width = params['aneurysm_width']
        aneurysm_height = params['aneurysm_height']
        
        # Extract coordinates
        x_coord, y_coord = x[:, 0:1], x[:, 1:2]
        
        # Vessel centerline (assumed straight for simplicity)
        centerline_y = aneurysm_center_y - vessel_radius
        
        # Distance to vessel wall (without aneurysm)
        dist_to_vessel = jnp.abs(y_coord - centerline_y) - vessel_radius
        
        # Aneurysm shape (modeled as an elliptical bulge)
        # Distance to aneurysm boundary
        dx = (x_coord - aneurysm_center_x) / aneurysm_width
        dy = (y_coord - aneurysm_center_y) / aneurysm_height
        inside_aneurysm_region = (jnp.abs(x_coord - aneurysm_center_x) < aneurysm_width) & (y_coord > centerline_y)
        
        # Elliptical distance
        elliptical_dist = 1.0 - jnp.sqrt(dx**2 + dy**2)
        dist_to_aneurysm = -elliptical_dist  # Negative inside, positive outside
        
        # Combine vessel and aneurysm
        # Inside vessel or aneurysm: negative distance
        # Outside both: positive distance
        dist_to_boundary = jnp.where(inside_aneurysm_region, 
                                     jnp.minimum(dist_to_vessel, dist_to_aneurysm),
                                     dist_to_vessel)
        
        # Absolute distance to nearest boundary
        abs_dist = jnp.abs(dist_to_boundary)
        
        # For points inside the vessel or aneurysm, set distance appropriately
        inside_domain = dist_to_boundary < 0
        abs_dist = jnp.where(inside_domain, abs_dist, 0.0)
        
        # Normalize and apply modulation
        max_distance = vessel_radius  # Maximum distance for normalization
        normalized_dist = jnp.minimum(abs_dist / max_distance, 1.0) * jnp.pi/2
        
        return jnp.sin(normalized_dist)
    
    return boundary_fn


def apply_boundary_enforcement(u_nn, v_nn, p_nn, boundary_mask):
    """
    Apply boundary enforcement to neural network outputs.
    
    Args:
        u_nn: Raw u-velocity from neural network
        v_nn: Raw v-velocity from neural network
        p_nn: Raw pressure from neural network
        boundary_mask: Mask that enforces boundary conditions (0 at boundary, >0 elsewhere)
    
    Returns:
        Tuple of (u, v, p) with enforced boundary conditions
    """
    # Apply boundary conditions to velocity
    u = u_nn * boundary_mask
    v = v_nn * boundary_mask
    
    # Pressure is not directly constrained by no-slip
    p = p_nn
    
    return u, v, p


def enforce_inlet_outlet_conditions(u, v, p, x, inlet_fn, outlet_fn):
    """
    Enforce inlet and outlet boundary conditions.
    
    Args:
        u: u-velocity field
        v: v-velocity field
        p: Pressure field
        x: Coordinates
        inlet_fn: Function that applies inlet condition
        outlet_fn: Function that applies outlet condition
    
    Returns:
        Tuple of (u, v, p) with enforced inlet/outlet conditions
    """
    # Extract x-coordinate
    x_coord = x[:, 0:1]
    
    # Apply inlet velocity profile
    u = inlet_fn(x, u)
    
    # Apply outlet pressure gradient
    # For example, set zero pressure gradient at outlet
    p_x = jax.grad(lambda x: p[0])(x_coord)
    p_x = outlet_fn(x, p_x)
    
    return u, v, p