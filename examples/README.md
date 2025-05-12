# Physics-Constrained Neural Networks for Fluid Flow - Examples

This directory contains example scripts demonstrating the physics-constrained neural network approach for fluid flow simulations without requiring training data. These examples leverage JAX for automatic differentiation and enforce the Navier-Stokes equations directly in the loss function.

## Available Examples

### 1. Poiseuille Flow (`poiseuille_flow.py`)

A simple parallel plate channel flow with an analytical solution for validation.

```bash
python poiseuille_flow.py [options]
```

### 2. Stenotic Flow (`stenotic_flow.py`)

Flow through a channel with a Gaussian constriction (stenosis), demonstrating the approach for more complex geometries.

```bash
python stenotic_flow.py [options]
```

### 3. Uncertainty Quantification (`uncertainty_quantification.py`)

Performs Monte Carlo uncertainty quantification on a pre-trained stenotic flow model, analyzing the impact of stenosis parameter variations.

```bash
python uncertainty_quantification.py [options]
```

## Common Command-Line Options

All examples support the following standard options:

| Option | Description | Default |
|--------|-------------|---------|
| `--iterations` | Number of training iterations | 5000 |
| `--batch_size` | Number of collocation points per batch | 200 |
| `--learning_rate` | Learning rate for optimizer | 0.001 |
| `--save_dir` | Directory to save results | results |

## Example-Specific Options

### Poiseuille Flow Options

| Option | Description | Default |
|--------|-------------|---------|
| `--height` | Channel height | 1.0 |
| `--dp_dx` | Pressure gradient | -1.0 |
| `--rho` | Fluid density | 1.0 |
| `--mu` | Dynamic viscosity | 0.01 |

Example:
```bash
python poiseuille_flow.py --batch_size 100 --iterations 2000 --height 2.0 --mu 0.005
```

### Stenotic Flow Options

| Option | Description | Default |
|--------|-------------|---------|
| `--alpha` | Stenosis severity parameter (0 to 1) | 0.5 |
| `--rho` | Fluid density | 1.0 |
| `--mu` | Dynamic viscosity | 0.01 |
| `--adaptive` | Use adaptive collocation sampling | True |

Example:
```bash
python stenotic_flow.py --alpha 0.7 --batch_size 200 --iterations 3000 --adaptive True
```

### Uncertainty Quantification Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model_path` | Path to pre-trained stenotic model | results/stenotic_model_alpha_0.50.pkl |
| `--alpha_mean` | Mean value of stenosis parameter | 0.5 |
| `--alpha_std` | Standard deviation of stenosis parameter | 0.1 |
| `--samples` | Number of Monte Carlo samples | 100 |

Example:
```bash
python uncertainty_quantification.py --model_path results/stenotic_model_alpha_0.50.pkl --samples 200 --alpha_std 0.2
```

## Tips for Running Examples

### Memory Considerations

For machines with limited memory:

1. **Reduce batch size**: The batch size significantly impacts memory usage
   ```bash
   python poiseuille_flow.py --batch_size 50
   ```

2. **Use a smaller network**: Modify the code to use fewer or smaller layers
   ```python
   # Change from
   model = PoiseuillePINN(features=[20, 20, 20, 3], height=height, dp_dx=dp_dx)
   # To
   model = PoiseuillePINN(features=[10, 10, 3], height=height, dp_dx=dp_dx)
   ```

3. **Run for fewer iterations during testing**:
   ```bash
   python poiseuille_flow.py --iterations 500
   ```

### Performance Optimization

1. **Chunk size adjustment**: You can modify the `chunk_size` parameter in the `chunked_train_step` function to balance memory usage and performance

2. **Adaptive sampling**: For stenotic flow, try toggling the adaptive sampling
   ```bash
   python stenotic_flow.py --adaptive False
   ```

3. **JAX JIT compilation**: These examples use JAX's just-in-time compilation for speed. The first run may be slower due to compilation overhead.

## Expected Outputs

Each example generates:

1. **Training progress updates**: Displayed in the console while running
2. **Loss plots**: Saved periodically during training and at completion  
3. **Visualization plots**: Generated at the end of training
4. **Saved model**: Stored as a pickle file for later use/analysis

All outputs are saved to the specified `save_dir` (defaults to "results").

## Extending the Examples

These examples can be modified to explore:

1. Different geometries by changing the boundary condition functions
2. New physics by modifying the residual calculations  
3. Parameter studies by varying the physical parameters
4. Multi-physics coupling by adding additional physics terms to the loss

Feel free to use these examples as starting points for your own physics-constrained deep learning projects!
