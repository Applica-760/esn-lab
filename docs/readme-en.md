# esn-lab

A Python package for Echo State Network (ESN) experiments and training.

## Overview

esn-lab provides an implementation of Echo State Networks, a type of reservoir computing. It enables training and prediction of time series data using state vectors from the reservoir layer.

## Installation

```bash
pip install -e .
```

## Key Components

## Usage Example

※ For detailed usage examples, see ./projects

```python
from esn_lab import ESN, Tikhonov, train

# Initialize model
model = ESN(
    N_u=1,          # Input dimension
    N_y=1,          # Output dimension
    N_x=100,        # Number of reservoir nodes
    density=0.1,    # Connection density
    input_scale=1.0,
    rho=0.9         # Spectral radius
)

# Initialize optimizer
optimizer = Tikhonov(N_x=100, N_y=1, beta=1e-6)

# Execute training
output_weight = train(model, optimizer, U_list, D_list)

# Set output weights
model.Output.setweight(output_weight)
```

## License

MIT License

## Development Status

Development Status: Alpha
