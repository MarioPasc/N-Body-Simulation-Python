# N-Body Simulation Project

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)


## Overview
This project explores the N-Body problem, a classic issue in various scientific and engineering disciplines, including physics, proteomics, and machine learning. Due to the lack of an analytical solution, the N-Body problem is approached through numerical methods and simulations to estimate the movements of bodies influenced by gravitational forces.

The main objective of this project is to implement and compare different strategies for N-Body simulations using Euler's method. This includes sequential and concurrent calculations on CPU, and parallel computations on GPU.

## Features
- **Sequential CPU Implementation**: Utilizes a straightforward approach to simulate body interactions.
- **Concurrent CPU Implementation**: Attempts to leverage multiple CPU cores despite Python’s GIL limitations.
- **GPU Parallel Implementation**: Employs CUDA to harness the computational power of modern GPUs for efficient simulations.

## Installation

Clone the repository to your local machine:
```bash
git clone https://github.com/MarioPasc/N-Body-Simulation-Python.git
cd n-body-simulation-python
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Results

### Overall Complexity

<p align="center">
  <img src="https://github.com/MarioPasc/N-Body-Simulation-Python/assets/120520768/cecc775b-60b3-4daf-a582-76d979bbb890" alt="Overall"/>
</p>

## GPU Complexity
<p align="center">
  <img src="https://github.com/MarioPasc/N-Body-Simulation-Python/assets/120520768/428721b7-01e6-4734-903e-22886cd9317d" alt="Overall"/>
</p>

## Simulation Examples

### Stable 2 Body System

<p align="center">
  <img src="https://github.com/MarioPasc/N-Body-Simulation-Python/assets/120520768/8c682869-c29c-477e-ac4b-1f865a530d3b" alt="Overall"/>
</p>

### 3 Body System

<p align="center">
  <img src="https://github.com/MarioPasc/N-Body-Simulation-Python/assets/120520768/c030cf14-c64c-4b52-931c-7481f594b92d" alt="Overall"/>
</p>

### Lagrange Points
<p align="center">
  <img src="https://github.com/MarioPasc/N-Body-Simulation-Python/assets/120520768/350795ce-8a61-43f0-9127-297b7522c1ad" alt="Overall"/>
</p>


## Contributing
Contributions to this project are welcome. Here are some ways you can contribute:
- Reporting bugs
- Suggesting enhancements
- Adding new features
- Improving documentation

Please fork the repository and use a feature branch. Pull requests are warmly welcome.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Contact
For any queries, please reach out to my [LinkedIn](https://www.linkedin.com/in/mario-pascual-gonzalez). 

