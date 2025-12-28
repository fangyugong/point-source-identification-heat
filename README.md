# Identification of a Point Source in the Heat Equation from Sparse Boundary Measurements

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the reproducible MATLAB code for the numerical experiments presented in the paper:

**"Identification of a Point Source in the Heat Equation from Sparse Boundary Measurements"**


## Prerequisites

- **MATLAB** (Tested on version R2025b)
- **MATLAB PDE Toolbox**

## File Structure

The project is organized as follows:

```text
.
├── functions/               # Helper functions (FEM solvers, geometry, math tools)
├── main_ex*.m               # Main scripts for each example in the paper
├── reproduce_all.m          # Master script to reproduce all tables
└── README.md                # Project documentation
```

## Setup

1. Clone or download this repository.
2. Open MATLAB and navigate to the repository folder.
3. No need to manually add paths; the scripts automatically handle dependency loading.

## Usage

### 1. Reproduce All Results
To sequentially run all experiments and generate the summary tables (Tables 1-4) as they appear in the paper, simply run:

```matlab
reproduce_all
```

### 2. Run Individual Examples
Each script corresponds to a specific example in the paper.

> **Note:** Please refer to the header comments in each `.m` file for the detailed **mathematical formulation** (PDE, domain, source), **algorithm descriptions** (forward/inverse solvers), and **parameter definitions**.

| Script | Corresponds to |
| :--- | :--- |
| `main_ex01_disk_const.m` | **Example 4.1** (Unit Disk, Constant Amplitude) |
| `main_ex02a_disk_known_g.m` | **Example 4.2.(a)** (Unit Disk, Known *g(t)*) |
| `main_ex02b_ellipse_known_g.m` | **Example 4.2.(b)** (Ellipse, Known *g(t)*) |
| `main_ex03_disk_pwc.m` | **Example 4.3** (Unit Disk, Piecewise Constant *g(t)*) |
| `main_ex04_ellipse_unknown.m` | **Example 4.4** (Ellipse, Unknown *g(t)*) |

You can also run a specific example with a custom noise level:

```matlab
% Run Example 1 with 2% noise
main_ex01_disk_const(0.02);
```
**Tip**: Certain scripts accept additional optional arguments (e.g., custom source amplitude or geometry parameters). Please refer to the documentation inside each `.m` file for details.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
