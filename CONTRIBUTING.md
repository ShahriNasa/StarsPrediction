# Contributing Guidelines

Thank you for your interest in contributing to this project.

Contributions are welcome and may include improvements to source code, model architectures, datasets, documentation, experiments, evaluation methods, or reproducibility.

Please read these guidelines before submitting changes.

---

## Ways to Contribute

You can contribute in several ways:

### Code Contributions
- Bug fixes
- Performance improvements
- Training or inference optimizations
- Additional preprocessing pipelines
- Visualization tools
- Model architecture enhancements

### Research Contributions
- New experiments
- Additional datasets
- Evaluation benchmarks
- Alternative methods
- Ablation studies
- Reproducibility validation

### Documentation Contributions
- Improve explanations
- Add examples
- Correct errors
- Expand setup instructions

---

## Before You Start

Before implementing a major change:

1. Check existing issues and pull requests
2. Open a new issue if necessary
3. Explain:
   - the problem
   - proposed solution
   - expected impact

This helps avoid duplicated work.

---

## Development Setup

Clone the repository:

```bash
git clone https://github.com/ShahriNasa/StarsPrediction.git
cd REPOSITORY
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate environment:

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Branch Naming

Please create descriptive branch names.

Examples:

```bash
feature/add-new-dataset
feature/improve-model-head
fix/training-bug
docs/update-readme
experiment/transformer-backbone
```

---

## Code Style Guidelines

Please follow these conventions:

### Python
- Follow PEP 8
- Use descriptive variable names
- Add comments where necessary
- Include docstrings for functions and classes

Example:

```python
def preprocess_lightcurve(data):
    """
    Normalize and preprocess input light curve data.

    Parameters
    ----------
    data : ndarray
        Input signal.

    Returns
    -------
    ndarray
        Processed output.
    """
```

---

## Commit Message Style

Use clear commit messages.

Examples:

```bash
feat: add uncertainty estimation module
fix: resolve training memory leak
docs: update installation instructions
refactor: simplify preprocessing pipeline
```

Avoid messages such as:

```bash
update
fix stuff
changes
```

---

## Pull Request Guidelines

Before submitting a pull request:

- [ ] Code runs successfully
- [ ] Documentation is updated if needed
- [ ] New functionality is explained
- [ ] Existing functionality is not broken
- [ ] Relevant experiments/results are included
- [ ] Pull request references related issue(s)

Provide a clear description:

```md
## Description
Short summary of changes

## Motivation
Why this change is needed

## Changes made
- Item 1
- Item 2

## Results
Include metrics, plots, or observations if applicable
```

---

## Reproducibility Requirements

For contributions involving experiments or models, please include:

- Dataset information
- Hyperparameters
- Training configuration
- Random seed (if applicable)
- Hardware specifications
- Evaluation metrics

Example:

```yaml
Batch size: 16
Learning rate: 5e-4
Epochs: 300
Image size: 640
GPU: NVIDIA RTX 4090
Seed: 42
```

---

## Dataset Contributions

If contributing datasets:

- Verify redistribution permissions
- Include source references
- Document preprocessing steps
- Describe annotation format

---

## Code of Conduct

Please maintain respectful and professional communication.

Constructive discussion and collaboration are encouraged.

---

## Questions

If you have questions, open an issue for discussion.

Thank you for contributing to this project.
