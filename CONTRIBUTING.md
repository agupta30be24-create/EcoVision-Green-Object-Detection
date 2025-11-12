# Contributing to EcoVision

Thank you for your interest in contributing to EcoVision! This guide will help you get started with the project and understand our contribution process.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## Ways to Contribute

### 1. Report Bugs
- Check if the bug has already been reported in Issues
- Provide a clear description and steps to reproduce
- Include environment details (Python version, dependencies)
- Share error messages and logs

### 2. Suggest Features
- Describe the feature and its benefits
- Explain why it would be valuable to the project
- Consider how it aligns with green AI principles

### 3. Improve Documentation
- Fix typos and clarify explanations
- Add examples and use cases
- Update deployment guides

### 4. Submit Code
- Add new features or optimization techniques
- Improve model performance
- Optimize for energy efficiency
- Support new edge devices

## Getting Started

### Prerequisites
```bash
Python 3.8+
Git
Virtual environment tool (venv or conda)
```

### Setup Development Environment

1. **Fork the repository**
   ```bash
   Click "Fork" button on GitHub
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR-USERNAME/EcoVision-Green-Object-Detection.git
   cd EcoVision-Green-Object-Detection
   ```

3. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8 isort  # Development tools
   ```

5. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Code Style

- **Python**: Follow PEP 8 guidelines
  ```bash
  black --line-length 100 src/
  flake8 src/
  isort src/
  ```

- **Naming**: Clear, descriptive names
  - Functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`

### Documentation

- Add docstrings to all functions and classes
  ```python
  def quantize_model(model, calibration_data, output_path):
      """
      Apply INT8 quantization to the model.
      
      Args:
          model: PyTorch model to quantize
          calibration_data: Data for calibration
          output_path: Path to save quantized model
          
      Returns:
          Path to quantized model
          
      Example:
          >>> quantized_path = quantize_model(model, data, 'output/')
      """
  ```

### Testing

- Write tests for new features
- Ensure existing tests pass
  ```bash
  pytest tests/
  ```
- Aim for >80% code coverage

### Green AI Considerations

When contributing:
- Prioritize energy efficiency
- Measure and report carbon footprint
- Use CodeCarbon for tracking
- Consider edge device compatibility
- Document performance metrics

## Commit Guidelines

### Message Format
```
<type>: <subject>

<body>

<footer>
```

### Type
- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `opt`: Optimization (energy, model size)
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring

### Example
```
opt: Reduce model size through structured pruning

Implement LAMP algorithm to prune 50% of parameters.
Maintains 91.5% mAP while reducing model from 44MB to 22MB.
Energy consumption reduced by 35%.

Closes #123
```

## Pull Request Process

1. **Create a Pull Request**
   - Clear title describing changes
   - Reference related issues (#123)
   - Describe what changed and why

2. **Fill the PR Template**
   - Type of change (feature/fix/optimization)
   - Testing done
   - Performance metrics (if applicable)
   - Carbon footprint change (if applicable)

3. **Code Review**
   - Address reviewer comments
   - Keep commits clean and logical
   - Update based on feedback

4. **Merging**
   - Squash commits for clarity
   - Ensure CI/CD passes
   - Final approval from maintainer

## Areas for Contribution

### High Priority
- [ ] Additional datasets (autonomous driving, industrial)
- [ ] More edge device support (Qualcomm Snapdragon, MediaTek)
- [ ] Enhanced pruning algorithms
- [ ] Federated learning implementation

### Medium Priority
- [ ] Better visualization tools
- [ ] Performance benchmarking suite
- [ ] Multi-model ensemble approaches
- [ ] Hardware acceleration options

### Community Contributions Welcome
- [ ] Language translations for documentation
- [ ] Tutorial videos
- [ ] Blog posts about green AI
- [ ] Use case examples

## Performance Targets

When submitting optimization improvements, target:
- **Accuracy**: >90% mAP on BDD100K
- **Latency**: <50ms per frame on edge devices
- **Power**: <15W on Jetson Nano
- **Model Size**: <20MB
- **Carbon/Inference**: <100g CO2/hour

## Questions?

- Check existing issues for answers
- Create a discussion for questions
- Email: [project-email]

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to sustainable AI! 🌱**
