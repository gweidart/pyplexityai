# Contributing to PyPlexityAI

First off, thanks for taking the time to contribute! 🎉

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [gweidart@gmail.com].

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include error messages and stack traces
* Include your environment details (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you expected to see instead
* Explain why this enhancement would be useful

### Pull Requests

* Fill in the required template
* Follow the Python style guides
* Include tests for new features
* Update documentation for changes
* End all files with a newline
* Follow semantic commit messages

## Development Process

1. Fork the repo
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the test suite
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Setup Development Environment

```bash
# Clone your fork
git clone https://github.com/gweidart/pyplexityai.git
cd pyplexityai

# Create virtual environment
uv venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -e ".[dev]"
```

### Code Style

We use several tools to maintain code quality:

* **uv** for python version management, dependency, and virtual environment management
* **ruff** for linting and formatting
* **pyright** for static type checking

```bash
# Run all checks
ruff check . --fix --unsafe-fixes
ruff format .
pyright .
```

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

* `feat:` new feature
* `fix:` bug fix
* `docs:` documentation only changes
* `style:` formatting, missing semi colons, etc
* `refactor:` code change that neither fixes a bug nor adds a feature
* `perf:` code change that improves performance
* `test:` adding missing tests
* `chore:` updating grunt tasks etc

### Testing

```bash
# Run all tests
python test.py

# Run specific test file
python test.py test_errors.py

# Run with coverage
coverage run -m pytest
coverage report
```

## Project Structure

```
pyplexityai/
├── pyplexityai/
│   ├── __init__.py
│   ├── client.py          # Base sync client
│   ├── async_client.py    # Async client implementation
│   ├── client_types.py    # Type definitions
│   ├── errors.py          # Custom exceptions
│   └── openai.py          # OpenAI compatibility layer
├── tests/
│   ├── __init__.py
│   ├── test_client.py
│   ├── test_async.py
│   └── test_errors.py
├── .github/
│   ├── workflows/         # CI/CD configurations
│   ├── pull_request_template.md # Pull request template
│   └── ISSUE_TEMPLATE/    # Issue templates
│   │   ├── bug_report.md
│   │   └── feature_request.md
├── docs/                  # Documentation
│   └── style_guide.md     # Style guide
├── examples/              # Usage examples
├── LICENSE
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── pyproject.toml
├── ruff.toml
├── setup.py
├── py.typed
├── uv.lock
├── .editorconfig
├── .bumpversion.cfg
└── .gitignore
```

## Documentation

* Keep docstrings up to date
* Follow Google style docstrings
* Include type hints
* Update README.md for public API changes
* Add examples for new features

## Release Process

1. Update CHANGELOG.md
2. GitHub Actions will take care of versioning, tagging, and releasing

## Questions?

Feel free to open an issue for any questions or concerns. 