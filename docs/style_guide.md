# Style Guide

This project follows PEP 8 primarily, with some additional guidelines:

1. **Line Length**: Aim for 88 characters per line.  
2. **Docstrings**: Use Google-style docstrings with type hints.  
3. **Imports**: 
   - Standard library imports first
   - Third-party imports next
   - Project-specific imports last
4. **Naming**:
   - Functions and variables: `snake_case`
   - Classes and exceptions: `PascalCase`
   - Constants: `UPPER_CASE`
5. **Typing**: 
   - Add type hints wherever possible.
   - Use `typing` or built-ins like `dict`, `list`, etc.
6. **Linting & Formatting**: 
   - Use `ruff check .` and `ruff format .` to automatically fix lint issues and format code.

Always ensure code remains clear, concise, and well-documented. 