# G.O.D Project Guidelines for Claude

## Commands
- **Run tests**: `pytest validator/tests/` or single test `pytest validator/tests/test_run_evaluation.py::test`
- **Lint code**: `ruff check --fix .`
- **Setup dev environment**: `task setup`
- **Install dependencies**: `pip install -e .` or with dev deps `pip install -e ".[dev]"`
- **Run miner**: `task miner`
- **Run validator**: `task validator`

## Code Style
- **Formatting**: 130 character line length
- **Imports**: Sorted with standard library first, third-party next, then local
- **Types**: Always use type hints
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error handling**: Use try/except with specific exceptions, log errors with loguru

## Conventions
- **Logging**: Use `from validator.utils.logging import get_logger` and `logger = get_logger(__name__)`
- **Modules**: Organize code in logical modules with clear responsibility
- **Models**: Use dataclasses or Pydantic models for data validation and configuration
- **Tests**: Write pytest tests with appropriate fixtures and descriptive names
- **Documentation**: Document functions, classes, and modules with clear docstrings