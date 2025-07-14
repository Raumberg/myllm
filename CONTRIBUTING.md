# Contributing to myllm

First of all: **thank you for even thinking about contributing!**
We want this project to be a playground for modern, clean, and robust LLM fine-tuning. If you’re here to make things better, you’re already awesome.

But let’s be real: we value quality, clarity, and a bit of fun. If you’re going to send a PR, please don’t make us cry. Read this guide, and everyone will be happier (especially you).

---

## Philosophy
- **No bs.** Code should be clean, readable, and actually work.
- **Modern ML only.** We love new tech, but only if it’s not cargo cult.
- **Reproducibility is king.** If your change breaks reproducibility, it’s a no-go.
- **Don’t be rude.** Be kind in issues, reviews, and discussions.

---

## How to Contribute

### 1. Fork & Branch
- Fork the repo and create a new branch for your feature or bugfix.
- Use clear branch names: `fix/fp8-logging`, `feat/quantization`, `docs/readme-update`.

### 2. Code Style
- **Python 3.12+** only.
- You can follow [PEP8](https://peps.python.org/pep-0008/), but since we are in ML, it's hard to be codestyle-god sometimes.
- You can use [ruff](https://github.com/astral-sh/ruff) for linting (see `requirements-dev.txt`).
- Docstrings for all public functions/classes. Comments are good, but don’t explain the obvious.
- Keep imports clean. Optional deps? Use the `myllm.utils` lazy importers.
- Don't be obsessed with `clean-code` bs. We need to understand things and be ready to change and research, not to follow abstract classes' childs.

### 3. Tests
- All new features and bugfixes **ideally** have tests. But.. we are in ML, remember that. Testing a GPU runs can be costly and time-consuming.
- Use `pytest`. Place tests in `tests/`.
- Run `pytest` before pushing. If it’s red, don’t even try a PR.

### 4. Commits & PRs
- Write clear, meaningful commit messages. No `fix stuff` or `update`.
- Squash trivial commits before PR (keep history clean).
- PRs should be focused: one feature/fix per PR. Don’t mix unrelated changes.
- Reference related issues in your PR description.
- Add a summary of what/why/how in the PR. If it’s a big change, explain your reasoning.

### 5. Local Development
- Clone, then run:
  ```sh
  uv pip install -e .
  make help
  ```
- Use `make` targets for common tasks (see `Makefile`).
- For dev dependencies: `uv pip install -r requirements-dev.txt`.
- Use a virtualenv or `uv venv`.

### 6. Issues & Feature Requests
- Search existing issues before opening a new one.
- Bug reports: include logs, stack traces, and steps to reproduce.
- Feature requests: explain the use case, not just the feature.
- Be respectful. We’re all here to build cool stuff.

### 7. What NOT to Do
- Don’t add heavy dependencies unless absolutely necessary.
- Don’t break existing APIs without discussion.
- Don’t submit code you haven’t run locally.
- Don’t write code you wouldn’t want to maintain yourself.

---

## FAQ / Common Pitfalls
- **CI fails on install?** Check your `pyproject.toml` and try `uv pip install -e .` locally.
- **Optional dependency import errors?** Use the lazy importers from `myllm.utils`.
- **Tests are slow?** Use smaller configs/datasets for testing.
- **Not sure if your idea fits?** Open an issue and ask!

---

## Contact & Feedback
- For questions, open an issue or start a discussion.
- For private stuff, email the maintainer (see `pyproject.toml`).
- If you’re angry, take a walk, then write a polite message.

---

## Final Words
We love bold ideas, but we love working code even more. If you make our lives easier, we’ll love you forever. If you break things, we’ll find you (just kidding… or not).

Happy hacking! 🚀 