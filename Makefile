# Makefile for MyLLM v2 Development

# Colors for better output
GREEN  := $(shell tput -T screen setaf 2)
YELLOW := $(shell tput -T screen setaf 3)
RESET  := $(shell tput -T screen sgr0)

# Tools
UV ?= uv

## ---------------------------------------------------------------------------
## Project Setup & Dependency Management
## ---------------------------------------------------------------------------
.PHONY: install
install: ## Install production dependencies using uv
	@echo "$(GREEN)--> Installing production dependencies...$(RESET)"
	@$(UV) pip install -e .

.PHONY: install-dev
install-dev: ## Install all dependencies for development (prod + dev)
	@echo "$(GREEN)--> Installing all development dependencies...$(RESET)"
	@$(UV) pip install -r requirements-dev.txt -e .

.PHONY: update-deps
update-deps: ## Update all dependencies in pyproject.toml to their latest versions
	@echo "$(YELLOW)--> Updating all dependencies in pyproject.toml to latest versions...$(RESET)"
	@sed -n '/^dependencies = \[$/,/^\]$/p' pyproject.toml | grep -o '".*"' | xargs -n 1 $(UV) pip install -U
	@echo "$(GREEN)--> Dependencies updated. Please commit the changes to pyproject.toml.$(RESET)"


## ---------------------------------------------------------------------------
## Linting, Formatting & Static Analysis
## ---------------------------------------------------------------------------
.PHONY: lint
lint: ## Run linter (ruff)
	@echo "$(GREEN)--> Running linter (ruff)...$(RESET)"
	@$(UV) run ruff check myllm

.PHONY: fmt
fmt: ## Format code (ruff format)
	@echo "$(GREEN)--> Formatting code (ruff format)...$(RESET)"
	@$(UV) run ruff format myllm

## ---------------------------------------------------------------------------
## Testing
## ---------------------------------------------------------------------------
.PHONY: test
test: ## Run tests with pytest
	@echo "$(GREEN)--> Running tests with pytest...$(RESET)"
	@$(UV) run pytest -q

## ---------------------------------------------------------------------------
## CI
## ---------------------------------------------------------------------------
.PHONY: ci
ci: ## Run local CI using act
	@echo "$(YELLOW)--> Running local CI with act...$(RESET)"
	@./bin/act push -P ubuntu-latest=ghcr.io/catthehacker/ubuntu:act-22.04
	@echo "$(GREEN)--> Local CI run finished.$(RESET)"

## ---------------------------------------------------------------------------
## Publishing
## ---------------------------------------------------------------------------
.PHONY: publish-tools
publish-tools: ## Install tools required for publishing (build, twine)
	@echo "$(GREEN)--> Installing publishing tools...$(RESET)"
	@$(UV) pip install build twine

.PHONY: build
build: ## Build the package for distribution
	@echo "$(GREEN)--> Building source and wheel distributions...$(RESET)"
	@rm -rf dist/
	@python -m build

.PHONY: publish-test
publish-test: build ## Publish package to TestPyPI
	@echo "$(YELLOW)--> Publishing to TestPyPI...$(RESET)"
	@python -m twine upload --repository testpypi dist/*

.PHONY: publish
publish: build ## Publish package to PyPI
	@echo "$(YELLOW)--> PUBLISHING TO REAL PYPI! ARE YOU SURE? (Ctrl-C to cancel)$(RESET)"
	@read -p "Press Enter to continue..."
	@python -m twine upload dist/*

## ---------------------------------------------------------------------------
## Help - http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
## ---------------------------------------------------------------------------
.PHONY: help
help: ## Show this help message
	@printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*?##/ {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

## ---------------------------------------------------------------------------
## Docker Operations
## ---------------------------------------------------------------------------
IMAGE_NAME ?= myllm
IMAGE_TAG ?= latest

.PHONY: docker-build
docker-build: ## Build the Docker image for the application
	@echo "$(GREEN)--> Building Docker image $(IMAGE_NAME):$(IMAGE_TAG)...$(RESET)"
	@docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -f Dockerfile .

.PHONY: docker-run
docker-run: ## Run the Docker container with GPU support
	@echo "$(GREEN)--> Running Docker container $(IMAGE_NAME):$(IMAGE_TAG)...$(RESET)"
	@docker run -it --rm --gpus all \
		-v $(shell pwd)/experiments:/libllm/experiments \
		-v $(shell pwd)/configs:/libllm/configs \
		-v $(HOME)/.cache:/libllm/.cache \
		$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: docker-shell
docker-shell: ## Start an interactive shell inside the Docker container
	@echo "$(GREEN)--> Starting interactive shell in $(IMAGE_NAME):$(IMAGE_TAG)...$(RESET)"
	@docker run -it --rm --gpus all \
		-v $(shell pwd)/experiments:/libllm/experiments \
		-v $(shell pwd)/configs:/libllm/configs \
		-v $(HOME)/.cache:/libllm/.cache \
		$(IMAGE_NAME):$(IMAGE_TAG) /bin/bash