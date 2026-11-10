ENV_NAME := .venv-kg
PYTHON := python3

# Create virtual environment
.PHONY: venv
venv:
	$(PYTHON) -m venv $(ENV_NAME)
	. $(ENV_NAME)/bin/activate && pip install --upgrade pip

# Install dependencies directly from requirements.txt
.PHONY: install
install:
	. $(ENV_NAME)/bin/activate && pip install -r requirements.txt

# Full setup: venv + install
.PHONY: setup
setup: venv install

# Regenerate requirements.txt based on current venv
.PHONY: freeze
freeze:
	@pip install pip-tools
	@pip-compile requirements.in

# Show activation command
.PHONY: activate
activate:
	@echo "Run this to activate your environment:"
	@echo "source $(ENV_NAME)/bin/activate"
