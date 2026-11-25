PYTHON ?= python3.11
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

export OMP_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export NUMEXPR_NUM_THREADS := 1
export MPLCONFIGDIR := $(CURDIR)/.mplconfig
export XDG_CACHE_HOME := $(CURDIR)/.cache
export JOBLIB_TEMP_FOLDER := $(CURDIR)/.joblib

.PHONY: install data all fast test clean

$(VENV)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $(VENV)/.installed

install: $(VENV)/.installed

data: $(VENV)/.installed
	$(PY) -m src.run_all --stage data

all: $(VENV)/.installed
	$(PY) -m src.run_all --stage all

fast: $(VENV)/.installed
	$(PY) -m src.run_all --stage all --fast

test: $(VENV)/.installed
	$(PY) -m pytest -q

clean:
	rm -rf results/tables results/figures results/models results/logs \
		data/processed docs/model_card.md docs/governance_mapping.md \
		.pytest_cache .mplconfig .cache .joblib
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
