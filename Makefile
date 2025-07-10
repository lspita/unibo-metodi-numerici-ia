PIP_REQUIREMENTS_FILE:=requirements.txt
PYTHON_VENV_DIR:=.venv

ifeq ($(OS),Windows_NT)
	PYTHON_VENV_ACTIVATE:=$(PYTHON_VENV_DIR)\Scripts\activate
else
	SHELL:=/bin/bash
	PYTHON_VENV_ACTIVATE:=source $(PYTHON_VENV_DIR)/bin/activate
endif

all: run

freeze: $(PIP_REQUIREMENTS_FILE)
	pip freeze --all > $(PIP_REQUIREMENTS_FILE)

install: $(PIP_REQUIREMENTS_FILE)
	pip install -r $(PIP_REQUIREMENTS_FILE)

.ONESHELL:
init:
	python3 -m venv $(PYTHON_VENV_DIR)
	$(PYTHON_VENV_ACTIVATE)
	make install

.ONESHELL:
run:
	$(PYTHON_VENV_ACTIVATE)
	jupyter lab
