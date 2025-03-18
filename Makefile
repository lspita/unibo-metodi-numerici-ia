PIP_REQUIREMENTS_FILE:=requirements.txt
PYTHON_VENV:=.venv/bin/activate

all: dev

freeze: $(PIP_REQUIREMENTS_FILE)
	pip freeze --all > $(PIP_REQUIREMENTS_FILE)

install: $(PIP_REQUIREMENTS_FILE)
	pip install -r $(PIP_REQUIREMENTS_FILE)

.ONESHELL:
dev:
	. $(PYTHON_VENV)
	jupyter lab