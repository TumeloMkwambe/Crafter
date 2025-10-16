ENV_NAME = crafter_env
PYTHON_VERSION = 3.12

.PHONY: crafter_env
crafter_env:
	conda create -y -n $(ENV_NAME) python=$(PYTHON_VERSION)

.PHONY: install
install: crafter_env
	conda run -n $(ENV_NAME) pip install --upgrade pip
	conda run -n $(ENV_NAME) pip install -r requirements.txt

.PHONY: clean
clean:
	conda remove -y -n $(ENV_NAME) --all