ENV_NAME = crafter_env
YAML_FILE = environment.yml

.PHONY: crafter_env
crafter_env:
	conda env create -f $(YAML_FILE) || conda env update -f $(YAML_FILE) --prune

.PHONY: install
install: crafter_env
	conda run -n $(ENV_NAME) pip install -r requirements.txt

.PHONY: clean
clean:
	conda remove -y -n $(ENV_NAME) --all
