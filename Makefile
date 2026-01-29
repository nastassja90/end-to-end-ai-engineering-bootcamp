default: help

.PHONY: help
help: # show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n\n"; done


run-docker-compose: # Run the docker compose environment
	uv sync --all-packages
	docker compose up --build

clean-notebook-outputs: # Clear outputs from all Jupyter notebooks
	jupyter nbconvert --clear-output --inplace notebooks/**/*.ipynb

run-api-local: # Run the API locally with auto-reload
	uv run uvicorn apps.api.src.api.app:app --reload

run-evals-retriever: # Run the evals retriever module
	uv sync --all-packages
	PYTHONPATH=${PWD}/apps/api:${PWD}/apps/api/src:$$PYTHONPATH:${PWD} uv run --env-file .env python -m evals.retriever

run-chatbot-ui-local: # Run the Chatbot UI locally with Streamlit
	PYTHONPATH=${PWD}/apps/chatbot_ui/src uv run streamlit run apps/chatbot_ui/src/chatbot_ui/app.py