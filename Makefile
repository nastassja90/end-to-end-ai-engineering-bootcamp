run-docker-compose:
	uv sync
	docker compose up --build

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/**/*.ipynb

run-api-local:
	uv run uvicorn apps.api.src.api.app:app --reload

run-evals-retriever:
	uv sync
	PYTHONPATH=${PWD}/apps/api:${PWD}/apps/api/src:$$PYTHONPATH:${PWD} uv run --env-file .env python -m evals.retriever

run-chatbot-ui-local:
	PYTHONPATH=${PWD}/apps/chatbot_ui/src uv run streamlit run apps/chatbot_ui/src/chatbot_ui/app.py