run-docker-compose:
	uv sync
	docker compose up --build

clean-notebook-outputs:
	jupyter nbconvert --clear-output --inplace notebooks/*/*.ipynb

run-api-local:
	uv run uvicorn apps.api.src.api.app:app --reload

run-chatbot-ui-local:
	PYTHONPATH=apps/chatbot_ui/src uv run streamlit run apps/chatbot_ui/src/chatbot_ui/app.py