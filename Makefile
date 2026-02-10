.PHONY: venv install run run-batch clean help

venv:
	python -m venv .venv

install:
	pip install -r requirements.txt

run:
	uvicorn src.app:app --reload

run-batch:
	python -m src.transcribe_batch --input ./audios --out ./output --model small

clean:
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

help:
	@echo "Targets:"
	@echo "  make venv       - create virtualenv .venv"
	@echo "  make install    - install dependencies"
	@echo "  make run        - start API (uvicorn)"
	@echo "  make run-batch  - run batch transcription (audios -> output)"
	@echo "  make clean      - remove caches"
	@echo "  make help       - this help"
