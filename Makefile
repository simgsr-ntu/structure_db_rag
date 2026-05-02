.PHONY: install scrape ingest run setup test clean

# Virtual environment settings
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
YEAR ?= $(shell date +%Y)

# Setup entire project (one-click install)
setup: install
	@echo "🧠 Running full initial ingestion via Dagster..."
	DAGSTER_HOME=$$(mktemp -d) $(VENV_DIR)/bin/dagster job execute -m dagster_pipeline -j full_ingestion_job
	@echo "✅ Setup complete! You can now run the app with 'make run'"

# Install dependencies and setup environment
install:
	@echo "📦 Setting up virtual environment..."
	python3 -m venv $(VENV_DIR)
	@echo "📦 Installing dependencies..."
	$(PIP) install -r requirements.txt
	@if [ ! -f .env ]; then \
		echo "📝 Creating .env from .env.example..."; \
		cp .env.example .env; \
	fi
	@echo "✅ Install complete."

# Scrape sermons via Dagster (defaults to current year, use YEAR=2024 for specific)
scrape:
	@echo "📥 Scraping sermons for year $(YEAR) via Dagster..."
	DAGSTER_HOME=$$(mktemp -d) $(VENV_DIR)/bin/dagster asset materialize --select sermon_scraping -m dagster_pipeline \
		--config-json '{"ops": {"sermon_scraping": {"config": {"year": $(YEAR)}}}}'

# Ingest sermons into SQLite and ChromaDB via Dagster
ingest:
	@echo "🧠 Ingesting sermons via Dagster..."
	DAGSTER_HOME=$$(mktemp -d) $(VENV_DIR)/bin/dagster asset materialize --select sermon_ingestion -m dagster_pipeline

# Run the Gradio Chat UI
run:
	@echo "🚀 Starting Gradio UI..."
	$(PYTHON) app.py

# Run Dagster for weekly scheduling
dagster:
	@echo "⏱️ Starting Dagster scheduler..."
	DAGSTER_HOME=$$(mktemp -d) $(VENV_DIR)/bin/dagster dev -m dagster_pipeline

# Run tests
test:
	@echo "🧪 Running tests..."
	$(PYTHON) -m pytest tests/ -v

# Clean up environment and data
clean:
	@echo "🧹 Cleaning up data and environment..."
	rm -rf $(VENV_DIR)
	rm -rf data/chroma_db
	rm -rf data/sermons.db
	rm -rf data/staging
	@echo "✅ Cleanup complete."
