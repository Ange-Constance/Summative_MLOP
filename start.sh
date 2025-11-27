#!/bin/bash
# activate virtual environment if needed
# source venv/bin/activate

# run FastAPI with uvicorn
uvicorn src.api_upload:app --host 0.0.0.0 --port ${PORT:-8000}
