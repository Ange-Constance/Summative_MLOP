FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir tensorflow streamlit fastapi uvicorn pandas
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]