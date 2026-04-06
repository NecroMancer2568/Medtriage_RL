FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 7860

# Just run the FastAPI server. It now carries the UI inside it!
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "7860"]