FROM python:3.13-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV GRADIO_SERVER_PORT=7860
EXPOSE 7860
CMD ["cv-coach"]
