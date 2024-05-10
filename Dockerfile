# Use a base image with Python installed
FROM python:3.12.1

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 7860 available to the world outside this container
EXPOSE 7860

# Update Environment Variables Below
ENV OPENAI_API_KEY= \
    OPENAI_MODEL=gpt-4-0125-preview \
    OPENAI_FUNCTIONS_MODEL=gpt-3.5-turbo-1106 \
    LANGCHAIN_TRACING_V2=true \
    LANGCHAIN_ENDPOINT= \
    LANGCHAIN_API_KEY= \
    LANGCHAIN_PROJECT=

# Run data ingestion
CMD ["python", "src/ingest.py"]

# Set the entrypoint command to run the Gradio app
CMD ["python", "src/app.py"]
