# Legal Text Decoder

This project is a machine learning application designed to classify the complexity of legal texts. It utilizes a fine-tuned LegalBERT model to predict how understandable a given legal text is on a scale from 1 to 5. The application provides a user-friendly GUI built with Streamlit.


## Prerequisites

Before running the application, ensure you have the following installed:



* Docker Desktop (or Docker Engine on Linux)
* NVIDIA Container Toolkit (if you plan to use GPU acceleration with --gpus all)


## Project Structure

Ensure your project directory is organized as follows before running the container. The Docker container expects to mount data and output directories from your host machine.

```text
project-root/ \
├── src/                # Source code (python scripts, Dockerfile)
├── data/               # Input data (JSON/CSV files)
├── output/             # Directory where trained models and logs will be saved
└── README.md \
```


## 1. Building the Docker Image

Open your terminal (PowerShell or Bash) in the project root directory and run the following command to build the Docker image:

```bash
docker build -t legal-text-app:1.0 .
```


## 2. Running the Application

Once the image is built, you can run the container. This will start the data processing, model training, evaluation pipeline, and finally launch the Streamlit web application.


### Windows (PowerShell)

Use this command if you are on Windows:

```bash
docker run --rm --gpus all `
  -v "${PWD}\data:/app/data" `
  -v "${PWD}\output:/app/output" `
  -p 8051:8501 `
  legal-text-app:1.0 > training_log.txt 2>&1
```


### Linux / Mac (Bash)

Use this command if you are on Linux or macOS:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  -p 8051:8501 \
  legal-text-app:1.0 > training_log.txt 2>&1
```


**Explanation of flags:**



* --rm: Automatically removes the container when it exits.
* --gpus all: Enables GPU support inside the container (requires NVIDIA drivers).
* -v ...: Mounts your local data and output folders to the container so files persist.
* -p 8051:8501: Maps the container's Streamlit port (8501) to port 8051 on your machine.
* > training_log.txt 2>&1: Redirects all logs (training progress, errors) to a text file instead of the console.


## 3. Accessing the GUI

After the training pipeline finishes (you can check progress in training_log.txt), the Streamlit application will start automatically.

Open your web browser and navigate to:

 http://127.0.0.1:8051/

Here you can enter legal texts and get real-time complexity predictions from the model.