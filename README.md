# Deep Learning Class (VITMMA19) Project Work Documentation


## Data Preparation

I provided the used data in the data folder. 

Due to hardware limitations, I used a subset of the available data to ensure the training process completes within a reasonable time (approx. 3 hours). The full dataset would have required significantly more computational resources.

The data preparation is handled automatically by the script data_processing_01.py.

Source: The script expects raw JSON files containing text annotations to be located in the data/all_data folder.

Processing:

It recursively searches for .json files.

Extracts the text and the corresponding label.

Filters out noise (e.g., texts shorter than 7 words).

Splits the data into Train/Validation/Test sets.

Output: The processed data is saved as data/df.csv, which is then used by the training and evaluation scripts.

## Project Details

### Project Information

- **Selected Topic**: Legal Text Decoder
- **Student Name**: Tolmácsi Eszter
- **Aiming for +1 Mark**: Yes

### Solution Description

The goal of this project is to assess the comprehensibility of Hungarian legal texts by classifying them into a 5-point scale (from "Hard to understand" to "Easy to understand").

**Model Architecture**: I utilized a pre-trained LegalBERT (SZTAKI-HLT/hubert-base-cc) model as the backbone. 
To adapt it for classification:

Backbone: The BERT layers are used to extract contextual embeddings. To prevent catastrophic forgetting and speed up training, the initial embedding layers and the first 6 encoder layers are frozen.

Classifier Head: I replaced the standard head with a custom sequential block consisting of:

Linear layer (Hidden size -> 256)

ReLU activation

Dropout (0.1) for regularization

Linear output layer (256 -> 5 classes)

**Input**: The model processes text tokenized with AutoTokenizer, truncated to a maximum length of 256 tokens.

**Training Methodology**: The model is fine-tuned using the AdamW optimizer with a linear learning rate scheduler (warmup included). To handle class imbalance in the dataset, I calculated and applied Class Weights to the CrossEntropyLoss. The training loop includes Early Stopping to prevent overfitting, saving the best model based on validation loss.

**Data Pipeline**: Raw JSON annotations are processed into a clean DataFrame. Texts shorter than 7 words are filtered out to ensure quality.

### Extra Credit Justification

**ML as a Service (GUI)**: I developed a fully functional Streamlit Web Application (streamlit_legal_text_app.py). This provides a user-friendly interface where users can input legal text and receive real-time predictions with confidence scores and interpreted labels.

**Advanced Evaluation**: Beyond basic accuracy, the pipeline generates and saves a Confusion Matrix heatmap to visualize misclassifications and provides a full classification report.

### Pre-trained Model & Quick Start
For convenience, a pre-trained model is included in the `output/` directory. This allows for immediate testing of the GUI without waiting for the full training pipeline to finish (it can take appr. 3 hours).

As a proof of my work, I saved the log file in the `logs/` folder and in the `output/models/bert_finetuned` you can find the confusion matrix. 

To launch the GUI immediately using the local environment (without Docker):


```bash
pip install -r requirements.txt
```

```bash
python -m streamlit run src/streamlit_legal_text_app.py
```

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### 1. Building the Docker Image

Open your terminal (PowerShell or Bash) in the project root directory and run the following command to build the Docker image:

```bash
docker build -t legal-text-app:1.0 .
```

#### 2. Running the Application

Once the image is built, you can run the container. This will start the data processing, model training, evaluation pipeline, and finally launch the Streamlit web application.


##### Windows (PowerShell)

Use this command if you are on Windows:

```bash
docker run --rm --gpus all `
  -v "${PWD}\data:/app/data" `
  -v "${PWD}\output:/app/output" `
  -v "${PWD}\logs:/app/logs" `
  -p 8051:8501 `
  legal-text-app:1.0
```


##### Linux / Mac (Bash)

Use this command if you are on Linux or macOS:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/logs:/app/logs" \
  -p 8051:8501 \
  legal-text-app:1.0
```

#### 3. Accessing the GUI

After the training pipeline finishes, the Streamlit application will start automatically.

Open your web browser and navigate to:

http://127.0.0.1:8051/

Here you can enter legal texts and get real-time complexity predictions from the model.

#### 4. Stopping the docker container
As the docker container starts the GUI, when you exit from the pipeline, the container can get stuck, therefore don't forget to stop the container after usage!

```bash
docker ps
```

```bash
docker stop <container_id>
```


### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `data_processing_01.py`: Scripts for loading, cleaning, and preprocessing the raw data.
    - `train_02.py`: The main script for defining the model and executing the training loop.
    - `evaluation_03.py`: Evaluates the saved model on the test set and generates a confusion matrix.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs) and paths.
    - `streamlit_legal_text_app.py`: The web application for user interaction and inference.

    - **`modules/`**:
            - `LegalBERT.py`: Definition of the custom LegalBERT class with the specific classifier head and layer freezing logic.

    - **`utils/`**:
        - `logger.py`: Custom logging configuration that handles multiprocessing and file output safely.

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `data_exploration.ipynb`: Notebook for initial exploratory data analysis and visualization.
    - `incremental_development.ipynb`: Notebook for incremental development process.

- **`logs/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
    - `run.sh`: Shell script that orchestrates the execution of the full pipeline (processing -> training -> evaluation -> app).
