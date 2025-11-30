#!/bin/bash 
set -e 

echo "Running data processing..." 
python data_processing_01.py 


echo "Running model training..." 
python train_02.py 


echo "Running evaluation..." 
python evaluation_03.py 


echo "Pipeline finished successfully." 

echo "Opening app in browser..."
python -m streamlit run streamlit_legal_text_app.py --server.address=0.0.0.0 --server.port=8501
