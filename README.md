# AI-Powered Customer Sentiment Analyzer

Classifies customer reviews as Positive, Negative or Neutral with a confidence score.
Built with Python, Scikit-learn, NLTK and Streamlit.

## Setup
pip install -r requirements.txt

## Train
Download the IMDB 50K reviews CSV from Kaggle, then:
python train_model.py --data IMDB_Dataset.csv --text-col review --label-col sentiment

## Run
python -m streamlit run app.py
