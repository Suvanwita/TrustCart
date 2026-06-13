# TrustCart AI

TrustCart AI is a fully textual Streamlit app for analyzing product review trust signals. It accepts pasted reviews, uploaded CSV files, or bundled sample data, then reports sentiment distribution, fake review risk, common product aspects, trust score, pros and cons, and a final buy/avoid suggestion.

## Run

```bash
cd trustcart-ai
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## CSV Format

Uploaded CSV files must include a `review` column. A `rating` column is optional.

```csv
review,rating
"Great battery life and solid build.",5
"Stopped working after one week.",1
```

## Notes

- No paid APIs are used.
- Transformer and KeyBERT models are loaded locally through open-source packages when available.
- The app includes deterministic fallback logic for sentiment, emotion, aspects, and summarization so it remains usable in lightweight environments.
