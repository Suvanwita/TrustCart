# TrustCart AI

TrustCart AI is a free, local-first Streamlit app for analyzing whether product reviews look trustworthy. It accepts pasted reviews, uploaded CSV files, or bundled sample data, then produces sentiment, emotion, fake/spam risk, product aspects, a weighted trust score, and a final buy/avoid report.

No paid APIs are required. Optional Hugging Face models run locally when available, and each model-backed feature has a deterministic fallback so the app can still run in constrained environments.

## Model Pipeline

1. **Review loading**
   - Paste text, upload CSV, or use `data/sample_reviews.csv`
   - Minimum review length filtering
   - Optional `rating` column support

2. **Sentiment and emotion**
   - Hugging Face `transformers` pipelines when available
   - Rule-based fallbacks if model loading or downloads fail

3. **Fake/spam review detection**
   - Linguistic signals: short reviews, repeated wording, uppercase ratio, exclamation marks, generic phrases, rating extremity
   - Sentence similarity with `sentence-transformers/all-MiniLM-L6-v2`
   - `IsolationForest` anomaly detection on engineered features
   - Safe fallback to non-embedding features if the sentence model cannot load

4. **Aspect mining**
   - KeyBERT keyword extraction when available
   - YAKE fallback
   - Keyword-frequency fallback
   - Groups mentions into product aspects such as battery, display, performance, delivery, quality, durability, comfort, support, and price

5. **Trust scoring**
   - Weighted score from sentiment, aspect quality, rating consistency, review volume, and authenticity
   - Labels: `Strong Buy`, `Buy with Caution`, `Mixed / Compare Alternatives`, `Avoid`
   - Warnings for high fake risk, duplicate reviews, negative aspect dominance, and too few reviews

6. **Final report**
   - Verdict
   - Pros and cons
   - Suspicious review patterns
   - Who should buy
   - Who should avoid
   - JSON and CSV export

## Tech Stack

- Streamlit
- pandas
- numpy
- scikit-learn
- transformers
- sentence-transformers
- KeyBERT
- YAKE
- Plotly
- joblib
- PyTorch CPU wheel

## How To Run

```bash
cd trustcart-ai
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

If you are running from the repository root:

```bash
cd /home/user/Desktop/TrustCart/trustcart-ai
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

## Screenshots

Add screenshots here after running the app locally:

- `screenshots/trust-score.png`
- `screenshots/fake-review-risk.png`
- `screenshots/final-report.png`

## Outputs

The app can export the final report from the UI:

- `trustcart_report.json`
- `trustcart_report.csv`

It also stores the latest analysis artifact in:

```text
outputs/latest_analysis.joblib
```

## Optional Local Environment

No environment variables are required. See `.env.example` for optional local model cache settings.

## Limitations

- The fake-review detector is heuristic and statistical, not proof of fraud.
- Hugging Face models may need an initial download and can be memory intensive.
- Aspect grouping is designed around common consumer-product categories and may miss niche domain terms.
- Sentiment and emotion models can misread sarcasm, mixed reviews, or multilingual text.
- Trust score weights are transparent defaults, not a certified risk model.

## Future Scope

- Add multilingual review support
- Add per-category aspect dictionaries
- Add trend analysis across review dates
- Add reviewer metadata support when available
- Add user-adjustable trust-score weights
- Add automated screenshots and a lightweight test suite
