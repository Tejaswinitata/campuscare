# CampusCare – AI Campus Complaint Management (Flask + SQLite + scikit-learn)
## Run
    pip install -r requirements.txt
    python train_model.py   # optional: creates dataset.csv + models.joblib (auto-runs on first launch)
    python app.py           # open http://127.0.0.1:5000
## Structure
- train_model.py – synthetic dataset + TF-IDF/Logistic Regression models (category, priority)
- app.py – Flask REST API, SQLite storage, duplicate detection (cosine similarity), rule overrides
- templates/index.html – pastel UI with prebuilt complaint buttons
## API
POST /api/complaints · GET /api/complaints?category=&priority=&status=&roll= · POST /api/complaints/<id>/status · POST /api/complaints/<id>/escalate · DELETE /api/complaints/<id> · GET /api/stats
## Improve
Replace dataset.csv with real complaints, try DistilBERT, add login/roles, email alerts.
