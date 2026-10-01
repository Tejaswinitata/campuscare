# Deploy CampusCare on Render (free)
1. Push this folder to a GitHub repo.
2. Render -> New -> Web Service -> connect the repo.
3. Build command: pip install -r requirements.txt && python train_model.py
4. Start command: gunicorn app:app
5. Instance type: Free.
6. Environment variables: SECRET_KEY (long random string), ADMIN_USER, ADMIN_PASSWORD, PRODUCTION=1
7. Deploy, then open the onrender.com URL. Free instances sleep after 15 min idle and local SQLite data resets on restart/redeploy.
