"""Builds a synthetic labeled dataset and trains two models:
   1) category classifier  2) priority classifier  (TF-IDF + Logistic Regression)."""
import random, csv, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA = {
 "Infrastructure": [("the ceiling fan in the classroom is not working","Low"),("projector in the seminar hall is broken","Medium"),
  ("electric socket in the classroom is sparking and giving shocks","High"),("roof is leaking and water is falling on the benches","Medium"),
  ("benches and door of the classroom are damaged","Low"),("exposed wires near the lift are dangerous","High"),("washroom tap is leaking and toilets are dirty","Low")],
 "Hostel & Mess": [("cockroaches found in the food served at the mess","High"),("mess food quality is poor and the dining area is dirty","Medium"),
  ("no water in the hostel block since morning","High"),("hostel room mattress and bed are in bad condition","Low"),("geyser in the hostel is not working","Low"),
  ("curd and breakfast timings in the canteen are irregular","Low"),("hostel warden is not responding to room repair requests","Medium")],
 "Academic": [("clash between two lectures in the timetable","Medium"),("internal marks are not updated in the portal","Medium"),
  ("faculty is not completing the syllabus before the exam","Medium"),("exam schedule has not been announced yet","Low"),
  ("revaluation result is delayed for many weeks","Medium"),("attendance was marked wrongly for the lab class","Low"),("need an extra class for the course lab-manual","Low")],
 "IT & Wi-Fi": [("wifi in the computer lab is slow","Low"),("internet is not working and we cannot submit assignments on the portal","Medium"),
  ("college website login page is not opening","Medium"),("lab computers are not starting and the software is missing","Medium"),
  ("network is down in the whole block during online exam","High"),("password reset for student portal is not working","Low"),("server is slow during registration","Medium")],
 "Transport": [("college bus is running late every day","Medium"),("bus driver is driving rashly and it is unsafe","High"),
  ("need a new bus route and pickup stop","Low"),("bus seats are broken and overcrowded","Low"),("bus did not come today and students missed the exam","High"),("bus timing needs to change","Low"),("bus breakdown on the way again","Medium")],
 "Safety & Harassment": [("seniors are ragging and threatening first year students","High"),("a student is being harassed and stalked near the hostel","High"),
  ("my laptop was stolen from the classroom","High"),("fight happened near the canteen and students are scared","High"),
  ("unsafe lighting near the girls hostel at night","Medium"),("bullying in the class group chat","High"),("unknown persons entering the campus at night","High")],
 "Administrative": [("fee receipt is not generated after payment","Medium"),("scholarship amount has not been credited","Medium"),
  ("bonafide certificate is pending for many days","Low"),("need a refund for the extra fee collected","Medium"),
  ("id card has not been issued yet","Low"),("office staff is not responding to admission queries","Low"),("wrong amount was printed on the challan","Low")],
}
PRE = ["", "", "Sir, ", "Please help, ", "I want to report that ", "Since many days "]
POST = ["", " Please fix it.", " Kindly take action.", " Many students are facing this."]
HI = [" Please act immediately.", " This is an emergency.", " It is urgent."]
LO = [" Not urgent.", " Please look into it when possible."]

def build(n=18):
    rows = []
    for cat, items in DATA.items():
        for text, pr in items:
            for _ in range(n):
                s = random.choice(PRE) + text + random.choice(POST)
                if pr == "High": s += random.choice(HI + [""])
                if pr == "Low":  s += random.choice(LO + ["", ""])
                rows.append((s.strip(), cat, pr))
    random.shuffle(rows)
    return rows

def train():
    random.seed(7)
    rows = build()
    with open("dataset.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["text", "category", "priority"]); w.writerows(rows)
    X = [r[0] for r in rows]
    models = {}
    for name, idx in (("category", 1), ("priority", 2)):
        y = [r[idx] for r in rows]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=1, stratify=y)
        m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=2000, C=8))
        m.fit(Xtr, ytr)
        print(f"== {name} model ==\n", classification_report(yte, m.predict(Xte), zero_division=0))
        m.fit(X, y)
        models[name] = m
    joblib.dump(models, "models.joblib")
    return models

if __name__ == "__main__":
    train()
