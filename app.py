import os, re, sqlite3, json
from datetime import datetime
from functools import wraps
import joblib
from flask import Flask, jsonify, request, render_template, g, session
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-me")
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  SESSION_COOKIE_SECURE=bool(os.environ.get("PRODUCTION")))
DB = os.environ.get("DB_PATH", "complaints.db")
STATUS = ["Submitted", "Categorized", "Assigned", "In Progress", "Resolved", "Closed"]
DEPT = {"Infrastructure": ("Maintenance Office", 48), "Hostel & Mess": ("Hostel Warden", 24), "Academic": ("Head of Department", 72),
        "IT & Wi-Fi": ("IT Cell", 24), "Transport": ("Transport In-charge", 48), "Safety & Harassment": ("Anti-Ragging / Disciplinary Cell", 4),
        "Administrative": ("Administrative Office", 72)}
URGENT = re.compile(r"fire|spark|shock|emergency|danger|threat|ragging|harass|unsafe|stolen|assault|injur|accident|immediately|urgent|bleed", re.I)
ORDER = ["Low", "Medium", "High"]

if not os.path.exists("models.joblib"):
    import train_model; train_model.train()
MODELS = joblib.load("models.joblib")

def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB); g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close(_):
    d = g.pop("db", None)
    if d: d.close()

def init():
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, name TEXT, roll TEXT,
          password_hash TEXT, role TEXT DEFAULT 'student')""")
        c.execute("""CREATE TABLE IF NOT EXISTS complaints(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, anon INTEGER DEFAULT 0,
          name TEXT, roll TEXT, text TEXT, category TEXT, confidence REAL, priority TEXT, score REAL, why TEXT, dept TEXT, sla INTEGER,
          status INTEGER DEFAULT 1, dup_of TEXT, created TEXT)""")
        if not c.execute("SELECT 1 FROM users WHERE role='admin'").fetchone():
            c.execute("INSERT INTO users(username,name,roll,password_hash,role) VALUES(?,?,?,?,'admin')",
                      (os.environ.get("ADMIN_USER", "admin").upper(), "Administrator", "-",
                       generate_password_hash(os.environ.get("ADMIN_PASSWORD", "admin123"))))
init()

def login_required(role=None):
    def deco(f):
        @wraps(f)
        def w(*a, **k):
            if "uid" not in session: return jsonify(error="Please log in."), 401
            if role and session.get("role") != role: return jsonify(error="Admins only."), 403
            return f(*a, **k)
        return w
    return deco

# ---------- auth ----------
@app.post("/api/register")
def register():
    j = request.get_json(force=True); roll = (j.get("roll") or "").strip().upper(); name = (j.get("name") or "").strip()
    if len(roll) < 3 or not name or len(j.get("password") or "") < 6:
        return jsonify(error="Enter name, roll number (3+ chars) and a password of 6+ characters."), 400
    try:
        db().execute("INSERT INTO users(username,name,roll,password_hash) VALUES(?,?,?,?)", (roll, name, roll, generate_password_hash(j["password"])))
        db().commit()
    except sqlite3.IntegrityError:
        return jsonify(error="This roll number is already registered."), 409
    return jsonify(ok=True), 201

@app.post("/api/login")
def login():
    j = request.get_json(force=True)
    u = db().execute("SELECT * FROM users WHERE username=?", ((j.get("username") or "").strip().upper(),)).fetchone()
    if not u or not check_password_hash(u["password_hash"], j.get("password") or ""):
        return jsonify(error="Invalid username or password."), 401
    session.clear(); session.update(uid=u["id"], role=u["role"], name=u["name"], roll=u["roll"])
    return jsonify(name=u["name"], role=u["role"], roll=u["roll"])

@app.post("/api/logout")
def logout(): session.clear(); return jsonify(ok=True)

@app.get("/api/me")
def me():
    if "uid" not in session: return jsonify(error="not logged in"), 401
    return jsonify(name=session["name"], role=session["role"], roll=session["roll"])

# ---------- AI ----------
def analyze(text):
    cm, pm = MODELS["category"], MODELS["priority"]
    cp = cm.predict_proba([text])[0]; cat = cm.classes_[cp.argmax()]; conf = float(cp.max())
    pp = dict(zip(pm.classes_, pm.predict_proba([text])[0])); pri = max(pp, key=pp.get)
    why = [f"ML priority model: {pri} ({pp[pri]*100:.0f}%)"]
    if cat == "Safety & Harassment" and pri != "High":
        pri = "High"; why.append("safety category always escalated")
    if URGENT.search(text) and pri != "High":
        pri = ORDER[min(2, ORDER.index(pri) + 1)]; why.append("urgent keyword detected, priority raised")
    score = round(100 * (pp.get("High", 0) + .5 * pp.get("Medium", 0)))
    if pri == "High": score = max(score, 70)
    return cat, conf, pri, score, why

def find_dup(text):
    vec = MODELS["category"].steps[0][1]
    rows = db().execute("SELECT id,text FROM complaints WHERE status<4").fetchall()
    if not rows: return None
    sims = cosine_similarity(vec.transform([text]), vec.transform([r["text"] for r in rows]))[0]
    i = sims.argmax()
    return f"CMP{rows[i]['id']}" if sims[i] > .6 else None

def row(r):
    d = dict(r); d["id"] = f"CMP{d['id']}"; d["why"] = json.loads(d["why"]); d["status_name"] = STATUS[d["status"]]
    if d["anon"] and session.get("role") == "admin": d["name"], d["roll"] = "Anonymous", "-"
    age = (datetime.now() - datetime.fromisoformat(d["created"])).total_seconds() / 3600
    d["breached"] = d["status"] < 4 and age > d["sla"]; d.pop("user_id", None)
    return d

# ---------- complaints ----------
@app.get("/")
def index(): return render_template("index.html")

@app.post("/api/complaints")
@login_required("student")
def create():
    text = (request.get_json(force=True).get("text") or "").strip()
    if len(text) < 15: return jsonify(error="Please describe the problem in at least 15 characters."), 400
    anon = 1 if request.get_json(force=True).get("anonymous") else 0
    cat, conf, pri, score, why = analyze(text); dept, sla = DEPT[cat]
    cur = db().execute("INSERT INTO complaints(user_id,anon,name,roll,text,category,confidence,priority,score,why,dept,sla,dup_of,created) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (session["uid"], anon, session["name"], session["roll"], text, cat, conf, pri, score, json.dumps(why), dept, sla, find_dup(text), datetime.now().isoformat()))
    db().commit()
    return jsonify(row(db().execute("SELECT * FROM complaints WHERE id=?", (cur.lastrowid,)).fetchone())), 201

@app.get("/api/complaints")
@login_required()
def lst():
    q, args = "SELECT * FROM complaints WHERE 1=1", []
    if session["role"] == "student": q += " AND user_id=?"; args.append(session["uid"])
    for k in ("category", "priority"):
        if request.args.get(k): q += f" AND {k}=?"; args.append(request.args[k])
    if request.args.get("status") not in (None, ""): q += " AND status=?"; args.append(int(request.args["status"]))
    s = request.args.get("roll", "")
    if s:
        q += " AND (roll LIKE ? OR id=?)"; args += [f"%{s.upper()}%", int(re.sub(r"\D", "", s) or 0)]
    q += " ORDER BY CASE priority WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END, score DESC, id DESC"
    return jsonify([row(r) for r in db().execute(q, args)])

@app.post("/api/complaints/<int:cid>/status")
@login_required("admin")
def status(cid):
    r = db().execute("SELECT status FROM complaints WHERE id=?", (cid,)).fetchone()
    if r: db().execute("UPDATE complaints SET status=? WHERE id=?", (min(5, r["status"] + 1), cid)); db().commit()
    return jsonify(ok=True)

@app.post("/api/complaints/<int:cid>/escalate")
@login_required("admin")
def escalate(cid):
    db().execute("UPDATE complaints SET priority='High', score=MAX(score,85) WHERE id=?", (cid,)); db().commit(); return jsonify(ok=True)

@app.delete("/api/complaints/<int:cid>")
@login_required("admin")
def delete(cid):
    db().execute("DELETE FROM complaints WHERE id=?", (cid,)); db().commit(); return jsonify(ok=True)

@app.post("/api/complaints/<int:cid>/feedback")
@login_required("student")
def feedback(cid):
    new = 5 if request.get_json(force=True).get("action") == "confirm" else 3
    db().execute("UPDATE complaints SET status=? WHERE id=? AND user_id=? AND status=4", (new, cid, session["uid"])); db().commit()
    return jsonify(ok=True)

@app.get("/api/stats")
@login_required("admin")
def stats():
    r = db().execute("SELECT COUNT(*) t, SUM(status<4) o, SUM(status>=4) r, SUM(priority='High' AND status<4) h FROM complaints").fetchone()
    cats = {x["category"]: x["n"] for x in db().execute("SELECT category, COUNT(*) n FROM complaints GROUP BY category")}
    return jsonify(total=r["t"], open=r["o"] or 0, resolved=r["r"] or 0, high=r["h"] or 0, categories=cats)

if __name__ == "__main__":
    app.run(debug=True)
