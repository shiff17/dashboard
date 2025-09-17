# 🛡 Vulnerability Data Dashboard

A Streamlit dashboard for:
- Uploading CSVs
- Cleaning nulls
- KMeans clustering
- Baseline regression
- RL-style cleaning to improve accuracy
- Before vs After visualizations

## 🚀 Run locally

```bash
pip install --upgrade pip
setuptools wheel
pip install -r requirements.txt
streamlit run app.py
cd~/projects/dashboard
sed -i 's/\xc2\xao/ /g' app.py
get diff app.py
git add app.py
git commit -m "fix non-breaking spaces in app.py"
git push origin main
