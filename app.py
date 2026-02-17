"""
╔══════════════════════════════════════════════════════════════════════════╗
║          HomeValue AI — Interactive House Price Prediction App           ║
║          Enhanced app.py  |  Flask + Scikit-learn + Chart.js            ║
╚══════════════════════════════════════════════════════════════════════════╝

Run:
    pip install flask scikit-learn pandas numpy joblib
    python app.py

Then open:  http://127.0.0.1:5000
"""

import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

# ── Try loading the saved model & scaler ────────────────────────────────────
try:
    import joblib
    MODEL  = joblib.load("house_price_model.pkl")
    SCALER = joblib.load("scaler.pkl")
    MODEL_LOADED = True
    print("✅  Loaded house_price_model.pkl + scaler.pkl")
except Exception as e:
    MODEL_LOADED = False
    print(f"⚠️  Could not load saved model ({e}). Using built-in fallback model.")

    # ── Fallback: train a quick Random Forest on synthetic data ─────────────
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    rng = np.random.RandomState(42)
    N   = 2000

    sqft        = rng.randint(500,   8000, N).astype(float)
    beds        = rng.randint(1,        7, N).astype(float)
    baths       = rng.choice([1, 1.5, 2, 2.5, 3, 4], N).astype(float)
    age         = rng.randint(0,      100, N).astype(float)
    condition   = rng.randint(1,        6, N).astype(float)
    garage      = rng.randint(0,        4, N).astype(float)
    stories     = rng.randint(1,        4, N).astype(float)
    lot         = rng.randint(1000,  25000, N).astype(float)
    neigh       = rng.randint(1,        5, N).astype(float)   # 1=budget…4=luxury
    proximity   = rng.randint(1,        4, N).astype(float)   # 1=urban 3=rural
    pool        = rng.randint(0,        2, N).astype(float)
    basement    = rng.randint(0,        2, N).astype(float)

    noise = rng.normal(0, 15000, N)
    price = (
        sqft * 145
        + beds * 8000
        + baths * 7500
        + (condition - 3) * 20000
        + garage * 9000
        + neigh * 25000
        - proximity * 15000
        - age * 400
        + lot * 0.5
        + pool * 28000
        + basement * 22000
        + noise
    ).clip(50000)

    X = np.column_stack([sqft, beds, baths, age, condition, garage,
                         stories, lot, neigh, proximity, pool, basement])

    SCALER = StandardScaler()
    X_scaled = SCALER.fit_transform(X)

    MODEL = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
    MODEL.fit(X_scaled, price)
    print("✅  Fallback Random Forest trained on synthetic data.")


# ── Feature names (must match training order) ────────────────────────────────
FEATURE_NAMES = [
    "sqft", "beds", "baths", "age", "condition",
    "garage", "stories", "lot", "neighborhood", "proximity",
    "pool", "basement",
]

app = Flask(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML Template (single-file, no external template folder needed)
# ═══════════════════════════════════════════════════════════════════════════════
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HomeValue AI — House Price Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0d0f14;--surface:#161a24;--card:#1e2332;--border:rgba(255,255,255,0.07);
  --accent:#e8a84c;--accent2:#5b8cff;--accent3:#52d9a4;--text:#f0ede6;--muted:#7a8099;--danger:#ff6b6b;}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;overflow-x:hidden;}
body::before{content:'';position:fixed;top:-20%;left:-10%;width:70vw;height:70vw;
  background:radial-gradient(circle,rgba(91,140,255,.06) 0%,transparent 60%);pointer-events:none;z-index:0;}
body::after{content:'';position:fixed;bottom:-20%;right:-10%;width:60vw;height:60vw;
  background:radial-gradient(circle,rgba(232,168,76,.07) 0%,transparent 60%);pointer-events:none;z-index:0;}

header{position:sticky;top:0;z-index:100;padding:1.2rem 3rem;display:flex;align-items:center;
  justify-content:space-between;border-bottom:1px solid var(--border);
  backdrop-filter:blur(14px);background:rgba(13,15,20,.75);}
.logo{display:flex;align-items:center;gap:.6rem;}
.logo-icon{width:34px;height:34px;background:linear-gradient(135deg,var(--accent),#f0c07a);
  border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:1rem;}
.logo-text{font-family:'DM Serif Display',serif;font-size:1.3rem;letter-spacing:-.02em;}
.logo-text span{color:var(--accent);}
.badge{font-size:.7rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
  background:rgba(82,217,164,.12);border:1px solid rgba(82,217,164,.25);
  color:var(--accent3);padding:.25rem .7rem;border-radius:100px;}

.hero{position:relative;z-index:1;padding:3.5rem 3rem 2rem;max-width:1100px;margin:0 auto;}
.hero-tag{display:inline-flex;align-items:center;gap:.4rem;background:rgba(232,168,76,.1);
  border:1px solid rgba(232,168,76,.25);border-radius:100px;padding:.25rem .9rem;
  font-size:.72rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;
  color:var(--accent);margin-bottom:1.2rem;animation:fadeUp .5s ease both;}
.hero h1{font-family:'DM Serif Display',serif;font-size:clamp(2rem,4.5vw,3.8rem);
  line-height:1.1;letter-spacing:-.03em;max-width:640px;margin-bottom:1rem;
  animation:fadeUp .5s ease .1s both;}
.hero h1 em{font-style:italic;color:var(--accent);}
.hero p{color:var(--muted);font-size:.98rem;max-width:480px;line-height:1.7;
  animation:fadeUp .5s ease .2s both;}

.container{position:relative;z-index:1;max-width:1100px;margin:0 auto 3rem;padding:0 3rem;}

/* stat strip */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin-bottom:2rem;
  animation:fadeUp .5s ease .3s both;}
.stat{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.2rem 1.4rem;}
.stat .icon{font-size:1.2rem;margin-bottom:.5rem;}
.stat .val{font-family:'DM Serif Display',serif;font-size:1.6rem;letter-spacing:-.03em;}
.stat .desc{font-size:.74rem;color:var(--muted);}

/* two-col layout */
.grid{display:grid;grid-template-columns:1fr 400px;gap:1.5rem;}

/* Form */
.panel{background:var(--card);border:1px solid var(--border);border-radius:18px;
  padding:2rem;animation:fadeUp .5s ease .35s both;}
.panel-head{margin-bottom:1.5rem;}
.panel-head h2{font-family:'DM Serif Display',serif;font-size:1.25rem;margin-bottom:.2rem;}
.panel-head p{font-size:.8rem;color:var(--muted);}

.sec-label{font-size:.7rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);padding-bottom:.5rem;border-bottom:1px solid var(--border);margin-bottom:1rem;margin-top:1.4rem;}
.fg{display:grid;grid-template-columns:1fr 1fr;gap:1rem;}
.fgroup{display:flex;flex-direction:column;gap:.4rem;}
.fgroup.full{grid-column:1/-1;}
label.lbl{font-size:.73rem;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);}
input[type=number],select{background:var(--surface);border:1.5px solid var(--border);border-radius:9px;
  padding:.65rem .9rem;color:var(--text);font-family:'DM Sans',sans-serif;font-size:.9rem;
  outline:none;width:100%;-webkit-appearance:none;transition:border-color .2s,box-shadow .2s;}
input[type=number]:focus,select:focus{border-color:var(--accent2);box-shadow:0 0 0 3px rgba(91,140,255,.12);}
select option{background:var(--surface);}

.slider-row{display:flex;justify-content:space-between;align-items:center;}
.slider-val{font-size:.82rem;font-weight:600;color:var(--accent);background:rgba(232,168,76,.1);
  padding:.18rem .55rem;border-radius:5px;min-width:3.2rem;text-align:center;}
input[type=range]{-webkit-appearance:none;width:100%;height:5px;background:var(--surface);
  border-radius:100px;outline:none;border:none;cursor:pointer;margin-top:.3rem;}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:17px;height:17px;
  border-radius:50%;background:var(--accent);cursor:pointer;
  box-shadow:0 0 7px rgba(232,168,76,.4);transition:transform .15s;}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.2);}

.checks{display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin-top:.3rem;}
.chk{position:relative;}
.chk input{position:absolute;opacity:0;width:0;height:0;}
.chk label{display:flex;align-items:center;gap:.4rem;background:var(--surface);
  border:1.5px solid var(--border);border-radius:9px;padding:.5rem .7rem;cursor:pointer;
  font-size:.77rem;font-weight:500;color:var(--muted);transition:all .2s;}
.chk input:checked+label{border-color:var(--accent3);background:rgba(82,217,164,.08);color:var(--accent3);}
.chk-box{width:14px;height:14px;border:2px solid currentColor;border-radius:3px;
  display:flex;align-items:center;justify-content:center;font-size:.6rem;flex-shrink:0;}
.chk input:checked+label .chk-box::after{content:'✓';}

.predict-btn{width:100%;padding:.95rem;background:linear-gradient(135deg,var(--accent),#f0b842);
  border:none;border-radius:11px;color:#1a1200;font-family:'DM Sans',sans-serif;font-size:.95rem;
  font-weight:700;cursor:pointer;margin-top:1.6rem;transition:transform .15s,box-shadow .2s,opacity .2s;
  display:flex;align-items:center;justify-content:center;gap:.5rem;}
.predict-btn:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(232,168,76,.35);}
.predict-btn:active{transform:translateY(0);}
.predict-btn.loading{opacity:.65;cursor:not-allowed;}
.spinner{width:15px;height:15px;border:2.5px solid rgba(26,18,0,.3);border-top-color:#1a1200;
  border-radius:50%;animation:spin .7s linear infinite;display:none;}
.loading .spinner{display:block;}

/* Right col */
.right{display:flex;flex-direction:column;gap:1.2rem;animation:fadeUp .5s ease .4s both;}

.result-card{background:var(--card);border:1px solid var(--border);border-radius:18px;
  padding:1.8rem;position:relative;overflow:hidden;min-height:200px;
  display:flex;flex-direction:column;justify-content:center;}
.result-card::before{content:'';position:absolute;inset:0;
  background:linear-gradient(135deg,rgba(232,168,76,.05),transparent 60%);pointer-events:none;}
.placeholder{display:flex;flex-direction:column;align-items:center;gap:.8rem;opacity:.4;text-align:center;}
.placeholder .big{font-size:2.5rem;}
.placeholder p{font-size:.85rem;color:var(--muted);line-height:1.5;}
.result-body{display:none;}
.has-result .placeholder{display:none;}
.has-result .result-body{display:block;}
.res-lbl{font-size:.72rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:.3rem;}
.res-price{font-family:'DM Serif Display',serif;font-size:2.8rem;letter-spacing:-.04em;color:var(--accent);line-height:1;}
.res-range{font-size:.78rem;color:var(--muted);margin-bottom:1.3rem;}
.mini-grid{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;margin-bottom:1.1rem;}
.mini-item{background:var(--surface);border-radius:9px;padding:.7rem;}
.mi-lbl{font-size:.68rem;color:var(--muted);font-weight:500;text-transform:uppercase;letter-spacing:.04em;margin-bottom:.25rem;}
.mi-val{font-size:.9rem;font-weight:600;}
.green{color:var(--accent3);}.blue{color:var(--accent2);}.gold{color:var(--accent);}
.conf-wrap{margin-top:.5rem;}
.conf-row{display:flex;justify-content:space-between;font-size:.74rem;color:var(--muted);margin-bottom:.35rem;}
.conf-track{background:var(--surface);height:7px;border-radius:100px;overflow:hidden;}
.conf-fill{height:100%;border-radius:100px;background:linear-gradient(90deg,var(--accent2),var(--accent3));
  width:0%;transition:width 1s ease;}

.chart-card{background:var(--card);border:1px solid var(--border);border-radius:18px;padding:1.5rem;}
.chart-title{font-family:'DM Serif Display',serif;font-size:1rem;margin-bottom:1rem;}

.fi-card{background:var(--card);border:1px solid var(--border);border-radius:18px;padding:1.5rem;}
.fi-title{font-family:'DM Serif Display',serif;font-size:1rem;margin-bottom:1rem;}
.fi-bar-wrap{display:flex;flex-direction:column;gap:.55rem;}
.fi-row{display:flex;align-items:center;gap:.7rem;}
.fi-name{font-size:.75rem;color:var(--muted);width:95px;flex-shrink:0;text-align:right;}
.fi-track{flex:1;background:var(--surface);height:8px;border-radius:100px;overflow:hidden;}
.fi-fill{height:100%;border-radius:100px;background:linear-gradient(90deg,var(--accent2),var(--accent3));transition:width 1s ease;}
.fi-pct{font-size:.72rem;color:var(--muted);width:35px;text-align:right;}

/* History table */
.history-section{margin-top:2rem;animation:fadeUp .5s ease .5s both;}
.history-section h3{font-family:'DM Serif Display',serif;font-size:1.1rem;margin-bottom:.8rem;}
.hist-table{width:100%;border-collapse:collapse;font-size:.82rem;}
.hist-table th{text-align:left;padding:.5rem .8rem;font-size:.68rem;letter-spacing:.06em;
  text-transform:uppercase;color:var(--muted);border-bottom:1px solid var(--border);}
.hist-table td{padding:.6rem .8rem;border-bottom:1px solid rgba(255,255,255,.03);color:var(--text);}
.hist-table tr:hover td{background:var(--surface);}
.empty-hist{text-align:center;padding:1.5rem;color:var(--muted);font-size:.85rem;}

/* Toast */
.toast{position:fixed;bottom:1.5rem;right:1.5rem;background:var(--card);border:1px solid var(--border);
  border-radius:12px;padding:.9rem 1.4rem;font-size:.85rem;z-index:999;
  transform:translateY(100px);opacity:0;transition:all .3s ease;display:flex;align-items:center;gap:.6rem;}
.toast.show{transform:translateY(0);opacity:1;}
.toast.success{border-color:rgba(82,217,164,.35);}
.toast.error{border-color:rgba(255,107,107,.35);}

@keyframes fadeUp{from{opacity:0;transform:translateY(18px);}to{opacity:1;transform:translateY(0);}}
@keyframes spin{to{transform:rotate(360deg);}}
@keyframes countUp{from{opacity:.4;}to{opacity:1;}}

.counting{animation:countUp .1s ease both;}

@media(max-width:860px){
  header,.hero,.container{padding-left:1.4rem;padding-right:1.4rem;}
  .grid{grid-template-columns:1fr;}
  .right{order:-1;}
  .stats{grid-template-columns:1fr 1fr;}
  .fg{grid-template-columns:1fr;}
  .checks{grid-template-columns:1fr 1fr;}
}
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-icon">🏠</div>
    <div class="logo-text">Home<span>Value</span> AI</div>
  </div>
  <span class="badge">Flask + ML</span>
</header>

<section class="hero">
  <div class="hero-tag">✦ ML-Powered Real Estate Analysis</div>
  <h1>Predict your home's <em>true value</em> instantly</h1>
  <p>Enter property details to get an AI-driven price estimate backed by a trained regression model.</p>
</section>

<div class="container">

  <!-- Stats strip -->
  <div class="stats">
    <div class="stat">
      <div class="icon">🎯</div>
      <div class="val" style="color:var(--accent)" id="r2-stat">—</div>
      <div class="desc">Model R² Score</div>
    </div>
    <div class="stat">
      <div class="icon">📊</div>
      <div class="val" style="color:var(--accent2)">12</div>
      <div class="desc">Feature dimensions</div>
    </div>
    <div class="stat">
      <div class="icon">⚡</div>
      <div class="val" style="color:var(--accent3)" id="pred-count">0</div>
      <div class="desc">Predictions made</div>
    </div>
    <div class="stat">
      <div class="icon">🔬</div>
      <div class="val" style="color:var(--danger)" id="model-type">RF</div>
      <div class="desc">Algorithm</div>
    </div>
  </div>

  <div class="grid">

    <!-- ── FORM ── -->
    <div class="panel">
      <div class="panel-head">
        <h2>Property Details</h2>
        <p>Adjust sliders and dropdowns, then click Predict</p>
      </div>

      <div class="sec-label">📐 Size &amp; Structure</div>
      <div class="fg">
        <div class="fgroup">
          <div class="slider-row"><label class="lbl">Living Area (sq ft)</label><span class="slider-val" id="sqft-v">1800</span></div>
          <input type="range" id="sqft" min="500" max="8000" value="1800" step="50"
                 oninput="sv('sqft-v',+this.value,'',{fmt:true})">
        </div>
        <div class="fgroup">
          <div class="slider-row"><label class="lbl">Lot Size (sq ft)</label><span class="slider-val" id="lot-v">6000</span></div>
          <input type="range" id="lot" min="1000" max="25000" value="6000" step="500"
                 oninput="sv('lot-v',+this.value,'',{fmt:true})">
        </div>
        <div class="fgroup">
          <label class="lbl">Bedrooms</label>
          <select id="beds">
            <option value="1">1 Bedroom</option><option value="2">2 Bedrooms</option>
            <option value="3" selected>3 Bedrooms</option><option value="4">4 Bedrooms</option>
            <option value="5">5 Bedrooms</option><option value="6">6+ Bedrooms</option>
          </select>
        </div>
        <div class="fgroup">
          <label class="lbl">Bathrooms</label>
          <select id="baths">
            <option value="1">1 Bath</option><option value="1.5">1.5 Baths</option>
            <option value="2" selected>2 Baths</option><option value="2.5">2.5 Baths</option>
            <option value="3">3 Baths</option><option value="4">4+ Baths</option>
          </select>
        </div>
        <div class="fgroup">
          <label class="lbl">Stories</label>
          <select id="stories">
            <option value="1">1 Story</option><option value="2" selected>2 Stories</option>
            <option value="3">3 Stories</option>
          </select>
        </div>
        <div class="fgroup">
          <label class="lbl">Garage Spaces</label>
          <select id="garage">
            <option value="0">No Garage</option><option value="1">1 Car</option>
            <option value="2" selected>2 Cars</option><option value="3">3 Cars</option>
          </select>
        </div>
      </div>

      <div class="sec-label">📅 Age &amp; Condition</div>
      <div class="fg">
        <div class="fgroup">
          <div class="slider-row"><label class="lbl">Year Built</label><span class="slider-val" id="year-v">1998</span></div>
          <input type="range" id="year" min="1900" max="2024" value="1998" step="1"
                 oninput="sv('year-v',this.value)">
        </div>
        <div class="fgroup">
          <label class="lbl">Overall Condition (1-5)</label>
          <select id="condition">
            <option value="1">1 — Poor</option><option value="2">2 — Fair</option>
            <option value="3" selected>3 — Average</option><option value="4">4 — Good</option>
            <option value="5">5 — Excellent</option>
          </select>
        </div>
      </div>

      <div class="sec-label">📍 Location</div>
      <div class="fg">
        <div class="fgroup">
          <label class="lbl">Neighborhood Tier</label>
          <select id="neighborhood">
            <option value="1">1 — Budget</option><option value="2" selected>2 — Mid-range</option>
            <option value="3">3 — Premium</option><option value="4">4 — Luxury</option>
          </select>
        </div>
        <div class="fgroup">
          <label class="lbl">City Proximity</label>
          <select id="proximity">
            <option value="1">Urban (&lt;5 mi)</option>
            <option value="2" selected>Suburban (5–20 mi)</option>
            <option value="3">Rural (&gt;20 mi)</option>
          </select>
        </div>
      </div>

      <div class="sec-label">✨ Amenities</div>
      <div class="checks">
        <div class="chk"><input type="checkbox" id="c-pool"><label for="c-pool"><span class="chk-box"></span>🏊 Pool</label></div>
        <div class="chk"><input type="checkbox" id="c-fireplace" checked><label for="c-fireplace"><span class="chk-box"></span>🔥 Fireplace</label></div>
        <div class="chk"><input type="checkbox" id="c-basement"><label for="c-basement"><span class="chk-box"></span>🏗️ Basement</label></div>
        <div class="chk"><input type="checkbox" id="c-newroof"><label for="c-newroof"><span class="chk-box"></span>🏠 New Roof</label></div>
        <div class="chk"><input type="checkbox" id="c-solar"><label for="c-solar"><span class="chk-box"></span>☀️ Solar</label></div>
        <div class="chk"><input type="checkbox" id="c-renovated" checked><label for="c-renovated"><span class="chk-box"></span>🔨 Renovated</label></div>
      </div>

      <button class="predict-btn" id="pred-btn" onclick="predict()">
        <div class="spinner"></div>
        <span id="btn-txt">✦ Predict Price</span>
      </button>
    </div>

    <!-- ── RIGHT COL ── -->
    <div class="right">

      <!-- Result -->
      <div class="result-card" id="result-card">
        <div class="placeholder">
          <div class="big">🏡</div>
          <p>Fill in property details and click <strong>Predict</strong> to see the estimated market value.</p>
        </div>
        <div class="result-body">
          <div class="res-lbl">Estimated Market Value</div>
          <div class="res-price" id="res-price">—</div>
          <div class="res-range" id="res-range">Range: —</div>
          <div class="mini-grid">
            <div class="mini-item"><div class="mi-lbl">Price / sqft</div><div class="mi-val blue" id="mi-ppsf">—</div></div>
            <div class="mini-item"><div class="mi-lbl">Value Score</div><div class="mi-val green" id="mi-vs">—</div></div>
            <div class="mini-item"><div class="mi-lbl">Appreciation</div><div class="mi-val gold" id="mi-appr">—</div></div>
            <div class="mini-item"><div class="mi-lbl">Est. Days to Sell</div><div class="mi-val" id="mi-dts">—</div></div>
          </div>
          <div class="conf-wrap">
            <div class="conf-row"><span>Model Confidence</span><span id="conf-pct">—</span></div>
            <div class="conf-track"><div class="conf-fill" id="conf-fill"></div></div>
          </div>
        </div>
      </div>

      <!-- Radar / Breakdown Chart -->
      <div class="chart-card">
        <div class="chart-title">📊 Price Factor Breakdown</div>
        <canvas id="radarChart" height="200"></canvas>
      </div>

      <!-- Feature Importance -->
      <div class="fi-card">
        <div class="fi-title">🔍 Feature Importance</div>
        <div class="fi-bar-wrap" id="fi-bars">
          <p style="color:var(--muted);font-size:.82rem;">Run a prediction to see feature importance.</p>
        </div>
      </div>

    </div>
  </div>

  <!-- Prediction History -->
  <div class="history-section">
    <h3>🕒 Prediction History</h3>
    <table class="hist-table">
      <thead>
        <tr>
          <th>#</th><th>Sqft</th><th>Beds/Baths</th><th>Yr Built</th>
          <th>Condition</th><th>Neighborhood</th><th>Predicted Price</th>
        </tr>
      </thead>
      <tbody id="hist-body">
        <tr><td colspan="7" class="empty-hist">No predictions yet</td></tr>
      </tbody>
    </table>
  </div>

</div>

<div class="toast" id="toast"></div>

<script>
// ── Helpers ─────────────────────────────────────────────────────────────────
function sv(id, val, suffix='', opts={}) {
  const el = document.getElementById(id);
  el.textContent = opts.fmt ? Number(val).toLocaleString() + suffix : val + suffix;
}

function fmt(n) {
  if (n >= 1e6) return '$' + (n/1e6).toFixed(2) + 'M';
  return '$' + Math.round(n).toLocaleString();
}

function showToast(msg, type='success') {
  const t = document.getElementById('toast');
  t.className = 'toast ' + type;
  t.innerHTML = (type==='success' ? '✅' : '❌') + ' ' + msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3200);
}

// ── Chart setup ─────────────────────────────────────────────────────────────
const radarCtx = document.getElementById('radarChart').getContext('2d');
const radarChart = new Chart(radarCtx, {
  type: 'radar',
  data: {
    labels: ['Size', 'Location', 'Condition', 'Age', 'Amenities', 'Layout'],
    datasets: [{
      label: 'Your Property',
      data: [0,0,0,0,0,0],
      backgroundColor: 'rgba(232,168,76,.15)',
      borderColor: '#e8a84c',
      borderWidth: 2,
      pointBackgroundColor: '#e8a84c',
      pointRadius: 4,
    }]
  },
  options: {
    responsive: true,
    scales: {
      r: {
        min: 0, max: 10,
        ticks: { display: false },
        grid: { color: 'rgba(255,255,255,.06)' },
        angleLines: { color: 'rgba(255,255,255,.06)' },
        pointLabels: { color: '#7a8099', font: { family: 'DM Sans', size: 11 } }
      }
    },
    plugins: { legend: { display: false } }
  }
});

// ── Prediction count & model info ────────────────────────────────────────────
let predCount = 0;
let histRows  = [];

// Fetch model info on load
fetch('/model_info')
  .then(r => r.json())
  .then(d => {
    document.getElementById('r2-stat').textContent  = d.r2 !== null ? d.r2.toFixed(3) : 'N/A';
    document.getElementById('model-type').textContent = d.model_type;
  })
  .catch(() => {});

// ── Main predict ─────────────────────────────────────────────────────────────
async function predict() {
  const btn = document.getElementById('pred-btn');
  btn.classList.add('loading');
  btn.disabled = true;
  document.getElementById('btn-txt').textContent = 'Analyzing…';

  const payload = {
    sqft:         +document.getElementById('sqft').value,
    beds:         +document.getElementById('beds').value,
    baths:        +document.getElementById('baths').value,
    year:         +document.getElementById('year').value,
    condition:    +document.getElementById('condition').value,
    garage:       +document.getElementById('garage').value,
    stories:      +document.getElementById('stories').value,
    lot:          +document.getElementById('lot').value,
    neighborhood: +document.getElementById('neighborhood').value,
    proximity:    +document.getElementById('proximity').value,
    pool:         document.getElementById('c-pool').checked ? 1 : 0,
    basement:     document.getElementById('c-basement').checked ? 1 : 0,
  };

  try {
    const resp = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload)
    });
    const data = await resp.json();

    if (!resp.ok) { showToast(data.error || 'Prediction failed', 'error'); return; }

    // ── Update result card ──────────────────────────────────────────────────
    document.getElementById('result-card').classList.add('has-result');

    const priceEl = document.getElementById('res-price');
    animateCount(priceEl, 0, data.price, 900, fmt);
    document.getElementById('res-range').textContent =
      `Range: ${fmt(data.range_low)} – ${fmt(data.range_high)}`;

    const ppsf  = Math.round(data.price / payload.sqft);
    const avgPpsf = [110, 135, 185, 260][payload.neighborhood - 1];
    const vs = ppsf < avgPpsf * 0.93 ? 'Below Market' : ppsf > avgPpsf * 1.07 ? 'Above Market' : 'Fair Value';
    const apprMap = [4.4, 3.2, 2.3, 1.7];
    const dts = Math.round(18 + (5 - payload.condition) * 5 + (3 - payload.neighborhood) * 3);

    document.getElementById('mi-ppsf').textContent = '$' + ppsf.toLocaleString() + '/sqft';
    document.getElementById('mi-vs').textContent   = vs;
    document.getElementById('mi-appr').textContent = '+' + apprMap[payload.neighborhood-1] + '%/yr';
    document.getElementById('mi-dts').textContent  = dts + ' days avg';

    const conf = Math.min(98, Math.max(72, data.confidence || 85));
    document.getElementById('conf-pct').textContent = conf + '%';
    setTimeout(() => { document.getElementById('conf-fill').style.width = conf + '%'; }, 200);

    // ── Radar chart ─────────────────────────────────────────────────────────
    radarChart.data.datasets[0].data = [
      Math.min(10, payload.sqft / 700),                     // Size
      (payload.neighborhood - 1) * 3.3,                    // Location
      (payload.condition - 1) * 2.5,                       // Condition
      Math.max(0, 10 - (2024 - payload.year) / 12),        // Age
      (payload.pool + payload.basement + (document.getElementById('c-fireplace').checked?1:0) +
       (document.getElementById('c-solar').checked?1:0) +
       (document.getElementById('c-renovated').checked?1:0)) * 2, // Amenities
      (payload.beds * 1.2 + payload.baths * 1.0 + payload.garage * 0.8) // Layout
    ].map(v => Math.min(10, Math.max(0, v)));
    radarChart.update();

    // ── Feature importance bars ──────────────────────────────────────────────
    if (data.feature_importance) {
      renderFiBars(data.feature_importance);
    }

    // ── History ──────────────────────────────────────────────────────────────
    predCount++;
    document.getElementById('pred-count').textContent = predCount;
    addHistRow(payload, data.price);

    showToast('Prediction complete — ' + fmt(data.price));
  } catch (err) {
    showToast('Server error: ' + err.message, 'error');
  } finally {
    btn.classList.remove('loading');
    btn.disabled = false;
    document.getElementById('btn-txt').textContent = '✦ Predict Price';
  }
}

function animateCount(el, from, to, dur, formatter) {
  const start = performance.now();
  function step(now) {
    const t = Math.min((now - start) / dur, 1);
    const ease = 1 - Math.pow(1 - t, 3);
    el.textContent = formatter(from + (to - from) * ease);
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function renderFiBars(fi) {
  const container = document.getElementById('fi-bars');
  container.innerHTML = '';
  const sorted = Object.entries(fi).sort((a,b) => b[1]-a[1]).slice(0,8);
  const max = sorted[0][1];
  sorted.forEach(([name, val]) => {
    const pct = Math.round(val / max * 100);
    container.innerHTML += `
      <div class="fi-row">
        <span class="fi-name">${name}</span>
        <div class="fi-track"><div class="fi-fill" style="width:0%" data-pct="${pct}"></div></div>
        <span class="fi-pct">${pct}%</span>
      </div>`;
  });
  setTimeout(() => {
    container.querySelectorAll('.fi-fill').forEach(el => {
      el.style.width = el.dataset.pct + '%';
    });
  }, 100);
}

function addHistRow(p, price) {
  const tbody = document.getElementById('hist-body');
  if (predCount === 1) tbody.innerHTML = '';
  const condMap = ['','Poor','Fair','Avg','Good','Excel'];
  const neighMap = ['','Budget','Mid','Premium','Luxury'];
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td style="color:var(--muted)">${predCount}</td>
    <td>${Number(p.sqft).toLocaleString()}</td>
    <td>${p.beds}bd / ${p.baths}ba</td>
    <td>${p.year}</td>
    <td>${condMap[p.condition]}</td>
    <td>${neighMap[p.neighborhood]}</td>
    <td style="color:var(--accent);font-weight:600">${fmt(price)}</td>`;
  tbody.insertBefore(tr, tbody.firstChild);
  if (tbody.children.length > 10) tbody.removeChild(tbody.lastChild);
}
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/model_info")
def model_info():
    """Return basic model metadata."""
    model_type = type(MODEL).__name__
    # Abbreviate long class names
    abbrev = {
        "RandomForestRegressor": "RF",
        "GradientBoostingRegressor": "GBR",
        "LinearRegression": "LR",
        "Ridge": "Ridge",
        "Lasso": "Lasso",
        "SVR": "SVR",
        "XGBRegressor": "XGB",
    }
    short = abbrev.get(model_type, model_type[:6])

    # Try to get R² from model (not always available)
    r2 = None
    if hasattr(MODEL, "oob_score_"):
        r2 = MODEL.oob_score_

    return jsonify({"model_type": short, "r2": r2, "loaded_from_disk": MODEL_LOADED})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON body with property features.
    Returns predicted price + supporting analytics.
    """
    try:
        data = request.get_json(force=True)

        # Build feature vector (must match FEATURE_NAMES order)
        year  = float(data.get("year", 1998))
        age   = 2025 - year

        features = np.array([[
            float(data.get("sqft",        1800)),
            float(data.get("beds",           3)),
            float(data.get("baths",          2)),
            age,
            float(data.get("condition",      3)),
            float(data.get("garage",         2)),
            float(data.get("stories",        2)),
            float(data.get("lot",         6000)),
            float(data.get("neighborhood",   2)),
            float(data.get("proximity",      2)),
            float(data.get("pool",           0)),
            float(data.get("basement",       0)),
        ]])

        # Scale & predict
        features_scaled = SCALER.transform(features)
        price = float(MODEL.predict(features_scaled)[0])
        price = max(price, 50_000)

        # Confidence: use prediction std-dev across trees if available
        confidence = 85
        if hasattr(MODEL, "estimators_"):
            preds = np.array([e.predict(features_scaled)[0] for e in MODEL.estimators_])
            std   = preds.std()
            cv    = std / max(price, 1)
            confidence = int(np.clip(100 - cv * 120, 70, 98))

        spread     = price * (1 - confidence / 100) * 0.9
        range_low  = max(price - spread, 0)
        range_high = price + spread

        # Feature importance (if the model exposes it)
        fi_dict = {}
        if hasattr(MODEL, "feature_importances_"):
            fi_raw = MODEL.feature_importances_
            labels = ["Sqft", "Beds", "Baths", "Age", "Condition",
                      "Garage", "Stories", "Lot", "Neighborhood", "Proximity",
                      "Pool", "Basement"]
            total = fi_raw.sum() or 1
            fi_dict = {labels[i]: round(float(fi_raw[i] / total * 100), 1)
                       for i in range(len(fi_raw))}

        return jsonify({
            "price":              round(price, 2),
            "range_low":          round(range_low, 2),
            "range_high":         round(range_high, 2),
            "confidence":         confidence,
            "feature_importance": fi_dict,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL_LOADED})


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "─"*60)
    print("  HomeValue AI  |  http://127.0.0.1:5000")
    print("  Model loaded from disk:", MODEL_LOADED)
    print("─"*60 + "\n")
    app.run(debug=True, port=5000)
