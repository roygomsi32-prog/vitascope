"""
VitaScope - Observatoire du Bien-Être Étudiant
Application d'analyse des habitudes de vie et de bien-être en milieu universitaire
Développée dans le cadre du cours INF 232 EC2 - Master Informatique
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import io
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ─── Configuration globale de la page ────────────────────────────────────────
st.set_page_config(
    page_title="VitaScope — Bien-Être Étudiant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personnalisé ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #30363d;
  }
  [data-testid="stSidebar"] * { color: #e6edf3 !important; }
  [data-testid="stSidebar"] .stRadio label { 
    padding: 6px 10px; border-radius: 6px; cursor: pointer;
  }
  [data-testid="stSidebar"] .stRadio label:hover {
    background: #21262d;
  }

  /* En-tête */
  .vitascope-header {
    background: linear-gradient(135deg, #0a3d2e 0%, #155c42 50%, #1a7a58 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 1.5rem;
    border: 1px solid #2ea04326;
    position: relative;
    overflow: hidden;
  }
  .vitascope-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, #2ea04320 0%, transparent 70%);
    pointer-events: none;
  }
  .vitascope-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #7ee787;
    margin: 0;
    letter-spacing: -1px;
  }
  .vitascope-subtitle {
    color: #adbac7;
    font-size: 1rem;
    margin-top: 0.3rem;
    font-weight: 300;
  }

  /* Cartes métriques */
  .metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.2s;
  }
  .metric-card:hover { border-color: #2ea043; }
  .metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #7ee787;
  }
  .metric-label { color: #8b949e; font-size: 0.82rem; margin-top: 0.2rem; }

  /* Sections */
  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #e6edf3;
    border-left: 4px solid #2ea043;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
  }

  /* Boutons */
  .stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 500;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px #2ea04340;
  }

  /* Info, succès, avertissements */
  .info-box {
    background: #0d2a3d;
    border: 1px solid #1f6feb;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #58a6ff;
    margin: 0.8rem 0;
    font-size: 0.9rem;
  }
  .success-box {
    background: #0d2818;
    border: 1px solid #2ea043;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #7ee787;
    margin: 0.8rem 0;
    font-size: 0.9rem;
  }

  /* Formulaire */
  .form-section {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
  }
  .form-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #7ee787;
    margin-bottom: 1rem;
  }

  /* Fond principal */
  .main .block-container {
    background: #0d1117;
    padding-top: 1.5rem;
  }

  /* Logo sidebar */
  .sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #7ee787 !important;
    text-align: center;
    padding: 1rem 0;
    border-bottom: 1px solid #30363d;
    margin-bottom: 1rem;
  }
  .sidebar-tagline {
    font-size: 0.72rem;
    color: #8b949e !important;
    text-align: center;
    margin-top: -0.5rem;
    margin-bottom: 1rem;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    color: #8b949e;
  }
  .stTabs [aria-selected="true"] {
    color: #7ee787;
    border-bottom-color: #2ea043;
  }

  /* Résultats ML */
  .result-badge {
    display: inline-block;
    background: #1f6feb20;
    border: 1px solid #1f6feb;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    color: #58a6ff;
    font-size: 0.82rem;
    font-weight: 500;
  }
  .result-badge.green {
    background: #2ea04320;
    border-color: #2ea043;
    color: #7ee787;
  }
  .result-badge.orange {
    background: #e3b34120;
    border-color: #e3b341;
    color: #e3b341;
  }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DONNÉES : Stockage persistant (fichier JSON local)
# ═══════════════════════════════════════════════════════════════════════════════

DATA_FILE = "vitascope_data.json"

def charger_donnees():
    """Charge les données depuis le fichier JSON local."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def sauvegarder_donnees(donnees: list):
    """Sauvegarde les données dans le fichier JSON."""
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(donnees, f, ensure_ascii=False, indent=2)

def generer_donnees_demo(n=120):
    """Génère un jeu de données synthétiques réalistes pour la démonstration."""
    np.random.seed(42)
    filieres = ['Informatique', 'Mathématiques', 'Physique', 'Biologie', 'Économie', 'Droit']
    niveaux = ['Licence 1', 'Licence 2', 'Licence 3', 'Master 1', 'Master 2']
    genres = ['Homme', 'Femme', 'Non-binaire']
    logements = ['Résidence U', 'Appartement', 'Domicile familial', 'Colocation']
    activites = ['Aucune', '1-2h/semaine', '3-5h/semaine', '>5h/semaine']

    data = []
    for i in range(n):
        heures_sommeil = np.clip(np.random.normal(6.8, 1.2), 3, 10)
        sport = np.random.choice([0, 1, 2, 3, 5, 7], p=[0.15, 0.25, 0.25, 0.2, 0.1, 0.05])
        stress = np.clip(np.random.normal(6.2, 2.0), 1, 10)
        qualite_alimentation = np.clip(np.random.normal(5.5, 2.0), 1, 10)
        ecran_loisirs = np.clip(np.random.normal(3.5, 1.8), 0, 10)
        social = np.clip(np.random.normal(5.0, 2.0), 1, 10)
        temps_etude = np.clip(np.random.normal(5.0, 2.0), 1, 12)
        revenu_mensuel = np.clip(np.random.normal(650, 250), 100, 1500)

        # Moyenne académique corrélée à plusieurs variables
        moyenne = (
            12
            + 0.4 * heures_sommeil
            + 0.3 * qualite_alimentation
            - 0.25 * stress
            + 0.6 * temps_etude
            + 0.2 * sport
            - 0.1 * ecran_loisirs
            + np.random.normal(0, 1.5)
        )
        moyenne = np.clip(moyenne, 7, 20)

        # Score bien-être global
        score_bienetre = (
            heures_sommeil * 0.3
            + qualite_alimentation * 0.2
            + sport * 0.5
            - stress * 0.4
            + social * 0.2
            + np.random.normal(0, 0.5)
        )
        score_bienetre = np.clip(score_bienetre, 1, 10)

        data.append({
            "timestamp": datetime.now().isoformat(),
            "age": int(np.clip(np.random.normal(21, 2.5), 18, 30)),
            "genre": np.random.choice(genres, p=[0.48, 0.48, 0.04]),
            "filiere": np.random.choice(filieres),
            "niveau": np.random.choice(niveaux),
            "logement": np.random.choice(logements, p=[0.3, 0.25, 0.25, 0.2]),
            "heures_sommeil": round(heures_sommeil, 1),
            "heures_sport": round(float(sport), 1),
            "stress": round(stress, 1),
            "qualite_alimentation": round(qualite_alimentation, 1),
            "heures_ecran_loisirs": round(ecran_loisirs, 1),
            "vie_sociale": round(social, 1),
            "heures_etude_semaine": round(temps_etude, 1),
            "revenu_mensuel": round(revenu_mensuel, 0),
            "moyenne_academique": round(moyenne, 2),
            "score_bienetre": round(score_bienetre, 2),
            "activite_extrascolaire": np.random.choice(activites),
            "satisfaction_generale": int(np.clip(np.random.normal(6, 2), 1, 10)),
        })
    return data

def obtenir_dataframe():
    """Retourne un DataFrame : données réelles + démo si nécessaire."""
    donnees = charger_donnees()
    if len(donnees) < 5:
        demo = generer_donnees_demo(120)
        all_data = demo + donnees
        df = pd.DataFrame(all_data)
        df["source"] = (["Démo"] * len(demo)) + (["Réelle"] * len(donnees))
        return df, True  # True = mode démo actif
    df = pd.DataFrame(donnees)
    df["source"] = "Réelle"
    return df, False


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🌿 VitaScope</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Observatoire du Bien-Être Étudiant</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Accueil",
            "📋 Collecte de données",
            "📊 Tableau de bord",
            "📈 Régression simple",
            "📉 Régression multiple",
            "🔭 Réduction ACP",
            "🎯 Classification supervisée",
            "🔵 Clustering",
            "📤 Export des résultats",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    df, mode_demo = obtenir_dataframe()
    n_reelles = len(df[df["source"] == "Réelle"]) if "source" in df.columns else 0

    st.markdown(f"""
    <div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:0.8rem;font-size:0.82rem;'>
      <div style='color:#8b949e;'>Base de données</div>
      <div style='color:#7ee787;font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;'>{len(df)}</div>
      <div style='color:#8b949e;'>observations totales</div>
      <hr style='border-color:#30363d;margin:0.5rem 0;'>
      <div style='color:#{'e3b341' if mode_demo else '7ee787'};'>{'⚠️ Mode démo actif' if mode_demo else '✅ Données réelles'}</div>
      <div style='color:#8b949e;'>{n_reelles} réponse(s) réelle(s)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:2rem;color:#484f58;font-size:0.72rem;text-align:center;'>
      VitaScope v1.0 · INF 232 EC2<br>Master Informatique · 2025-2026
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : ACCUEIL
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Accueil":
    st.markdown("""
    <div class="vitascope-header">
      <p class="vitascope-title">🌿 VitaScope</p>
      <p class="vitascope-subtitle">Observatoire du Bien-Être Étudiant · INF 232 EC2 — Master Informatique</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-title">À propos du projet</div>', unsafe_allow_html=True)
        st.markdown("""
        **VitaScope** est une plateforme d'analyse des habitudes de vie en milieu universitaire.
        Elle collecte des données sur le sommeil, le stress, l'alimentation, le sport et les performances
        académiques des étudiants, puis les analyse avec des méthodes statistiques et de machine learning.

        **Objectifs scientifiques :**
        - Identifier les facteurs influençant la réussite académique
        - Détecter des profils types d'étudiants par clustering
        - Prédire le bien-être global à partir des habitudes de vie
        - Réduire la dimensionnalité pour visualiser les patterns cachés
        """)

        st.markdown('<div class="section-title">Modules d\'analyse disponibles</div>', unsafe_allow_html=True)

        modules = [
            ("📈", "Régression Linéaire Simple", "Relation entre sommeil et performance académique"),
            ("📉", "Régression Linéaire Multiple", "Prédiction de la moyenne à partir de plusieurs variables"),
            ("🔭", "Analyse en Composantes Principales", "Réduction dimensionnelle et visualisation 2D/3D"),
            ("🎯", "Classification Supervisée", "Random Forest & SVM pour prédire le niveau de bien-être"),
            ("🔵", "Clustering K-Means", "Identification de profils types d'étudiants"),
        ]

        for icon, titre, desc in modules:
            st.markdown(f"""
            <div style='display:flex;gap:1rem;align-items:start;margin-bottom:0.8rem;
                        background:#161b22;border:1px solid #30363d;border-radius:10px;padding:0.8rem 1rem;'>
              <span style='font-size:1.5rem;'>{icon}</span>
              <div>
                <div style='color:#e6edf3;font-weight:600;font-size:0.95rem;'>{titre}</div>
                <div style='color:#8b949e;font-size:0.82rem;'>{desc}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Statistiques en temps réel</div>', unsafe_allow_html=True)

        cols_m = st.columns(2)
        metriques = [
            (len(df), "Observations"),
            (len(df.columns) - 2, "Variables"),
            (n_reelles, "Réponses réelles"),
            (5, "Modèles ML"),
        ]
        for i, (val, label) in enumerate(metriques):
            with cols_m[i % 2]:
                st.markdown(f"""
                <div class="metric-card" style='margin-bottom:0.8rem;'>
                  <div class="metric-value">{val}</div>
                  <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Variables collectées</div>', unsafe_allow_html=True)
        variables = [
            "Âge, genre, filière, niveau",
            "Heures de sommeil / nuit",
            "Activité physique (h/semaine)",
            "Niveau de stress (1-10)",
            "Qualité de l'alimentation (1-10)",
            "Temps d'écran loisirs (h/j)",
            "Vie sociale (1-10)",
            "Temps d'étude (h/semaine)",
            "Revenu mensuel (FCFA/€)",
            "Moyenne académique (/20)",
            "Score de bien-être global",
        ]
        for v in variables:
            st.markdown(f"<div style='color:#8b949e;font-size:0.85rem;padding:0.2rem 0;'>• {v}</div>",
                        unsafe_allow_html=True)

        if mode_demo:
            st.markdown("""
            <div class="info-box">
              ℹ️ <strong>Mode démonstration actif</strong><br>
              Moins de 5 réponses réelles collectées. Les analyses utilisent des données synthétiques réalistes.
              Remplissez le formulaire pour alimenter la base avec vos données !
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : FORMULAIRE DE COLLECTE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📋 Collecte de données":
    st.markdown("""
    <div class="vitascope-header">
      <p class="vitascope-title">📋 Collecte de données</p>
      <p class="vitascope-subtitle">Partagez vos habitudes de vie — données anonymes et confidentielles</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
      ℹ️ Ce formulaire est <strong>entièrement anonyme</strong>. Aucune donnée personnelle identifiable n'est collectée.
      Vos réponses contribuent à une meilleure compréhension du bien-être étudiant.
    </div>
    """, unsafe_allow_html=True)

    with st.form("formulaire_vitascope", clear_on_submit=True):
        # ── Section 1 : Profil ────────────────────────────────────────────────
        st.markdown('<div class="form-section-title">👤 Section 1 — Profil académique</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Âge", min_value=17, max_value=35, value=21, step=1)
        with c2:
            genre = st.selectbox("Genre", ["Homme", "Femme", "Non-binaire", "Préfère ne pas répondre"])
        with c3:
            niveau = st.selectbox("Niveau d'études", ["Licence 1", "Licence 2", "Licence 3", "Master 1", "Master 2", "Doctorat"])

        c4, c5 = st.columns(2)
        with c4:
            filiere = st.selectbox("Filière", ["Informatique", "Mathématiques", "Physique", "Biologie",
                                                "Économie", "Droit", "Médecine", "Lettres", "Autre"])
        with c5:
            logement = st.selectbox("Type de logement", ["Résidence universitaire", "Appartement seul",
                                                          "Colocation", "Domicile familial", "Autre"])

        st.markdown("---")
        # ── Section 2 : Habitudes de vie ─────────────────────────────────────
        st.markdown('<div class="form-section-title">🌙 Section 2 — Habitudes de vie</div>', unsafe_allow_html=True)
        c6, c7 = st.columns(2)
        with c6:
            heures_sommeil = st.slider("Heures de sommeil par nuit (en moyenne)", 3.0, 12.0, 7.0, 0.5,
                                       help="Nombre moyen d'heures dormies par nuit sur la semaine")
            heures_sport = st.slider("Activité physique (heures/semaine)", 0.0, 20.0, 3.0, 0.5)
            heures_ecran = st.slider("Heures d'écran loisirs par jour (hors études)", 0.0, 12.0, 3.0, 0.5)
        with c7:
            qualite_alim = st.slider("Qualité de votre alimentation (1=très mauvaise, 10=excellente)", 1, 10, 6)
            vie_sociale = st.slider("Qualité de votre vie sociale (1=isolé·e, 10=très sociable)", 1, 10, 5)
            heures_etude = st.slider("Heures d'étude par semaine (hors cours)", 1.0, 20.0, 5.0, 0.5)

        activite_extra = st.selectbox("Activité extrascolaire régulière",
                                       ["Aucune", "1-2h/semaine", "3-5h/semaine", ">5h/semaine"])

        st.markdown("---")
        # ── Section 3 : Bien-être et performance ─────────────────────────────
        st.markdown('<div class="form-section-title">🎓 Section 3 — Performance & Bien-être</div>', unsafe_allow_html=True)
        c8, c9, c10 = st.columns(3)
        with c8:
            stress = st.slider("Niveau de stress (1=très faible, 10=extrême)", 1, 10, 5)
        with c9:
            moyenne = st.number_input("Moyenne académique (/20)", min_value=0.0, max_value=20.0, value=12.0, step=0.5)
        with c10:
            satisfaction = st.slider("Satisfaction générale de votre vie (1-10)", 1, 10, 6)

        revenu = st.number_input("Revenu mensuel approximatif (€ ou FCFA)", min_value=0, max_value=10000, value=500, step=50,
                                  help="Aides, bourse, job étudiant, soutien familial... Laissez 0 si préférez ne pas répondre")

        st.markdown("---")
        submitted = st.form_submit_button("✅ Soumettre mes réponses", use_container_width=True)

    if submitted:
        # Validation basique
        erreurs = []
        if age < 17 or age > 35:
            erreurs.append("L'âge doit être entre 17 et 35 ans.")
        if moyenne < 0 or moyenne > 20:
            erreurs.append("La moyenne doit être entre 0 et 20.")

        if erreurs:
            for e in erreurs:
                st.error(f"❌ {e}")
        else:
            # Calcul du score de bien-être
            score_bienetre = (
                heures_sommeil * 0.3
                + qualite_alim * 0.2
                + heures_sport * 0.5
                - stress * 0.4
                + vie_sociale * 0.2
            )
            score_bienetre = round(np.clip(score_bienetre, 1, 10), 2)

            nouvelle_entree = {
                "timestamp": datetime.now().isoformat(),
                "age": age,
                "genre": genre,
                "filiere": filiere,
                "niveau": niveau,
                "logement": logement,
                "heures_sommeil": heures_sommeil,
                "heures_sport": heures_sport,
                "stress": stress,
                "qualite_alimentation": qualite_alim,
                "heures_ecran_loisirs": heures_ecran,
                "vie_sociale": vie_sociale,
                "heures_etude_semaine": heures_etude,
                "revenu_mensuel": revenu,
                "moyenne_academique": moyenne,
                "score_bienetre": score_bienetre,
                "activite_extrascolaire": activite_extra,
                "satisfaction_generale": satisfaction,
            }

            donnees = charger_donnees()
            donnees.append(nouvelle_entree)
            sauvegarder_donnees(donnees)

            st.markdown(f"""
            <div class="success-box">
              ✅ <strong>Merci pour votre participation !</strong><br>
              Votre score de bien-être calculé : <strong>{score_bienetre}/10</strong><br>
              Les analyses se mettront à jour automatiquement avec vos données.
            </div>
            """, unsafe_allow_html=True)
            st.balloons()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : TABLEAU DE BORD
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Tableau de bord":
    st.markdown("""
    <div class="vitascope-header">
      <p class="vitascope-title">📊 Tableau de bord</p>
      <p class="vitascope-subtitle">Analyse descriptive complète des données collectées</p>
    </div>
    """, unsafe_allow_html=True)

    if mode_demo:
        st.markdown('<div class="info-box">⚠️ Mode démo — données synthétiques incluses dans l\'analyse.</div>',
                    unsafe_allow_html=True)

    cols_m = st.columns(4)
    stats_data = [
        (f"{df['moyenne_academique'].mean():.1f}/20", "Moyenne académique", "#7ee787"),
        (f"{df['score_bienetre'].mean():.1f}/10", "Bien-être moyen", "#58a6ff"),
        (f"{df['stress'].mean():.1f}/10", "Stress moyen", "#e3b341"),
        (f"{df['heures_sommeil'].mean():.1f}h", "Sommeil moyen/nuit", "#bc8cff"),
    ]
    for i, (val, label, color) in enumerate(stats_data):
        with cols_m[i]:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value" style='color:{color};'>{val}</div>
              <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔗 Corrélations", "👥 Démographie", "📋 Statistiques"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="moyenne_academique", nbins=20, title="Distribution des moyennes académiques",
                               color_discrete_sequence=["#2ea043"], template="plotly_dark")
            fig.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#e6edf3")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.histogram(df, x="heures_sommeil", nbins=20, title="Distribution du sommeil",
                                color_discrete_sequence=["#58a6ff"], template="plotly_dark")
            fig2.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#e6edf3")
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            fig3 = px.histogram(df, x="score_bienetre", nbins=20, title="Distribution du score de bien-être",
                                color_discrete_sequence=["#bc8cff"], template="plotly_dark")
            fig3.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#e6edf3")
            st.plotly_chart(fig3, use_container_width=True)

            fig4 = px.histogram(df, x="stress", nbins=10, title="Distribution du niveau de stress",
                                color_discrete_sequence=["#e3b341"], template="plotly_dark")
            fig4.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#e6edf3")
            st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        vars_num = ["heures_sommeil", "heures_sport", "stress", "qualite_alimentation",
                    "heures_ecran_loisirs", "vie_sociale", "heures_etude_semaine",
                    "moyenne_academique", "score_bienetre"]
        corr = df[vars_num].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", title="Matrice de corrélation",
                             color_continuous_scale="RdYlGn", zmin=-1, zmax=1,
                             template="plotly_dark")
        fig_corr.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#e6edf3",
                               height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_sc = px.scatter(df, x="heures_sommeil", y="moyenne_academique", color="stress",
                                title="Sommeil vs Moyenne (couleur=Stress)", template="plotly_dark",
                                color_continuous_scale="RdYlGn_r")
            fig_sc.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#e6edf3")
            st.plotly_chart(fig_sc, use_container_width=True)
        with col_b:
            fig_sc2 = px.scatter(df, x="heures_etude_semaine", y="moyenne_academique", color="score_bienetre",
                                 title="Temps d'étude vs Moyenne", template="plotly_dark",
                                 color_continuous_scale="Viridis")
            fig_sc2.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#e6edf3")
            st.plotly_chart(fig_sc2, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig_f = px.pie(df, names="filiere", title="Répartition par filière",
                           template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Set2)
            fig_f.update_layout(paper_bgcolor="#161b22", font_color="#e6edf3")
            st.plotly_chart(fig_f, use_container_width=True)
        with col2:
            fig_g = px.pie(df, names="genre", title="Répartition par genre",
                           template="plotly_dark", color_discrete_sequence=["#2ea043", "#58a6ff", "#bc8cff"])
            fig_g.update_layout(paper_bgcolor="#161b22", font_color="#e6edf3")
            st.plotly_chart(fig_g, use_container_width=True)

        fig_box = px.box(df, x="filiere", y="moyenne_academique", color="filiere",
                         title="Moyennes par filière", template="plotly_dark",
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_box.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#e6edf3",
                               showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with tab4:
        vars_stats = ["heures_sommeil", "heures_sport", "stress", "qualite_alimentation",
                      "heures_ecran_loisirs", "vie_sociale", "heures_etude_semaine",
                      "revenu_mensuel", "moyenne_academique", "score_bienetre"]
        stats_df = df[vars_stats].describe().T.round(2)
        stats_df.columns = ["N", "Moyenne", "Écart-type", "Min", "Q1", "Médiane", "Q3", "Max"]
        st.dataframe(stats_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : RÉGRESSION LINÉAIRE SIMPLE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Régression simple":
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split

    st.markdown("""
    <div class="vitascope-header">
      <p class="vitascope-title">📈 Régression Linéaire Simple</p>
      <p class="vitascope-subtitle">Modélisation de la relation entre deux variables quantitatives</p>
    </div>
    """, unsafe_allow_html=True)

    vars_num = ["heures_sommeil", "heures_sport", "stress", "qualite_alimentation",
                "heures_ecran_loisirs", "vie_sociale", "heures_etude_semaine",
                "revenu_mensuel", "score_bienetre"]

    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("Variable explicative (X)", vars_num, index=0,
                             format_func=lambda x: x.replace("_", " ").title())
    with col2:
        y_var = st.selectbox("Variable à expliquer (Y)", ["moyenne_academique", "score_bienetre"],
                             format_func=lambda x: x.replace("_", " ").title())

    df_clean = df[[x_var, y_var]].dropna()
    X = df_clean[[x_var]].values
    y = df_clean[y_var].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modele = LinearRegression()
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)
    y_pred_all = modele.predict(X)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    coef = modele.coef_[0]
    intercept = modele.intercept_

    # Métriques
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col_m, val, label, color in [
        (mc1, f"{r2:.3f}", "R² (test)", "#7ee787"),
        (mc2, f"{rmse:.3f}", "RMSE", "#58a6ff"),
        (mc3, f"{coef:.3f}", "Coefficient β₁", "#e3b341"),
        (mc4, f"{intercept:.3f}", "Intercept β₀", "#bc8cff"),
    ]:
        with col_m:
            st.markdown(f"""
            <div class="metric-card" style='margin-top:0.5rem;'>
              <div class="metric-value" style='color:{color};font-size:1.5rem;'>{val}</div>
              <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    col_g, col_r = st.columns([2, 1])
    with col_g:
        # Nuage de points + droite de régression
        x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_range = modele.predict(x_range)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X.flatten(), y=y, mode="markers", name="Observations",
                                 marker=dict(color="#58a6ff", opacity=0.6, size=6)))
        fig.add_trace(go.Scatter(x=x_range.flatten(), y=y_range, mode="lines",
                                 name=f"Régression (R²={r2:.3f})",
                                 line=dict(color="#2ea043", width=2.5)))
        fig.update_layout(
            title=f"Régression : {x_var} → {y_var}",
            xaxis_title=x_var.replace("_", " ").title(),
            yaxis_title=y_var.replace("_", " ").title(),
            template="plotly_dark",
            paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#e6edf3"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Équation du modèle</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#0d2818;border:1px solid #2ea043;border-radius:10px;padding:1.2rem;font-family:monospace;color:#7ee787;margin:0.5rem 0;'>
          ŷ = {intercept:.4f} + {coef:.4f} × X
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Interprétation</div>', unsafe_allow_html=True)
        direction = "augmente" if coef > 0 else "diminue"
        force = "forte" if abs(r2) > 0.5 else ("modérée" if abs(r2) > 0.2 else "faible")
        st.markdown(f"""
        <div style='color:#adbac7;font-size:0.88rem;line-height:1.6;'>
        • Une augmentation d'une unité de <strong>{x_var}</strong> est associée
          à une variation de <strong>{coef:.4f}</strong> de <strong>{y_var}</strong>.<br><br>
        • Le modèle explique <strong>{r2*100:.1f}%</strong> de la variance — corrélation <strong>{force}</strong>.<br><br>
        • RMSE = <strong>{rmse:.3f}</strong> : erreur moyenne de prédiction.
        </div>
        """, unsafe_allow_html=True)

        # Résidus
        residus = y_test - y_pred
        fig_res = px.histogram(residus, nbins=15, title="Résidus", template="plotly_dark",
                               color_discrete_sequence=["#e3b341"])
        fig_res.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#e6edf3",
                               height=250, margin=dict(t=40, b=20))
        st.plotly_chart(fig_res, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : RÉGRESSION LINÉAIRE MULTIPLE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📉 Régression multiple":
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler

    st.markdown("""
    <div class="vitascope-header">
      <p class="vitascope-title">📉 Régression Linéaire Multiple</p>
      <p class="vitascope-subtitle">Prédiction par combinaison de plusieurs variables explicatives</p>
    </div>
    """, unsafe_allow_html=True)

    vars_disponibles = ["heures_sommeil", "heures_sport", "stress", "qualite_alimentation",
                        "heures_ecran_loisirs", "vie_sociale", "heures_etude_semaine",
                        "revenu_mensuel", "age", "satisfaction_generale"]

    col1, col2, col3 = st.columns(3)
    with col1:
        y_var = st.selectbox("Variable cible (Y)", ["moyenne_academique", "score_bienetre"])
    with col2:
        x_vars = st.multiselect("Variables explicatives (X)", vars_disponibles,
                                default=["heures_sommeil", "stress", "heures_etude_semaine", "qualite_alimentation"])
    with col3:
        modele_type = st.selectbox("Type de modèle", ["Régression classique (OLS)", "Ridge (L2)", "Lasso (L1)"])

    if len(x_vars) < 2:
        st.warning("⚠️ Sélectionnez au moins 2 variables explicatives.")
    else:
        df_m = df[x_vars + [y_var]].dropna()
        X = df_m[x_vars].values
        y = df_m[y_var].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        if modele_type == "Ridge (L2)":
            alpha = st.slider("Paramètre de régularisation α", 0.01, 10.0, 1.0, 0.01)
            reg = Ridge(alpha=alpha)
        elif modele_type == "Lasso (L1)":
            alpha = st.slider("Paramètre de régularisation α", 0.001, 5.0, 0.1, 0.001)
            reg = Lasso(alpha=alpha, max_iter=5000)
        else:
            reg = LinearRegression()

        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        cv_scores = cross_val_score(reg, X_scaled, y, cv=5, scoring="r2")

        mc = st.columns(4)
        for col_m, val, label, color in [
            (mc[0], f"{r2:.3f}", "R² Test", "#7ee787"),
            (mc[1], f"{rmse:.3f}", "RMSE", "#58a6ff"),
            (mc[2], f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}", "R² CV (5-fold)", "#e3b341"),
            (mc[3], len(x_vars), "Variables", "#bc8cff"),
        ]:
            with col_m:
                st.markdown(f"""
                <div class="metric-card" style='margin-top:0.5rem;'>
                  <div class="metric-value" style='color:{color};font-size:1.3rem;'>{val}</div>
                  <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        col_g, col_d = st.columns(2)
        with col_g:
            # Importance des coefficients
            coefs = pd.Series(reg.coef_, index=x_vars).sort_values(key=abs, ascending=True)
            colors_coef = ["#e05252" if c < 0 else "#2ea043" for c in coefs]
            fig_coef = go.Figure(go.Bar(y=coefs.index, x=coefs.values, orientation="h",
                                        marker_color=colors_coef))
            fig_coef.update_layout(title="Coefficients standardisés", template="plotly_dark",
                                   paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                                   font_color="#e6edf3", height=350)
            st.plotly_chart(fig_coef, use_container_width=True)

        with col_d:
            # Valeurs réelles vs prédites
            fig_vp = go.Figure()
            fig_vp.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                        marker=dict(color="#58a6ff", size=6, opacity=0.7)))
            fig_vp.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()],
                                        mode="lines", line=dict(color="#2ea043", dash="dash"),
                                        name="Idéal"))
            fig_vp.update_layout(title="Valeurs réelles vs Prédites", template="plotly_dark",
                                 paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                                 font_color="#e6edf3", height=350,
                                 xaxis_title="Valeurs réelles", yaxis_title="Valeurs prédites")
            st.plotly_chart(fig_vp, use_container_width=True)

        # Tableau récapitulatif
        st.markdown('<div class="section-title">Tableau des coefficients</div>', unsafe_allow_html=True)
        coef_abs_sum = np.abs(reg.coef_).sum()
        importance = (np.abs(reg.coef_) / coef_abs_sum * 100).round(1) if coef_abs_sum > 0 else np.zeros(len(x_vars))
        coef_df = pd.DataFrame({
            "Variable": x_vars,
            "Coefficient standardisé": reg.coef_.round(4),
            "Importance relative (%)": importance,
        }).sort_values("Importance relative (%)", ascending=False)
        st.dataframe(coef_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : ACP / RÉDUCTION DIMENSIONNALITÉ
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔭 Réduction ACP":
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    st.markdown("""
    <div class="vitascope-header">
      <p class="vitascope-title">🔭 Analyse en Composantes Principales</p>
      <p class="vitascope-subtitle">Réduction de dimensionnalité et découverte de structures latentes</p>
    </div>
    """, unsafe_allow_html=True)

    vars_acp = ["heures_sommeil", "heures_sport", "stress", "qualite_alimentation",
                "heures_ecran_loisirs", "vie_sociale", "heures_etude_semaine",
                "revenu_mensuel", "moyenne_academique", "score_bienetre", "satisfaction_generale"]

    col1, col2 = st.columns(2)
    with col1:
        n_comp = st.slider("Nombre de composantes à calculer", 2, min(len(vars_acp), 10), 6)
    with col2:
        color_var = st.selectbox("Colorer par", ["filiere", "genre", "niveau", "logement"])

    df_acp = df[vars_acp].dropna().reset_index(drop=True)
    color_vals = df[vars_acp + [color_var]].dropna().reset_index(drop=True)[color_var]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_acp)

    pca = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100

    mc = st.columns(3)
    for col_m, val, label, color in [
        (mc[0], f"{explained[0]:.1f}%", "Variance PC1", "#7ee787"),
        (mc[1], f"{explained[1]:.1f}%", "Variance PC2", "#58a6ff"),
        (mc[2], f"{sum(explained[:2]):.1f}%", "Variance 2D totale", "#bc8cff"),
    ]:
        with col_m:
            st.markdown(f"""
            <div class="metric-card" style='margin-top:0.5rem;'>
              <div class="metric-value" style='color:{color};'>{val}</div>
              <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🗺️ Projection 2D", "🌐 Projection 3D", "📊 Variance expliquée"])

    with tab1:
        df_coords = pd.DataFrame(coords[:, :2], columns=["PC1", "PC2"])
        df_coords[color_var] = color_vals.values
        fig_2d = px.scatter(df_coords, x="PC1", y="PC2", color=color_var,
                            title=f"Projection ACP 2D (PC1={explained[0]:.1f}%, PC2={explained[1]:.1f}%)",
                            template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Set2,
                            opacity=0.7)
        fig_2d.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#e6edf3")
        st.plotly_chart(fig_2d, use_container_width=True)

        # Cercle des corrélations
        loadings = pca.components_[:2].T
        fig_circle = go.Figure()
        theta = np.linspace(0, 2 * np.pi, 100)
        fig_circle.add_trace(go.Scatter(x=np.cos(theta), y=np.sin(theta), mode="lines",
                                        line=dict(color="#30363d"), showlegend=False))
        for i, v in enumerate(vars_acp):
            fig_circle.add_annotation(x=loadings[i, 0], y=loadings[i, 1], text=v,
                                      showarrow=True, arrowhead=2, arrowcolor="#2ea043",
                                      font=dict(color="#e6edf3", size=9))
        fig_circle.update_layout(title="Cercle des corrélations", template="plotly_dark",
                                 paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                                 font_color="#e6edf3", xaxis_range=[-1.2, 1.2], yaxis_range=[-1.2, 1.2],
                                 xaxis_title="PC1", yaxis_title="PC2")
        st.plotly_chart(fig_circle, use_container_width=True)

    with tab2:
        if n_comp >= 3:
            df_3d = pd.DataFrame(coords[:, :3], columns=["PC1", "PC2", "PC3"])
            df_3d[color_var] = color_vals.values
            fig_3d = px.scatter_3d(df_3d, x="PC1", y="PC2", z="PC3", color=color_var,
                                   template="plotly_dark",
                                   color_discrete_sequence=px.colors.qualitative.Set2,
                                   opacity=0.7,
                                   title="Projection ACP 3D")
            fig_3d.update_layout(paper_bgcolor="#161b22", font_color="#e6edf3", height=550)
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("Augmentez le nombre de composantes à ≥ 3 pour la projection 3D.")

    with tab3:
        cumulative = np.cumsum(explained)
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(n_comp)], y=explained,
                                 name="Individuelle", marker_color="#2ea043"))
        fig_var.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(n_comp)], y=cumulative,
                                     name="Cumulée", line=dict(color="#58a6ff", width=2),
                                     mode="lines+markers"))
        fig_var.add_hline(y=80, line_dash="dash", line_color="#e3b341",
                          annotation_text="Seuil 80%", annotation_font_color="#e3b341")
        fig_var.update_layout(title="Variance expliquée par composante", template="plotly_dark",
                              paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                              font_color="#e6edf3", yaxis_title="Variance (%)")
        st.plotly_chart(fig_var, use_container_width=True)

        # Tableau des loadings
        loadings_df = pd.DataFrame(pca.components_[:n_comp].T,
                                   columns=[f"PC{i+1}" for i in range(n_comp)],
                                   index=vars_acp).round(3)
        st.markdown('<div class="section-title">Matrice des loadings</div>', unsafe_allow_html=True)
        st.dataframe(loadings_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : CLASSIFICATION SUPERVISÉE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Classification supervisée":
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    st.markdown("""
    <div class="vitascope-header">
      <p class="vitascope-title">🎯 Classification Supervisée</p>
      <p class="vitascope-subtitle">Prédiction de la catégorie de bien-être à partir des habitudes de vie</p>
    </div>
    """, unsafe_allow_html=True)

    # Créer la variable cible ordinale
    df_cl = df.copy()
    df_cl["categorie_bienetre"] = pd.cut(df_cl["score_bienetre"],
                                          bins=[0, 4, 6, 8, 10],
                                          labels=["Faible", "Moyen", "Bon", "Excellent"])

    vars_features = ["heures_sommeil", "heures_sport", "stress", "qualite_alimentation",
                     "heures_ecran_loisirs", "vie_sociale", "heures_etude_semaine",
                     "revenu_mensuel", "age", "satisfaction_generale"]

    col1, col2 = st.columns(2)
    with col1:
        features = st.multiselect("Variables prédicteurs", vars_features, default=vars_features[:7])
    with col2:
        algo = st.selectbox("Algorithme", ["Random Forest", "Gradient Boosting", "SVM (RBF)", "K-Nearest Neighbors"])

    df_ml = df_cl[features + ["categorie_bienetre"]].dropna()
    X = df_ml[features].values
    y = df_ml["categorie_bienetre"].astype(str).values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2,
                                                          random_state=42, stratify=y_enc)

    if algo == "Random Forest":
        n_est = st.slider("Nombre d'arbres", 10, 300, 100, 10)
        clf = RandomForestClassifier(n_estimators=n_est, random_state=42)
    elif algo == "Gradient Boosting":
        lr = st.slider("Learning rate", 0.01, 0.5, 0.1, 0.01)
        clf = GradientBoostingClassifier(learning_rate=lr, random_state=42)
    elif algo == "SVM (RBF)":
        C = st.slider("Paramètre C", 0.1, 10.0, 1.0, 0.1)
        clf = SVC(C=C, kernel="rbf", random_state=42)
    else:
        k = st.slider("Nombre de voisins k", 3, 20, 5)
        clf = KNeighborsClassifier(n_neighbors=k)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(clf, X_scaled, y_enc, cv=5)

    mc = st.columns(3)
    for col_m, val, label, color in [
        (mc[0], f"{accuracy*100:.1f}%", "Accuracy (test)", "#7ee787"),
        (mc[1], f"{cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%", "CV Accuracy (5-fold)", "#58a6ff"),
        (mc[2], len(le.classes_), "Classes", "#e3b341"),
    ]:
        with col_m:
            st.markdown(f"""
            <div class="metric-card" style='margin-top:0.5rem;'>
              <div class="metric-value" style='color:{color};font-size:1.3rem;'>{val}</div>
              <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        class_labels = le.inverse_transform(sorted(set(y_enc)))
        fig_cm = px.imshow(cm, text_auto=True, x=class_labels, y=class_labels,
                           title="Matrice de confusion", template="plotly_dark",
                           color_continuous_scale="Greens")
        fig_cm.update_layout(paper_bgcolor="#161b22", font_color="#e6edf3")
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_right:
        # Importance des variables (si disponible)
        if hasattr(clf, "feature_importances_"):
            imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=True)
            fig_imp = go.Figure(go.Bar(y=imp.index, x=imp.values, orientation="h",
                                       marker_color="#2ea043"))
            fig_imp.update_layout(title="Importance des variables", template="plotly_dark",
                                  paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                                  font_color="#e6edf3")
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            # Pour SVM/KNN
            cv_bar = pd.DataFrame({"Fold": [f"Fold {i+1}" for i in range(5)],
                                   "Score": cv_scores})
            fig_cv = px.bar(cv_bar, x="Fold", y="Score", title="Scores par fold (CV-5)",
                            template="plotly_dark", color_discrete_sequence=["#58a6ff"])
            fig_cv.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#e6edf3")
            st.plotly_chart(fig_cv, use_container_width=True)

    # Rapport de classification
    st.markdown('<div class="section-title">Rapport de classification détaillé</div>', unsafe_allow_html=True)
    report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
    report_df = pd.DataFrame(report).T.round(3)
    # Garder seulement les colonnes numériques pour l'affichage propre
    report_df = report_df.fillna(0)
    st.dataframe(report_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : CLUSTERING NON SUPERVISÉ
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔵 Clustering":
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, davies_bouldin_score

    st.markdown("""
    <div class="vitascope-header">
      <p class="vitascope-title">🔵 Clustering Non Supervisé</p>
      <p class="vitascope-subtitle">Identification automatique de profils types d'étudiants</p>
    </div>
    """, unsafe_allow_html=True)

    vars_cluster = ["heures_sommeil", "heures_sport", "stress", "qualite_alimentation",
                    "heures_ecran_loisirs", "vie_sociale", "heures_etude_semaine",
                    "moyenne_academique", "score_bienetre"]

    col1, col2, col3 = st.columns(3)
    with col1:
        algo_cl = st.selectbox("Algorithme", ["K-Means", "Clustering Agglomératif", "DBSCAN"])
    with col2:
        features_cl = st.multiselect("Variables", vars_cluster, default=vars_cluster[:7])
    with col3:
        if algo_cl in ["K-Means", "Clustering Agglomératif"]:
            k = st.slider("Nombre de clusters (k)", 2, 8, 4)

    df_cl = df[features_cl].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cl)

    if algo_cl == "K-Means":
        modele_cl = KMeans(n_clusters=k, random_state=42, n_init=10)
    elif algo_cl == "Clustering Agglomératif":
        linkage = st.selectbox("Linkage", ["ward", "complete", "average"])
        modele_cl = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    else:
        eps = st.slider("eps (rayon de voisinage)", 0.1, 3.0, 0.8, 0.1)
        min_s = st.slider("min_samples", 2, 20, 5)
        modele_cl = DBSCAN(eps=eps, min_samples=min_s)

    labels = modele_cl.fit_predict(X_scaled)
    df_cl["Cluster"] = [f"Cluster {int(l)}" if l >= 0 else "Bruit" for l in labels]

    mask_valide = labels >= 0
    n_clusters = len(set(labels[mask_valide].tolist()))
    try:
        X_valide = X_scaled[mask_valide]
        labels_valide = labels[mask_valide]
        sil = float(silhouette_score(X_valide, labels_valide)) if n_clusters > 1 and len(X_valide) > 1 else 0.0
        db = float(davies_bouldin_score(X_valide, labels_valide)) if n_clusters > 1 and len(X_valide) > 1 else 0.0
    except Exception:
        sil, db = 0.0, 0.0

    mc = st.columns(4)
    for col_m, val, label, color in [
        (mc[0], n_clusters, "Clusters trouvés", "#7ee787"),
        (mc[1], f"{sil:.3f}", "Score Silhouette (↑)", "#58a6ff"),
        (mc[2], f"{db:.3f}", "Davies-Bouldin (↓)", "#e3b341"),
        (mc[3], int((labels == -1).sum()), "Points bruit", "#bc8cff"),
    ]:
        with col_m:
            st.markdown(f"""
            <div class="metric-card" style='margin-top:0.5rem;'>
              <div class="metric-value" style='color:{color};'>{val}</div>
              <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    # Projection ACP 2D pour visualisation
    pca2 = PCA(n_components=2, random_state=42)
    coords2d = pca2.fit_transform(X_scaled)
    df_viz = pd.DataFrame(coords2d, columns=["PC1", "PC2"])
    df_viz["Cluster"] = df_cl["Cluster"].values

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_cl = px.scatter(df_viz, x="PC1", y="PC2", color="Cluster",
                            title="Clusters projetés en 2D (ACP)",
                            template="plotly_dark",
                            color_discrete_sequence=px.colors.qualitative.Set2,
                            opacity=0.8)
        fig_cl.update_layout(paper_bgcolor="#161b22", plot_bgcolor="#0d1117", font_color="#e6edf3")
        st.plotly_chart(fig_cl, use_container_width=True)

    with col_g2:
        # Profil moyen par cluster
        profils = df_cl.groupby("Cluster")[features_cl].mean().round(2)
        fig_radar = go.Figure()
        colors_cl = px.colors.qualitative.Set2

        for i, cl in enumerate(profils.index):
            vals = profils.loc[cl].tolist()
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=features_cl + [features_cl[0]],
                fill="toself",
                name=cl,
                line=dict(color=colors_cl[i % len(colors_cl)]),
                opacity=0.6,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, color="#8b949e"),
                       bgcolor="#0d1117"),
            template="plotly_dark",
            paper_bgcolor="#161b22",
            font_color="#e6edf3",
            title="Profil moyen par cluster",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Méthode du coude (K-Means uniquement)
    if algo_cl == "K-Means":
        st.markdown('<div class="section-title">Méthode du coude (sélection optimale de k)</div>',
                    unsafe_allow_html=True)
        inertias = []
        k_range = range(2, 11)
        for ki in k_range:
            km_i = KMeans(n_clusters=ki, random_state=42, n_init=10)
            km_i.fit(X_scaled)
            inertias.append(km_i.inertia_)

        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=list(k_range), y=inertias, mode="lines+markers",
                                       line=dict(color="#2ea043", width=2),
                                       marker=dict(color="#7ee787", size=8)))
        fig_elbow.add_vline(x=k, line_dash="dash", line_color="#e3b341",
                             annotation_text=f"k={k} sélectionné", annotation_font_color="#e3b341")
        fig_elbow.update_layout(title="Inertie en fonction de k", template="plotly_dark",
                                paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                                font_color="#e6edf3", xaxis_title="k", yaxis_title="Inertie")
        st.plotly_chart(fig_elbow, use_container_width=True)

    # Tableau des profils
    st.markdown('<div class="section-title">Profils moyens par cluster</div>', unsafe_allow_html=True)
    profils_display = df_cl.groupby("Cluster")[features_cl].mean().round(2)
    counts = df_cl["Cluster"].value_counts()
    profils_display.insert(0, "N", profils_display.index.map(counts).fillna(0).astype(int))
    st.dataframe(profils_display, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE : EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📤 Export des résultats":
    st.markdown("""
    <div class="vitascope-header">
      <p class="vitascope-title">📤 Export des Résultats</p>
      <p class="vitascope-subtitle">Téléchargez vos données et analyses</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Données brutes", "📈 Statistiques", "📋 Rapport complet"])

    with tab1:
        st.markdown('<div class="section-title">Aperçu des données</div>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            csv_data = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="⬇️ Télécharger CSV complet",
                data=csv_data,
                file_name=f"vitascope_donnees_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            # Données réelles uniquement
            df_reelles = df[df["source"] == "Réelle"] if "source" in df.columns else df
            csv_reel = df_reelles.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                label="⬇️ Données réelles uniquement",
                data=csv_reel,
                file_name=f"vitascope_donnees_reelles_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab2:
        vars_stats = ["heures_sommeil", "heures_sport", "stress", "qualite_alimentation",
                      "heures_ecran_loisirs", "vie_sociale", "heures_etude_semaine",
                      "revenu_mensuel", "moyenne_academique", "score_bienetre"]
        stats_export = df[vars_stats].describe().T.round(3)
        stats_export.columns = ["N", "Moyenne", "Ecart_type", "Min", "Q1", "Mediane", "Q3", "Max"]
        st.dataframe(stats_export, use_container_width=True)

        stats_csv = stats_export.to_csv(encoding="utf-8-sig")
        st.download_button(
            label="⬇️ Télécharger statistiques CSV",
            data=stats_csv,
            file_name=f"vitascope_stats_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab3:
        # Rapport texte
        rapport = f"""
=======================================================
  VITASCOPE — Rapport d'Analyse
  Observatoire du Bien-Être Étudiant
  Généré le : {datetime.now().strftime('%d/%m/%Y à %H:%M')}
=======================================================

1. CONTEXTE
-----------
Application développée dans le cadre du cours INF 232 EC2
Master Informatique — 2025-2026

2. DESCRIPTION DU JEU DE DONNÉES
----------------------------------
Nombre total d'observations : {len(df)}
Dont données réelles         : {len(df[df['source']=='Réelle']) if 'source' in df.columns else 'N/A'}
Dont données démo            : {len(df[df['source']=='Démo']) if 'source' in df.columns else 'N/A'}
Nombre de variables          : {len(df.columns) - 2}
Mode démo actif              : {'Oui' if mode_demo else 'Non'}

3. STATISTIQUES DESCRIPTIVES CLÉS
-----------------------------------
Moyenne académique moyenne   : {df['moyenne_academique'].mean():.2f} / 20
Score de bien-être moyen     : {df['score_bienetre'].mean():.2f} / 10
Stress moyen                 : {df['stress'].mean():.2f} / 10
Heures de sommeil moyenne    : {df['heures_sommeil'].mean():.2f} h/nuit
Heures de sport moyenne      : {df['heures_sport'].mean():.2f} h/semaine
Heures d'étude moyenne       : {df['heures_etude_semaine'].mean():.2f} h/semaine

4. CORRÉLATIONS IMPORTANTES (avec moyenne académique)
------------------------------------------------------
"""
        for v in ["heures_sommeil", "stress", "heures_etude_semaine", "qualite_alimentation",
                  "heures_sport", "heures_ecran_loisirs", "vie_sociale"]:
            corr = df[[v, "moyenne_academique"]].dropna().corr().iloc[0, 1]
            rapport += f"  - {v:<30} r = {corr:.3f}\n"

        rapport += f"""
5. MODULES D'ANALYSE DISPONIBLES
----------------------------------
  ✓ Régression Linéaire Simple (avec choix de variables)
  ✓ Régression Linéaire Multiple (OLS, Ridge, Lasso)
  ✓ Analyse en Composantes Principales (2D, 3D, cercle des corrélations)
  ✓ Classification Supervisée (Random Forest, SVM, Gradient Boosting, KNN)
  ✓ Clustering K-Means (avec méthode du coude, radar des profils)

=======================================================
  VitaScope v1.0 — INF 232 EC2
=======================================================
"""
        st.text_area("Rapport d'analyse", rapport, height=400)
        st.download_button(
            label="⬇️ Télécharger le rapport (.txt)",
            data=rapport.encode("utf-8"),
            file_name=f"vitascope_rapport_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True,
        )
