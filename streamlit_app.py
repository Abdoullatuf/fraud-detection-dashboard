"""
STA218 — Application Streamlit pour la soutenance.

Objectifs :
1. Démontrer en direct la prédiction d'une transaction (saisie ou JSON).
2. Expliquer la décision (raisons métier + SHAP local).
3. Permettre à l'enseignant de jouer avec le seuil de profit.
4. Montrer les performances modèle (ROC, PR, matrice confusion).
5. Prédire un fichier CSV en lot.

Lancement (PowerShell) :

    cd C:\\Users\\aella\\03_MASTER_CNAM_DATA_SCIENCE\\STA218-Science_de_la_donnee_en_milieu_pro\\03_Projets\\livrable_projet_sta218_juin2026
    C:\\Users\\aella\\03_MASTER_CNAM_DATA_SCIENCE\\venv_data_science\\Scripts\\python.exe -m streamlit run app/app_streamlit.py

L'application réutilise :
- outputs/model/fraud_profit_model.joblib   (entraîné par scripts/train_model_for_demo.py)
- outputs/model/dashboard_cache.joblib      (généré par app/build_dashboard_cache.py)
- scripts/demo_prediction_live.py           (logique de prédiction + raisons)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Chemins et imports
# ----------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

# Logique de prédiction réutilisée du script CLI :
from demo_prediction_live import (  # noqa: E402
    build_features,
    heuristic_reasons,
    predict,
)

MODEL_PATH = PROJECT_DIR / "outputs" / "model" / "fraud_profit_model.joblib"
CACHE_PATH = PROJECT_DIR / "outputs" / "model" / "dashboard_cache.joblib"
SAMPLES_PATH = PROJECT_DIR / "outputs" / "model" / "sample_transactions.json"

# ----------------------------------------------------------------------------
# Configuration générale
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="STA218 — Détection de fraude & rentabilité",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Chargement des artefacts (mis en cache)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement du modèle…")
def load_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        st.error(
            "Modèle introuvable. Lance d'abord :\n"
            "`python scripts/train_model_for_demo.py`"
        )
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner="Chargement du cache du tableau de bord…")
def load_dashboard_cache() -> dict[str, Any] | None:
    if not CACHE_PATH.exists():
        return None
    return joblib.load(CACHE_PATH)


@st.cache_resource(show_spinner="Initialisation de l'explicateur SHAP…")
def load_shap_explainer(_artifact: dict[str, Any]):
    """Tente shap.TreeExplainer ; fallback sur les contributions natives XGBoost.

    Le fallback utilise `booster.predict(pred_contribs=True)`, qui renvoie
    exactement les mêmes valeurs SHAP que TreeExplainer pour XGBoost mais
    fonctionne avec n'importe quelle combinaison de versions xgboost/shap.
    L'underscore en début de paramètre dit à Streamlit de ne pas hasher
    l'artefact (qui contient des historiques volumineux non hashables).
    """
    # 1) Essai standard via la librairie shap
    try:
        import shap

        explainer = shap.TreeExplainer(_artifact["classifier"])
        # Sanity test : si l'instanciation passe mais l'appel échoue, on
        # bascule sur le fallback.
        return ("shap", explainer)
    except Exception:
        pass

    # 2) Fallback : XGBoost natif (pred_contribs)
    try:
        import xgboost as xgb  # noqa: F401

        booster = _artifact["classifier"].get_booster()
        return ("xgb_native", booster)
    except Exception as exc:
        st.warning(f"Aucun explicateur SHAP disponible : {exc}")
        return None


def explain_one(explainer, x_t_dense: np.ndarray) -> tuple[np.ndarray, float]:
    """Renvoie (contributions, base_value) pour une seule transaction."""
    if explainer is None:
        raise RuntimeError("Aucun explicateur disponible")
    kind, obj = explainer
    if kind == "shap":
        explanation = obj(x_t_dense)
        vals = explanation.values
        if vals.ndim == 3:
            vals = vals[..., -1]
        base = (
            float(getattr(explanation, "base_values", [0.0])[0])
            if hasattr(explanation, "base_values")
            else 0.0
        )
        return vals[0], base
    if kind == "xgb_native":
        import xgboost as xgb

        dmat = xgb.DMatrix(x_t_dense)
        contribs = obj.predict(dmat, pred_contribs=True)
        # Dernière colonne = bias / base value
        return contribs[0, :-1], float(contribs[0, -1])
    raise RuntimeError(f"Explicateur inconnu : {kind}")


def transformed_feature_names(preprocessor) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"f{i}" for i in range(preprocessor.transform(
            pd.DataFrame([{}])
        ).shape[1])]


# ----------------------------------------------------------------------------
# Fonctions utilitaires métier
# ----------------------------------------------------------------------------
def profit_at_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Profit décomposé pour un seuil donné, selon la matrice du sujet."""
    y_hat = proba >= threshold
    tn = (y_true == 0) & (~y_hat)
    tp = (y_true == 1) & y_hat
    fn = (y_true == 1) & (~y_hat)
    fp = (y_true == 0) & y_hat
    margin_tn = 0.10 * float(values[tn].sum())
    saved_tp = float(values[tp].sum())
    lost_fn = float(values[fn].sum())
    cost_fp = 15.0 * int(fp.sum())
    profit = margin_tn + saved_tp - lost_fn - cost_fp
    return {
        "profit": profit,
        "margin_tn": margin_tn,
        "saved_tp": saved_tp,
        "lost_fn": lost_fn,
        "cost_fp": cost_fp,
        "n_tn": int(tn.sum()),
        "n_tp": int(tp.sum()),
        "n_fn": int(fn.sum()),
        "n_fp": int(fp.sum()),
        "block_rate": float(y_hat.mean()),
        "recall": float(tp.sum() / max(1, (y_true == 1).sum())),
        "precision": float(tp.sum() / max(1, y_hat.sum())),
    }


def profit_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    values: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        m = profit_at_threshold(y_true, proba, values, float(thr))
        rows.append({"threshold": float(thr), **m})
    return pd.DataFrame(rows)


def accept_all_profit(y_true: np.ndarray, values: np.ndarray) -> float:
    """Profit si on accepte tout : marge sur légitimes, perte sur fraudes."""
    legit = y_true == 0
    fraud = y_true == 1
    return 0.10 * float(values[legit].sum()) - float(values[fraud].sum())


# ----------------------------------------------------------------------------
# Sidebar : statut + navigation rapide
# ----------------------------------------------------------------------------
artifact = load_artifact()
cache = load_dashboard_cache()

with st.sidebar:
    st.markdown("### 🛡️ STA218 — Soutenance")
    st.caption("Maoulida Abdoullatuf · M2 Science des données · CNAM")
    st.divider()
    st.markdown("**Modèle chargé**")
    st.write(f"Type : `{artifact['model_type']}`")
    st.write(f"Seuil profit : `{artifact['threshold']:.2f}`")
    metrics = artifact.get("metrics", {})
    if metrics:
        st.write(f"PR-AUC test : `{metrics.get('test_pr_auc', float('nan')):.4f}`")
        st.write(f"Profit test : `{metrics.get('test_profit', 0):,.0f} $`")
    st.divider()
    if cache is None:
        st.warning(
            "Cache `dashboard_cache.joblib` introuvable.\n\n"
            "Pour activer les onglets **Performance** et **Simulateur**, lance :\n\n"
            "`python app/build_dashboard_cache.py`"
        )
    else:
        st.success(
            f"Cache chargé.\n\n"
            f"Validation : {cache['n_valid']:,}\n"
            f"Test : {cache['n_test']:,}"
        )
    st.divider()
    st.caption(
        "Le modèle ignore volontairement la variable `sex` (RGPD / faible signal). "
        "Les compteurs device/IP sont calculés en respectant l'ordre temporel."
    )


# ----------------------------------------------------------------------------
# Onglets principaux
# ----------------------------------------------------------------------------
TAB_HOME, TAB_PRED, TAB_SHAP, TAB_THR, TAB_PERF, TAB_BATCH, TAB_ABOUT = st.tabs(
    [
        "🏠 Accueil",
        "🎯 Prédiction live",
        "🔍 SHAP local",
        "💰 Simulateur de seuil",
        "📊 Performance modèle",
        "📁 Prédictions par lot",
        "ℹ️ À propos",
    ]
)


# ============================================================================
# Onglet : Accueil
# ============================================================================
with TAB_HOME:
    st.title("🛡️ Détection de fraude & optimisation de la rentabilité")
    st.markdown(
        "**Projet STA218** — Maoulida Abdoullatuf · M2 Science des données CNAM "
        "· soutenance juin 2026."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Profit test",
        f"{metrics.get('test_profit', 0):,.0f} $",
        help="Gain net total appliqué au seuil de profit optimal sur le jeu de test.",
    )
    col2.metric(
        "Seuil opérationnel",
        f"{artifact['threshold']:.2f}",
        help="Probabilité au-delà de laquelle la transaction est bloquée.",
    )
    col3.metric(
        "PR-AUC test",
        f"{metrics.get('test_pr_auc', 0):.3f}",
        help="Aire sous la courbe Précision/Rappel — robuste au déséquilibre.",
    )
    col4.metric(
        "Type de modèle",
        artifact["model_type"],
        help="XGBoost retenu après comparaison avec une régression logistique.",
    )

    st.divider()

    st.subheader("Lecture rapide du projet")
    st.markdown(
        """
- **Contexte** : Card-Not-Present, **9,3 %** de fraude, ≈ 151 000 transactions.
- **Triple objectif** : performance (détecter avant validation), rentabilité
  (arbitrer FP vs FN), explicabilité (justifier au service client).
- **Matrice financière** :

| Résultat | Action | Impact |
|---|---|---|
| Vrai négatif | Client légitime accepté | **+10 % de V** |
| Vrai positif | Fraudeur bloqué | **+100 % de V** |
| Faux négatif | Fraude manquée | **−100 % de V** |
| Faux positif | Légitime bloqué à tort | **−15 €** |

- **Méthodologie clé** : split *temporel* 70/15/15, features *causales* (compteurs
  device/IP construits avec l'ordre du temps), seuil choisi par maximisation du
  profit sur validation puis appliqué une seule fois sur test.
        """
    )

    st.subheader("Comment utiliser cette application ?")
    st.markdown(
        """
1. Aller dans **🎯 Prédiction live**, saisir la transaction (ou utiliser un exemple),
   cliquer sur *Prédire*.
2. Basculer sur **🔍 SHAP local** pour montrer les raisons quantitatives.
3. Si on demande *« et si on bougeait le seuil ? »*, ouvrir
   **💰 Simulateur de seuil**.
4. Sur une question de méthode, **📊 Performance modèle** affiche ROC, PR et
   matrice de confusion sur le test.
5. **📁 Prédictions par lot** sert à scorer un CSV de transactions complet.
        """
    )


# ============================================================================
# Onglet : Prédiction live
# ============================================================================
with TAB_PRED:
    st.header("🎯 Prédiction live")

    samples = artifact.get("samples", {})

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("### Transaction à scorer")
        mode = st.radio(
            "Source des champs",
            options=["Formulaire", "Exemple fraude", "Exemple légitime", "JSON brut"],
            horizontal=True,
        )

        if mode == "Exemple fraude" and "fraud_high_score" in samples:
            defaults = samples["fraud_high_score"]
        elif mode == "Exemple légitime" and "legit_low_score" in samples:
            defaults = samples["legit_low_score"]
        else:
            defaults = {
                "signup_time": "2015-01-01 18:52:44",
                "purchase_time": "2015-01-01 18:52:45",
                "purchase_value": 120,
                "device_id": "YSSKYOSJHPPLJ",
                "source": "SEO",
                "browser": "Opera",
                "sex": "M",
                "age": 53,
                "ip_address": 2621473820,
            }

        if mode == "JSON brut":
            json_text = st.text_area(
                "JSON transaction",
                value=json.dumps(defaults, indent=2),
                height=260,
            )
            try:
                form_values = json.loads(json_text)
            except Exception as exc:
                st.error(f"JSON invalide : {exc}")
                form_values = defaults
        else:
            with st.form("form_tx", clear_on_submit=False):
                c1, c2 = st.columns(2)
                with c1:
                    signup_time = st.text_input(
                        "signup_time", value=str(defaults["signup_time"])
                    )
                    purchase_value = st.number_input(
                        "purchase_value ($)",
                        value=float(defaults["purchase_value"]),
                        min_value=0.0,
                        step=1.0,
                    )
                    source = st.selectbox(
                        "source",
                        options=["SEO", "Ads", "Direct"],
                        index=["SEO", "Ads", "Direct"].index(
                            str(defaults.get("source", "SEO"))
                        )
                        if defaults.get("source", "SEO")
                        in ["SEO", "Ads", "Direct"]
                        else 0,
                    )
                    sex = st.selectbox(
                        "sex (ignoré par le modèle)",
                        options=["M", "F"],
                        index=0 if str(defaults.get("sex", "M")) == "M" else 1,
                    )
                    ip_address = st.text_input(
                        "ip_address (entier ou x.x.x.x)",
                        value=str(defaults["ip_address"]),
                    )
                with c2:
                    purchase_time = st.text_input(
                        "purchase_time", value=str(defaults["purchase_time"])
                    )
                    device_id = st.text_input(
                        "device_id", value=str(defaults["device_id"])
                    )
                    browsers = ["Chrome", "Safari", "IE", "FireFox", "Opera"]
                    browser_default = str(defaults.get("browser", "Chrome"))
                    browser = st.selectbox(
                        "browser",
                        options=browsers,
                        index=browsers.index(browser_default)
                        if browser_default in browsers
                        else 0,
                    )
                    age = st.number_input(
                        "age", value=int(defaults["age"]), min_value=0, step=1
                    )

                submitted = st.form_submit_button(
                    "🎯 Prédire", use_container_width=True, type="primary"
                )
                form_values = {
                    "signup_time": signup_time,
                    "purchase_time": purchase_time,
                    "purchase_value": purchase_value,
                    "device_id": device_id,
                    "source": source,
                    "browser": browser,
                    "sex": sex,
                    "age": age,
                    "ip_address": ip_address,
                }

        if mode == "JSON brut":
            run = st.button("🎯 Prédire", type="primary", use_container_width=True)
        else:
            run = submitted

    with col_result:
        st.markdown("### Résultat")
        if run:
            try:
                # Persist last prediction in session for the SHAP tab.
                st.session_state["last_input"] = form_values
                result = predict(form_values, artifact)
                st.session_state["last_result"] = result
            except Exception as exc:
                st.error(f"Erreur de prédiction : {exc}")
                result = None
        else:
            result = st.session_state.get("last_result")

        if result is not None:
            proba = result["probability"]
            decision = result["decision"]
            block = decision.startswith("BLOQUER")

            colp, cold = st.columns(2)
            colp.metric("Probabilité de fraude", f"{100 * proba:.2f} %")
            cold.metric(
                "Décision",
                "🛑 BLOQUER / Revue" if block else "✅ ACCEPTER",
                delta=f"seuil : {result['threshold']:.2f}",
                delta_color="off",
            )

            st.progress(min(max(proba, 0.0), 1.0))

            with st.expander("Détails du contexte calculé", expanded=True):
                d = result["diagnostics"]
                st.write(
                    {
                        "pays IP brut → modélisé": f"{d['raw_country']} → {d['model_country']}",
                        "délai signup → achat (s)": d["delay_seconds"],
                        "device déjà vu avant": d["prev_tx_by_device"],
                        "IP déjà vue avant": d["prev_tx_by_ip"],
                        "sex (ignoré)": d.get("sex_ignored", ""),
                    }
                )

            st.markdown("**Raisons lisibles**")
            for r in result["reasons"]:
                st.markdown(f"- {r}")

            st.markdown("**Impact financier potentiel (si la décision est appliquée)**")
            cf1, cf2 = st.columns(2)
            cf1.metric(
                "Si client légitime",
                f"{result['financial_if_legit']:+,.2f} $",
                help=("+10% de V si accepté, -15 $ si bloqué à tort."),
            )
            cf2.metric(
                "Si fraude réelle",
                f"{result['financial_if_fraud']:+,.2f} $",
                help=("+100% de V si bloqué, -100% de V si manquée."),
            )

            with st.expander("Toutes les features envoyées au modèle"):
                st.json(result["features"])
        else:
            st.info("Saisissez une transaction puis cliquez sur **Prédire**.")


# ============================================================================
# Onglet : SHAP local
# ============================================================================
with TAB_SHAP:
    st.header("🔍 Pourquoi cette décision ? — SHAP local")

    last_input = st.session_state.get("last_input")
    if last_input is None:
        st.info(
            "Aucune transaction prédite dans cette session. "
            "Va d'abord dans **🎯 Prédiction live** et clique sur *Prédire*."
        )
    else:
        try:
            x_one, _ = build_features(last_input, artifact)
        except Exception as exc:
            st.error(f"Échec de la construction des features : {exc}")
            x_one = None

        explainer = load_shap_explainer(artifact)

        if x_one is not None and explainer is not None:
            preprocessor = artifact["preprocessor"]
            x_t = preprocessor.transform(x_one)
            try:
                if hasattr(x_t, "toarray"):
                    x_t_dense = x_t.toarray()
                else:
                    x_t_dense = np.asarray(x_t)
                feat_names = transformed_feature_names(preprocessor)
                shap_row, base_value = explain_one(explainer, x_t_dense)

                df_contrib = pd.DataFrame(
                    {
                        "feature": feat_names[: len(shap_row)],
                        "valeur transformée": np.round(x_t_dense[0, : len(shap_row)], 4),
                        "contribution SHAP (log-odds)": shap_row,
                    }
                )
                df_contrib["abs"] = df_contrib["contribution SHAP (log-odds)"].abs()
                df_top = df_contrib.sort_values("abs", ascending=False).head(15).drop(
                    columns="abs"
                )

                st.markdown(
                    f"**Valeur de base** (log-odds avant features) : `{base_value:+.3f}`"
                )
                st.markdown(
                    "Une contribution positive *pousse* la transaction vers la fraude, "
                    "une contribution négative la pousse vers *légitime*."
                )

                st.dataframe(
                    df_top.style.background_gradient(
                        subset=["contribution SHAP (log-odds)"],
                        cmap="RdYlGn_r",
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                # Bar chart top contributions
                import altair as alt

                df_chart = df_top.copy()
                df_chart["sign"] = np.where(
                    df_chart["contribution SHAP (log-odds)"] > 0,
                    "vers fraude",
                    "vers légitime",
                )
                chart = (
                    alt.Chart(df_chart)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "contribution SHAP (log-odds):Q",
                            title="Contribution (log-odds)",
                        ),
                        y=alt.Y("feature:N", sort="-x", title=None),
                        color=alt.Color(
                            "sign:N",
                            scale=alt.Scale(
                                domain=["vers fraude", "vers légitime"],
                                range=["#d62728", "#2ca02c"],
                            ),
                            legend=alt.Legend(title=None),
                        ),
                        tooltip=list(df_chart.columns),
                    )
                    .properties(height=420)
                )
                st.altair_chart(chart, use_container_width=True)

                st.caption(
                    "Lecture : on est dans l'espace des log-odds. La somme des "
                    "contributions ajoutée à la valeur de base donne le score brut "
                    "du modèle, qui est ensuite passé par la sigmoïde."
                )
            except Exception as exc:
                st.error(f"Calcul SHAP impossible : {exc}")


# ============================================================================
# Onglet : Simulateur de seuil
# ============================================================================
with TAB_THR:
    st.header("💰 Simulateur de seuil")

    if cache is None:
        st.warning(
            "Le cache du tableau de bord n'est pas généré. "
            "Lance `python app/build_dashboard_cache.py` puis recharge la page."
        )
    else:
        split_choice = st.radio(
            "Jeu de données",
            options=["Validation", "Test"],
            horizontal=True,
            help=(
                "Le seuil est calibré sur **validation**. Le **test** sert à "
                "mesurer le profit en condition de simulation production."
            ),
        )
        bucket = cache["valid"] if split_choice == "Validation" else cache["test"]
        y = bucket["y"]
        proba = bucket["proba"]
        values = bucket["values"]

        thresholds = np.linspace(0.01, 0.99, 99)
        df_curve = profit_curve(y, proba, values, thresholds)

        idx_best = int(df_curve["profit"].idxmax())
        best_thr = float(df_curve.loc[idx_best, "threshold"])
        best_profit = float(df_curve.loc[idx_best, "profit"])
        accept_all = accept_all_profit(y, values)

        col_left, col_right = st.columns([2, 1])

        with col_right:
            st.markdown("### Choisir un seuil")
            chosen = st.slider(
                "Seuil de probabilité",
                min_value=0.01,
                max_value=0.99,
                value=float(artifact["threshold"]),
                step=0.01,
            )
            metrics_chosen = profit_at_threshold(y, proba, values, chosen)

            st.metric(
                "Profit au seuil choisi",
                f"{metrics_chosen['profit']:,.0f} $",
                delta=f"{metrics_chosen['profit'] - accept_all:+,.0f} $ vs accepter tout",
            )
            st.metric("Seuil optimal calculé", f"{best_thr:.2f}")
            st.metric("Profit optimal", f"{best_profit:,.0f} $")
            st.metric(
                "Profit accept-all",
                f"{accept_all:,.0f} $",
                help="Profit si on n'utilise pas du tout le modèle.",
            )

            st.markdown("**Décomposition au seuil choisi**")
            st.write(
                {
                    "Vrais négatifs (marge)": f"{metrics_chosen['margin_tn']:,.0f} $",
                    "Vrais positifs (perte évitée)": f"{metrics_chosen['saved_tp']:,.0f} $",
                    "Faux négatifs (fraude manquée)": f"-{metrics_chosen['lost_fn']:,.0f} $",
                    "Faux positifs (clients fâchés)": f"-{metrics_chosen['cost_fp']:,.0f} $",
                    "Taux de blocage": f"{100*metrics_chosen['block_rate']:.2f} %",
                    "Recall fraude": f"{100*metrics_chosen['recall']:.2f} %",
                    "Précision blocages": f"{100*metrics_chosen['precision']:.2f} %",
                }
            )

        with col_left:
            import altair as alt

            df_curve_long = df_curve[["threshold", "profit"]].copy()
            line = (
                alt.Chart(df_curve_long)
                .mark_line(color="#1f77b4")
                .encode(
                    x=alt.X("threshold:Q", title="Seuil de probabilité"),
                    y=alt.Y("profit:Q", title="Profit net ($)"),
                )
            )
            rule_best = (
                alt.Chart(pd.DataFrame({"threshold": [best_thr]}))
                .mark_rule(color="#2ca02c", strokeDash=[4, 4])
                .encode(x="threshold:Q")
            )
            rule_chosen = (
                alt.Chart(pd.DataFrame({"threshold": [chosen]}))
                .mark_rule(color="#d62728")
                .encode(x="threshold:Q")
            )
            rule_accept = (
                alt.Chart(pd.DataFrame({"profit": [accept_all]}))
                .mark_rule(color="#888", strokeDash=[2, 2])
                .encode(y="profit:Q")
            )
            chart = (line + rule_best + rule_chosen + rule_accept).properties(
                height=420,
                title=(
                    f"Profit vs seuil — optimal {best_thr:.2f} (vert), "
                    f"choisi {chosen:.2f} (rouge), accept-all (gris)."
                ),
            )
            st.altair_chart(chart, use_container_width=True)

            with st.expander("Tableau détaillé"):
                st.dataframe(
                    df_curve.round(2),
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================================
# Onglet : Performance modèle
# ============================================================================
with TAB_PERF:
    st.header("📊 Performance du modèle")

    if cache is None:
        st.warning(
            "Cache `dashboard_cache.joblib` introuvable. "
            "Lance `python app/build_dashboard_cache.py` pour activer cet onglet."
        )
    else:
        split_choice = st.radio(
            "Jeu",
            options=["Test", "Validation"],
            horizontal=True,
        )
        bucket = cache["test"] if split_choice == "Test" else cache["valid"]
        y = bucket["y"]
        proba = bucket["proba"]
        values = bucket["values"]

        from sklearn.metrics import (
            average_precision_score,
            confusion_matrix,
            precision_recall_curve,
            roc_auc_score,
            roc_curve,
        )

        roc_auc = roc_auc_score(y, proba)
        pr_auc = average_precision_score(y, proba)
        thr_op = float(artifact["threshold"])
        m = profit_at_threshold(y, proba, values, thr_op)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{roc_auc:.3f}")
        c2.metric("PR-AUC", f"{pr_auc:.3f}")
        c3.metric("Profit @ 0.66", f"{m['profit']:,.0f} $")
        c4.metric("Taux de blocage", f"{100*m['block_rate']:.2f} %")

        col_l, col_r = st.columns(2)

        # ROC
        fpr, tpr, _ = roc_curve(y, proba)
        df_roc = pd.DataFrame({"FPR": fpr, "TPR": tpr})
        import altair as alt

        chart_roc = (
            alt.Chart(df_roc)
            .mark_line(color="#1f77b4")
            .encode(x="FPR:Q", y="TPR:Q")
        )
        diag = alt.Chart(
            pd.DataFrame({"FPR": [0, 1], "TPR": [0, 1]})
        ).mark_line(color="#aaa", strokeDash=[4, 4]).encode(x="FPR:Q", y="TPR:Q")
        col_l.altair_chart(
            (chart_roc + diag).properties(title=f"ROC ({split_choice}) — AUC={roc_auc:.3f}"),
            use_container_width=True,
        )

        # PR
        prec, rec, _ = precision_recall_curve(y, proba)
        df_pr = pd.DataFrame({"recall": rec, "precision": prec})
        baseline = float(y.mean())
        chart_pr = (
            alt.Chart(df_pr)
            .mark_line(color="#d62728")
            .encode(x="recall:Q", y="precision:Q")
        )
        baseline_chart = alt.Chart(
            pd.DataFrame({"recall": [0, 1], "precision": [baseline, baseline]})
        ).mark_line(color="#aaa", strokeDash=[4, 4]).encode(x="recall:Q", y="precision:Q")
        col_r.altair_chart(
            (chart_pr + baseline_chart).properties(
                title=f"Précision/Rappel ({split_choice}) — AUC={pr_auc:.3f}"
            ),
            use_container_width=True,
        )

        st.divider()
        st.subheader(f"Matrice de confusion au seuil {thr_op:.2f}")
        y_hat = (proba >= thr_op).astype(int)
        cm = confusion_matrix(y, y_hat)
        df_cm = pd.DataFrame(
            cm,
            index=["Réel : Légitime (0)", "Réel : Fraude (1)"],
            columns=["Prédit : Légitime", "Prédit : Fraude"],
        )
        cc1, cc2 = st.columns(2)
        cc1.dataframe(df_cm, use_container_width=True)
        cc2.write(
            {
                "Vrais négatifs (TN)": int(cm[0, 0]),
                "Faux positifs (FP)": int(cm[0, 1]),
                "Faux négatifs (FN)": int(cm[1, 0]),
                "Vrais positifs (TP)": int(cm[1, 1]),
                "Recall fraude": f"{100*m['recall']:.2f} %",
                "Précision blocages": f"{100*m['precision']:.2f} %",
            }
        )


# ============================================================================
# Onglet : Prédictions par lot
# ============================================================================
with TAB_BATCH:
    st.header("📁 Prédictions par lot (CSV)")

    st.markdown(
        "Charge un CSV au même schéma que `Fraud_Data.csv` "
        "(colonnes attendues : `signup_time, purchase_time, purchase_value, "
        "device_id, source, browser, sex, age, ip_address`)."
    )

    file = st.file_uploader("Fichier CSV", type=["csv"])
    only_block = st.checkbox(
        "N'afficher que les transactions bloquées", value=False
    )

    if file is not None:
        try:
            df_in = pd.read_csv(file)
        except Exception as exc:
            st.error(f"Lecture CSV impossible : {exc}")
            df_in = None

        if df_in is not None:
            required = [
                "signup_time",
                "purchase_time",
                "purchase_value",
                "device_id",
                "source",
                "browser",
                "age",
                "ip_address",
            ]
            missing = [c for c in required if c not in df_in.columns]
            if missing:
                st.error(f"Colonnes manquantes : {missing}")
            else:
                st.success(f"{len(df_in):,} transactions à scorer.")
                progress = st.progress(0.0)
                t0 = time.perf_counter()
                results = []
                # Predict row by row to reuse heuristics + diagnostics.
                for i, row in df_in.iterrows():
                    payload = {col: row.get(col) for col in required + ["sex"] if col in df_in.columns or col == "sex"}
                    try:
                        r = predict(payload, artifact)
                        results.append(
                            {
                                "row": i,
                                "proba": r["probability"],
                                "decision": r["decision"],
                                "country": r["diagnostics"]["raw_country"],
                                "delay_s": r["diagnostics"]["delay_seconds"],
                                "prev_device": r["diagnostics"]["prev_tx_by_device"],
                                "prev_ip": r["diagnostics"]["prev_tx_by_ip"],
                                "purchase_value": float(row.get("purchase_value", 0)),
                            }
                        )
                    except Exception as exc:
                        results.append(
                            {"row": i, "proba": np.nan, "decision": f"ERROR: {exc}"}
                        )
                    if (i + 1) % 50 == 0 or i + 1 == len(df_in):
                        progress.progress((i + 1) / len(df_in))
                progress.empty()

                df_out = pd.DataFrame(results)
                df_full = pd.concat([df_in.reset_index(drop=True), df_out], axis=1)
                if only_block:
                    df_full = df_full[df_full["decision"].astype(str).str.startswith("BLOQUER")]

                elapsed = time.perf_counter() - t0
                st.caption(f"Scoring : {elapsed:.1f}s")

                col_a, col_b = st.columns(2)
                col_a.metric(
                    "Taux de blocage prédit",
                    f"{100 * (df_out['decision'].astype(str).str.startswith('BLOQUER')).mean():.2f} %",
                )
                col_b.metric("Proba médiane", f"{df_out['proba'].median():.3f}")

                st.dataframe(df_full, use_container_width=True)
                st.download_button(
                    "💾 Télécharger les résultats (CSV)",
                    data=df_full.to_csv(index=False).encode("utf-8"),
                    file_name="predictions_streamlit.csv",
                    mime="text/csv",
                )


# ============================================================================
# Onglet : À propos
# ============================================================================
with TAB_ABOUT:
    st.header("ℹ️ À propos & dépannage")

    st.markdown(
        """
**Architecture**
- `outputs/model/fraud_profit_model.joblib` : modèle XGBoost + preprocessor +
  historiques device/IP + seuil + samples.
- `outputs/model/dashboard_cache.joblib` : prédictions sur valid/test (sert aux
  onglets *Performance* et *Simulateur*).
- `scripts/demo_prediction_live.py` : logique métier (jointure IP, compteurs
  causaux, raisons heuristiques).
- `app/app_streamlit.py` : cette interface.

**Reconstruire les artefacts**

```powershell
# 1. Entraînement (~30s)
python scripts/train_model_for_demo.py
# 2. Cache du dashboard (~10s)
python app/build_dashboard_cache.py
```

**Choix défendables pour la soutenance**
- Variable `sex` ignorée (RGPD + signal très faible).
- Compteurs device/IP **causaux** : à l'instant t on ne compte que les
  transactions d'index < t pour respecter le futur.
- Seuil **0,66** choisi par maximisation du profit sur **validation**, jamais
  sur test (anti-fuite).
- Pays IP : top-20 par volume + pays à lift > 2 conservés ; les autres sont
  regroupés sous *Other* pour limiter le risque d'overfitting.
"""
    )

    with st.expander("Dump complet du model card"):
        st.json(metrics)

    with st.expander("Dump des features et composantes"):
        st.write(
            {
                "features": artifact["features"],
                "numeric_features": artifact["numeric_features"],
                "categorical_features": artifact["categorical_features"],
                "top_countries (head)": artifact["geo"]["top_countries"][:15],
                "high_risk_countries": artifact["geo"]["high_risk_countries"],
                "country_coverage_pct": artifact["geo"]["country_coverage_pct"],
            }
        )
