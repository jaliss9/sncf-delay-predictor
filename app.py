import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

@st.cache_resource
def load_model():
    model = pickle.load(open('data/processed/model.pkl', 'rb'))
    le_liaison = pickle.load(open('data/processed/le_liaison.pkl', 'rb'))
    le_service = pickle.load(open('data/processed/le_service.pkl', 'rb'))
    model_info = pickle.load(open('data/processed/model_info.pkl', 'rb'))
    return model, le_liaison, le_service, model_info

@st.cache_data
def load_data():
    return pd.read_csv('data/processed/data_ml.csv')

model, le_liaison, le_service, model_info = load_model()
df_ml = load_data()

lang = st.sidebar.selectbox("🌐 Language / Langue", ["English", "Français"])

texts = {
    "English": {
        "title": "🚄 TGV Punctuality Predictor",
        "subtitle": "Predict the probability that a TGV train will be on time for a given route.",
        "select_route": "Select a route",
        "service_type": "Service type",
        "year": "Year",
        "month": "Month",
        "predict_btn": "Predict punctuality",
        "result": "Result",
        "predicted_rate": "Predicted punctuality rate",
        "avg_duration": "Average duration",
        "trains_month": "Trains/month",
        "excellent": "✅ Excellent punctuality expected",
        "good": "ℹ️ Good punctuality expected",
        "average": "⚠️ Average punctuality - plan a buffer",
        "poor": "❌ High risk of delay",
        "footer": "Data: SNCF Open Data | Model: Random Forest | Jalïss",
        "historical": "Historical Punctuality",
        "monthly_trend": "Monthly Trend",
        "comparison": "Route Comparison",
        "top_routes": "Top 10 Most Punctual Routes",
        "bottom_routes": "Top 10 Least Punctual Routes"
    },
    "Français": {
        "title": "🚄 Prédicteur de Régularité TGV",
        "subtitle": "Prédisez la probabilité qu'un TGV soit à l'heure sur une liaison donnée.",
        "select_route": "Choisissez une liaison",
        "service_type": "Type de service",
        "year": "Année",
        "month": "Mois",
        "predict_btn": "Prédire la régularité",
        "result": "Résultat",
        "predicted_rate": "Taux de régularité prédit",
        "avg_duration": "Durée moyenne",
        "trains_month": "Trains/mois",
        "excellent": "✅ Excellente ponctualité attendue",
        "good": "ℹ️ Bonne ponctualité attendue",
        "average": "⚠️ Ponctualité moyenne - prévoyez une marge",
        "poor": "❌ Risque élevé de retard",
        "footer": "Données : SNCF Open Data | Modèle : Random Forest | Par Jal pour Alstom Singapore",
        "historical": "Historique de ponctualité",
        "monthly_trend": "Tendance mensuelle",
        "comparison": "Comparaison des liaisons",
        "top_routes": "Top 10 liaisons les plus ponctuelles",
        "bottom_routes": "Top 10 liaisons les moins ponctuelles"
    }
}

t = texts[lang]

st.title(t["title"])
st.write(t["subtitle"])

liaisons = sorted(model_info['liaisons'])
liaison = st.selectbox(t["select_route"], liaisons)

service = st.selectbox(t["service_type"], ['National', 'International'])

col1, col2 = st.columns(2)
with col1:
    annee = st.selectbox(t["year"], [2024, 2025, 2026])
with col2:
    mois = st.selectbox(t["month"], list(range(1, 13)))

liaison_data = df_ml[df_ml['liaison'] == liaison]

if len(liaison_data) > 0:
    duree_moyenne = liaison_data['Durée moyenne du trajet'].mean()
    circulations_moyennes = liaison_data['Nombre de circulations prévues'].mean()
else:
    duree_moyenne = df_ml['Durée moyenne du trajet'].mean()
    circulations_moyennes = df_ml['Nombre de circulations prévues'].mean()

if st.button(t["predict_btn"], type="primary"):
    liaison_encoded = le_liaison.transform([liaison])[0]
    service_encoded = le_service.transform([service])[0]
    
    features = [[
        liaison_encoded,
        service_encoded,
        annee,
        mois,
        duree_moyenne,
        circulations_moyennes
    ]]
    
    prediction = model.predict(features)[0]
    prediction_pct = prediction * 100
    
    st.markdown("---")
    st.subheader(t["result"])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(t["predicted_rate"], f"{prediction_pct:.1f}%")
    with col2:
        st.metric(t["avg_duration"], f"{duree_moyenne:.0f} min")
    with col3:
        st.metric(t["trains_month"], f"{circulations_moyennes:.0f}")
    
    if prediction_pct >= 90:
        st.success(t["excellent"])
    elif prediction_pct >= 80:
        st.info(t["good"])
    elif prediction_pct >= 70:
        st.warning(t["average"])
    else:
        st.error(t["poor"])
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00cc96" if prediction_pct >= 80 else "#ffa15a" if prediction_pct >= 70 else "#ef553b"},
            'steps': [
                {'range': [0, 70], 'color': "rgba(239,85,59,0.2)"},
                {'range': [70, 80], 'color': "rgba(255,161,90,0.2)"},
                {'range': [80, 100], 'color': "rgba(0,204,150,0.2)"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': prediction_pct
            }
        },
        title={'text': t["predicted_rate"]}
    ))
    fig_gauge.update_layout(height=300, margin=dict(t=50, b=0, l=0, r=0))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")
st.subheader(f"📊 {t['historical']} - {liaison}")

if len(liaison_data) > 0:
    liaison_data_sorted = liaison_data.sort_values(['annee', 'mois'])
    liaison_data_sorted['date'] = pd.to_datetime(liaison_data_sorted['annee'].astype(str) + '-' + liaison_data_sorted['mois'].astype(str) + '-01')
    
    fig_trend = px.line(
        liaison_data_sorted, 
        x='date', 
        y='taux_regularite',
        title=t["monthly_trend"],
        labels={'date': '', 'taux_regularite': 'Punctuality Rate' if lang == "English" else 'Taux de régularité'}
    )
    fig_trend.update_traces(line_color='#00cc96')
    fig_trend.update_layout(yaxis_tickformat='.0%', hovermode='x unified')
    st.plotly_chart(fig_trend, use_container_width=True)
    
    monthly_avg = liaison_data.groupby('mois')['taux_regularite'].mean().reset_index()
    month_names_en = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_names_fr = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun', 'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc']
    monthly_avg['month_name'] = monthly_avg['mois'].apply(lambda x: month_names_en[x-1] if lang == "English" else month_names_fr[x-1])
    
    fig_monthly = px.bar(
        monthly_avg,
        x='month_name',
        y='taux_regularite',
        title='Punctuality by Month' if lang == "English" else 'Ponctualité par mois',
        labels={'month_name': '', 'taux_regularite': ''}
    )
    fig_monthly.update_traces(marker_color='#636efa')
    fig_monthly.update_layout(yaxis_tickformat='.0%')
    st.plotly_chart(fig_monthly, use_container_width=True)
else:
    st.info("No historical data available for this route." if lang == "English" else "Pas de données historiques disponibles pour cette liaison.")

st.markdown("---")
st.subheader(f"📈 {t['comparison']}")

col1, col2 = st.columns(2)

with col1:
    top_routes = df_ml.groupby('liaison')['taux_regularite'].mean().sort_values(ascending=False).head(10).reset_index()
    fig_top = px.bar(
        top_routes,
        x='taux_regularite',
        y='liaison',
        orientation='h',
        title=t["top_routes"],
        labels={'taux_regularite': '', 'liaison': ''}
    )
    fig_top.update_traces(marker_color='#00cc96')
    fig_top.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_tickformat='.0%', height=400)
    st.plotly_chart(fig_top, use_container_width=True)

with col2:
    bottom_routes = df_ml.groupby('liaison')['taux_regularite'].mean().sort_values(ascending=True).head(10).reset_index()
    fig_bottom = px.bar(
        bottom_routes,
        x='taux_regularite',
        y='liaison',
        orientation='h',
        title=t["bottom_routes"],
        labels={'taux_regularite': '', 'liaison': ''}
    )
    fig_bottom.update_traces(marker_color='#ef553b')
    fig_bottom.update_layout(yaxis={'categoryorder': 'total descending'}, xaxis_tickformat='.0%', height=400)
    st.plotly_chart(fig_bottom, use_container_width=True)

st.markdown("---")
st.caption(t["footer"])
