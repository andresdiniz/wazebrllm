import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import time
import mysql.connector
import pytz

# Compatibilidade com NumPy 2.x
if np.__version__.startswith('2.'):
    np.float_ = np.float64
    np.int_ = np.int_
    np.bool_ = np.bool_

# === CONFIGURAÇÕES INICIAIS ===
st.set_page_config(page_title="Análise de Rotas", layout="wide")
st.title("📊 Previsão de Velocidade e Análise de Anomalias")

# === CONEXÃO COM O BANCO DE DADOS ===
def get_data(start_date=None, end_date=None, route_id=None):
    try:
        mydb = mysql.connector.connect(
            host="185.213.81.52",
            user="u335174317_wazeportal",
            password="@Ndre2025.",
            database="u335174317_wazeportal"
        )
        mycursor = mydb.cursor()
        query = """
            SELECT hr.route_id, r.name AS route_name, hr.data, hr.velocidade
            FROM historic_routes hr
            JOIN routes r ON hr.route_id = r.id
        """
        conditions = []
        if start_date:
            conditions.append(f"hr.data >= '{start_date}'")
        if end_date:
            conditions.append(f"hr.data <= '{end_date}'")
        if route_id:
            conditions.append(f"hr.route_id = {route_id}")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY hr.data ASC"

        mycursor.execute(query)
        results = mycursor.fetchall()
        col_names = [desc[0] for desc in mycursor.description]
        df = pd.DataFrame(results, columns=col_names)
        mycursor.close()
        mydb.close()

        df['velocidade'] = pd.to_numeric(df['velocidade'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Erro ao conectar no banco: {str(e)}")
        st.stop()

# === LIMPEZA E FEATURE ENGINEERING ===
def clean_data(df, route):
    df = df[df['route_name'] == route].copy()
    df['data'] = pd.to_datetime(df['data']).dt.tz_localize(None)
    df = df.sort_values('data')
    df['velocidade'] = df['velocidade'].clip(upper=150).interpolate().ffill().bfill()
    df['day_of_week'] = df['data'].dt.day_name()
    df['hour'] = df['data'].dt.hour
    return df.dropna(subset=['velocidade'])

# === DETECÇÃO DE ANOMALIAS ===
def detect_anomalies(df):
    df = df.copy()
    df['vel_diff'] = df['velocidade'].diff().abs()
    threshold = df['vel_diff'].quantile(0.95) * 1.5
    return df[df['vel_diff'] > max(threshold, 20)]

# === PREVISÃO COM ARIMA ===
def create_arima_forecast(df, route_id):
    try:
        model = ARIMA(df['y'], order=(3, 1, 1))
        results = model.fit()
        forecast = results.get_forecast(steps=10)
        sao_paulo_tz = pytz.timezone('America/Sao_Paulo')
        future_timestamps = pd.date_range(start=pd.to_datetime(df['ds'].max()), periods=11, freq='3min', tz='America/Sao_Paulo')[1:]
        forecast_df = pd.DataFrame({
            'ds': future_timestamps,
            'yhat': forecast.predicted_mean,
            'yhat_lower': forecast.conf_int().iloc[:, 0],
            'yhat_upper': forecast.conf_int().iloc[:, 1],
            'id_route': route_id
        })
        return forecast_df
    except Exception as e:
        st.warning(f"Erro ARIMA: {e}. Usando previsão simples.")
        last_value = df['y'].iloc[-1]
        future_timestamps = pd.date_range(start=pd.to_datetime(df['ds'].max()), periods=11, freq='3min', tz='America/Sao_Paulo')[1:]
        return pd.DataFrame({
            'ds': future_timestamps,
            'yhat': [last_value]*10,
            'yhat_lower': [last_value - 5]*10,
            'yhat_upper': [last_value + 5]*10,
            'id_route': [route_id]*10
        })

# === SALVAR PREVISÃO NO BANCO ===
def save_forecast_to_db(forecast_df):
    try:
        engine = create_engine('mysql+mysqlconnector://u335174317_wazeportal:%40Ndre2025.@185.213.81.52/u335174317_wazeportal')
        with engine.begin() as conn:
            forecast_df.to_sql('forecast_history', con=conn, if_exists='append', index=False)
    except Exception as e:
        st.error(f"Erro ao salvar previsão: {e}")

# === INSIGHTS ===
def gerar_insights(df):
    insights = []
    insights.append(f"📌 Média de velocidade: **{df['velocidade'].mean():.2f} km/h**")
    dia_lento = df.groupby('data')['velocidade'].mean().idxmin()
    insights.append(f"🗓️ Dia mais lento: **{dia_lento.strftime('%Y-%m-%d')}**")
    insights.append(f"🗓️ Dia da semana mais lento: **{df.groupby('day_of_week')['velocidade'].mean().idxmin()}**")
    insights.append(f"🕒 Hora mais lenta: **{df.groupby('hour')['velocidade'].mean().idxmin()}:00**")
    return "\n\n".join(insights)

# === PLOTS ===
def seasonal_decomposition_plot(df):
    df = df.set_index('data')
    decomposition = seasonal_decompose(df['velocidade'], model='additive', period=24)
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    decomposition.observed.plot(ax=ax[0], title='Observado')
    decomposition.trend.plot(ax=ax[1], title='Tendência')
    decomposition.seasonal.plot(ax=ax[2], title='Sazonalidade')
    plt.tight_layout()
    st.pyplot(fig)

def plot_interactive_graph(df, x_col, y_col):
    fig = px.line(df, x=x_col, y=y_col, title=f'{y_col} ao Longo do Tempo')
    fig.update_layout(xaxis_title='Data', yaxis_title=y_col)
    st.plotly_chart(fig)

# === INTERFACE PRINCIPAL ===
def main():
    raw_df_all = get_data()
    route_name = st.selectbox("Selecione a rota:", raw_df_all['route_name'].unique())
    route_id = raw_df_all[raw_df_all['route_name'] == route_name]['route_id'].iloc[0]

    date_range = st.date_input("Intervalo de datas", [pd.to_datetime('today') - pd.Timedelta(days=7), pd.to_datetime('today')])
    df = get_data(date_range[0], date_range[1], route_id)

    with st.expander("📋 Dados Brutos"):
        st.dataframe(df)

    df = clean_data(df, route_name)
    if len(df) < 20:
        st.error("Dados insuficientes.")
        st.stop()

    date_range = st.date_input("Refinar intervalo", [df['data'].min(), df['data'].max()])
    min_v, max_v = st.slider("Faixa de velocidade (km/h)", int(df['velocidade'].min()), int(df['velocidade'].max()), (int(df['velocidade'].min()), int(df['velocidade'].max())))
    df = df[(df['data'] >= pd.to_datetime(date_range[0])) & (df['data'] <= pd.to_datetime(date_range[1])) & (df['velocidade'] >= min_v) & (df['velocidade'] <= max_v)]

    if df.empty:
        st.warning("Sem dados para os filtros selecionados.")
        st.stop()

    st.markdown("### 🧠 Insights")
    st.markdown(gerar_insights(df))

    arima_df = df[['data', 'velocidade']].rename(columns={'data': 'ds', 'velocidade': 'y'})
    arima_df['y'] = pd.to_numeric(arima_df['y'], errors='raise')
    forecast_df = create_arima_forecast(arima_df, route_id)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=arima_df['ds'], y=arima_df['y'], mode='lines', name='Histórico', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Previsão', line=dict(color='orange')))
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'].tolist() + forecast_df['ds'][::-1].tolist(),
        y=forecast_df['yhat_upper'].tolist() + forecast_df['yhat_lower'][::-1].tolist(),
        fill='toself', fillcolor='rgba(255,165,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'), showlegend=True,
        name='Intervalo de Confiança'))
    fig.update_layout(title="📈 Previsão ARIMA", xaxis_title="Data", yaxis_title="Velocidade (km/h)", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    save_forecast_to_db(forecast_df)

    st.subheader("📉 Decomposição da Série Temporal")
    seasonal_decomposition_plot(df)

    st.subheader("📊 Velocidade Interativa")
    plot_interactive_graph(df, 'data', 'velocidade')

    st.subheader("🚨 Anomalias Detectadas")
    anomalies = detect_anomalies(df)
    if not anomalies.empty:
        st.dataframe(anomalies[['data', 'velocidade', 'vel_diff']].sort_values('data', ascending=False))
    else:
        st.success("Nenhuma anomalia detectada.")

    st.subheader("📅 Tendência de Velocidade")
    fig_trend = px.scatter(df, x='data', y='velocidade', trendline='ols', title="Tendência de Velocidade")
    st.plotly_chart(fig_trend)

    st.subheader("📄 Exportar Dados")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar CSV", data=csv, file_name="dados_filtrados.csv", mime='text/csv')

    # Atualiza a cada 5 minutos
    time.sleep(300)
    st.experimental_rerun()

if __name__ == "__main__":
    main()