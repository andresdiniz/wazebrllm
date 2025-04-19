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

if np.__version__.startswith('2.'):
    np.float_ = np.float64
    np.int_ = np.int_
    np.bool_ = np.bool_

st.set_page_config(page_title="Análise de Rotas", layout="wide")
st.title("📊 Previsão de Velocidade e Análise de Anomalias")

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
            SELECT
                hr.route_id,
                r.name AS route_name,
                hr.data,
                hr.velocidade
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
        st.error(f"Falha na conexão com o banco: {str(e)}")
        st.stop()

def clean_data(df, route):
    df = df[df['route_name'] == route].copy()
    df['data'] = pd.to_datetime(df['data']).dt.tz_localize(None)
    df = df.sort_values('data')
    df['velocidade'] = (
        df['velocidade']
        .clip(upper=150)
        .interpolate(method='linear')
        .ffill()
        .bfill()
    )
    df['day_of_week'] = df['data'].dt.day_name()
    df['hour'] = df['data'].dt.hour
    return df.dropna(subset=['velocidade'])

def detect_anomalies(df):
    df = df.copy()
    df['vel_diff'] = df['velocidade'].diff().abs()
    threshold = df['vel_diff'].quantile(0.95) * 1.5
    return df[df['vel_diff'] > max(threshold, 20)]

def plot_interactive_graph(df, x_col, y_col):
    if 'data' not in df.columns:
        df['data'] = df.index
    fig = px.line(df, x=x_col, y=y_col, title=f'{y_col} ao Longo do Tempo')
    fig.update_layout(xaxis_title='Data', yaxis_title=y_col)
    st.plotly_chart(fig)

def seasonal_decomposition_plot(df):
    df = df.set_index('data')
    decomposition = seasonal_decompose(df['velocidade'], model='additive', period=24)
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    decomposition.observed.plot(ax=ax[0], title='Observado')
    decomposition.trend.plot(ax=ax[1], title='Tendência')
    decomposition.seasonal.plot(ax=ax[2], title='Sazonalidade')
    plt.tight_layout()
    st.pyplot(fig)

def create_arima_forecast(df, route_id):
    try:
        model = ARIMA(df['y'], order=(3, 1, 1))
        results = model.fit()
        forecast = results.get_forecast(steps=10)

        sao_paulo_tz = pytz.timezone('America/Sao_Paulo')
        last_timestamp_utc = pd.to_datetime(df['ds'].max())

        future_timestamps = pd.date_range(start=last_timestamp_utc, periods=len(forecast.predicted_mean) + 1, freq='3min', tz='America/Sao_Paulo')[1:]

        forecast_df = pd.DataFrame({
            'ds': future_timestamps,
            'yhat': forecast.predicted_mean,
            'yhat_lower': forecast.conf_int().iloc[:, 0],
            'yhat_upper': forecast.conf_int().iloc[:, 1]
        })
        forecast_df['id_route'] = route_id
        return forecast_df
    except Exception as e:
        st.warning(f"Erro ao treinar o modelo ARIMA: {e}.")
        last_value = df['y'].iloc[-1]
        future_timestamps = pd.date_range(start=pd.to_datetime(df['ds'].max()), periods=11, freq='3min', tz='America/Sao_Paulo')[1:]
        return pd.DataFrame({
            'ds': future_timestamps,
            'yhat': [last_value] * 10,
            'yhat_lower': [last_value - 5] * 10,
            'yhat_upper': [last_value + 5] * 10,
            'id_route': [route_id] * 10
        })

def save_forecast_to_db(forecast_df):
    try:
        engine = create_engine('mysql+mysqlconnector://u335174317_wazeportal:%40Ndre2025.@185.213.81.52/u335174317_wazeportal')
        with engine.begin() as connection:
            forecast_df.to_sql('forecast_history', con=connection, if_exists='append', index=False)
    except Exception as e:
        st.error(f"Erro ao salvar previsão no banco de dados: {e}")

def gerar_insights(df):
    insights = []
    media_geral = df['velocidade'].mean()
    dia_mais_lento = df.groupby('data')['velocidade'].mean().idxmin()
    velocidade_dia_mais_lento = df[df['data'] == dia_mais_lento]['velocidade'].mean()
    dia_da_semana_mais_lento = df.groupby('day_of_week')['velocidade'].mean().idxmin()
    hora_mais_lenta = df.groupby('hour')['velocidade'].mean().idxmin()

    insights.append(f"📌 Velocidade média geral: **{media_geral:.2f} km/h**")
    insights.append(f"📅 Dia mais lento: **{dia_mais_lento.strftime('%Y-%m-%d')}** ({velocidade_dia_mais_lento:.2f} km/h)")
    insights.append(f"📅 Dia da semana mais lento: **{dia_da_semana_mais_lento}**")
    insights.append(f"🕒 Hora mais lenta: **{hora_mais_lenta}:00**")

    return "\n\n".join(insights)

def main():
    raw_df_all_routes = get_data()
    route_name = st.selectbox("Selecione a rota:", raw_df_all_routes['route_name'].unique())
    route_id = raw_df_all_routes[raw_df_all_routes['route_name'] == route_name]['route_id'].unique()[0]

    st.subheader("🔎 Filtros")
    date_range = st.date_input("Intervalo de datas", [pd.to_datetime('today') - pd.Timedelta(days=7), pd.to_datetime('today')])
    date_start = date_range[0].strftime('%Y-%m-%d')
    date_end = date_range[1].strftime('%Y-%m-%d')

    raw_df = get_data(start_date=date_start, end_date=date_end, route_id=route_id)

    with st.expander("📋 Visualizar dados brutos"):
        st.dataframe(raw_df)

    processed_df = clean_data(raw_df, route_name)

    if len(processed_df) < 20:
        st.error("Dados insuficientes após limpeza (mínimo 20 registros).")
        st.stop()

    st.subheader("🔮 Previsão de Velocidade (ARIMA)")
    st.markdown("""
        Este gráfico mostra a previsão de velocidade para os próximos 30 minutos (10 passos de 3 minutos) usando o modelo ARIMA.
        A linha laranja representa a previsão, enquanto a faixa sombreada mostra o intervalo de confiança.
    """)

    st.markdown("### 🧠 Insights Automáticos")
    st.markdown(gerar_insights(processed_df))

    arima_df = processed_df[['data', 'velocidade']].rename(columns={'data': 'ds', 'velocidade': 'y'}).copy()
    arima_df['y'] = pd.to_numeric(arima_df['y'], errors='raise')
    arima_df_display = arima_df[-200:] if len(arima_df) > 200 else arima_df.copy()

    arima_forecast = create_arima_forecast(arima_df, route_id)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=arima_df_display['ds'], y=arima_df_display['y'], mode='lines', name='Histórico', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=arima_forecast['ds'], y=arima_forecast['yhat'], mode='lines', name='Previsão', line=dict(color='orange')))
    fig.add_trace(go.Scatter(
        x=arima_forecast['ds'].tolist() + arima_forecast['ds'][::-1].tolist(),
        y=arima_forecast['yhat_upper'].tolist() + arima_forecast['yhat_lower'][::-1].tolist(),
        fill='toself', fillcolor='rgba(255,165,0,0.2)', line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip", showlegend=True, name='Intervalo de Confiança'))

    fig.update_layout(
        title="📈 Previsão de Velocidade com ARIMA",
        xaxis_title="Data e Hora",
        yaxis_title="Velocidade (km/h)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    save_forecast_to_db(arima_forecast)

    st.subheader("📉 Decomposição de Série Temporal")
    seasonal_decomposition_plot(processed_df)

    st.subheader("📊 Gráfico Interativo de Velocidade")
    plot_interactive_graph(processed_df, 'data', 'velocidade')

    st.subheader("🚨 Detecção de Anomalias")
    anomalies = detect_anomalies(processed_df)
    if not anomalies.empty:
        st.dataframe(anomalies[['data', 'velocidade', 'vel_diff']].sort_values('data', ascending=False).style.highlight_max(color='#ff6666'))
    else:
        st.success("Nenhuma anomalia significativa detectada.")

    st.subheader("📥 Baixar Gráfico de Previsão")
    buffer = BytesIO()
    fig.write_image(buffer, format='png')
    buffer.seek(0)
    st.download_button("Baixar gráfico", buffer, file_name="forecast_plot.png")

    st.markdown("### 📉 Tendência Geral de Velocidade")
    trend_df = processed_df.copy()
    trend_sample = trend_df[::max(1, len(trend_df)//500)]
    fig = px.scatter(trend_sample, x='data', y='velocidade', trendline='ols', title="Velocidade com Linha de Tendência")
    fig.update_layout(template="plotly_white", xaxis_title="Data", yaxis_title="Velocidade (km/h)")
    st.plotly_chart(fig)

    st.subheader("📥 Exportar Dados Filtrados")
    csv = processed_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Baixar CSV", data=csv, file_name='dados_filtrados.csv', mime='text/csv')

    time.sleep(300)
    st.experimental_rerun()

if __name__ == "__main__":
    main()