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
import pytz  # Importe a biblioteca pytz para trabalhar com fusos horários

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

        st.write("Tipos de dados lidos do banco:")
        st.write(df.dtypes)

        # Tenta converter a coluna 'velocidade' para numérico, forçando erros para NaN
        df['velocidade'] = pd.to_numeric(df['velocidade'], errors='coerce')

        # Exibe as linhas onde a conversão falhou (eram valores não numéricos)
        non_numeric_rows = df[df['velocidade'].isna()]
        if not non_numeric_rows.empty:
            st.warning("Valores não numéricos encontrados na coluna 'velocidade' ao ler do banco (convertidos para NaN):")
            st.dataframe(non_numeric_rows)

        return df
    except Exception as e:
        st.error(f"Falha na conexão com o banco (usando mysql.connector): {str(e)}")
        st.stop()

# === LIMPEZA E FEATURE ENGINEERING ===
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

# === DETECÇÃO DE ANOMALIAS ===
def detect_anomalies(df):
    df = df.copy()
    df['vel_diff'] = df['velocidade'].diff().abs()
    threshold = df['vel_diff'].quantile(0.95) * 1.5
    return df[df['vel_diff'] > max(threshold, 20)]

# === GRÁFICO INTERATIVO COM PLOTLY ===
def plot_interactive_graph(df, x_col, y_col):
    if 'data' not in df.columns:
        df['data'] = df.index
    fig = px.line(df, x=x_col, y=y_col, title=f'{y_col} ao Longo do Tempo')
    fig.update_layout(xaxis_title='Data', yaxis_title=y_col)
    st.plotly_chart(fig)

# === DECOMPOSIÇÃO DE SÉRIE TEMPORAL ===
def seasonal_decomposition_plot(df):
    df = df.set_index('data')
    decomposition = seasonal_decompose(df['velocidade'], model='additive', period=24)
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    decomposition.observed.plot(ax=ax[0], title='Observado')
    decomposition.trend.plot(ax=ax[1], title='Tendência')
    decomposition.seasonal.plot(ax=ax[2], title='Sazonalidade')
    plt.tight_layout()
    st.pyplot(fig)

# === PREVISÃO COM ARIMA ===
def create_arima_forecast(df, route_id):
    # Experimentando com uma ordem diferente do ARIMA
    # Você pode tentar outras ordens como (5, 1, 0), (0, 1, 3), (2, 1, 2) etc.
    # A escolha da ordem ideal geralmente envolve análise da função de autocorrelação (ACF) e função de autocorrelação parcial (PACF).
    try:
        model = ARIMA(df['y'], order=(3, 1, 1))
        results = model.fit()
        # Reduzindo o número de passos da previsão para 10 (30 minutos)
        forecast = results.get_forecast(steps=10)

        # Define o fuso horário de São Paulo (GMT-3)
        sao_paulo_tz = pytz.timezone('America/Sao_Paulo')

        # Pega o último timestamp DOS DADOS LIMPOS (antes de qualquer manipulação de tz na previsão)
        last_timestamp_utc = pd.to_datetime(df['ds'].max())

        # Gera as datas futuras JÁ NO fuso horário de São Paulo
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
        st.warning(f"Erro ao treinar o modelo ARIMA: {e}. Retornando uma previsão simples.")
        # Em caso de erro, retorna uma previsão mais simples para evitar quebrar o app
        last_value = df['y'].iloc[-1]
        future_timestamps = pd.date_range(start=pd.to_datetime(df['ds'].max()), periods=11, freq='3min', tz='America/Sao_Paulo')[1:]
        return pd.DataFrame({
            'ds': future_timestamps,
            'yhat': [last_value] * 10,
            'yhat_lower': [last_value - 5] * 10,
            'yhat_upper': [last_value + 5] * 10,
            'id_route': [route_id] * 10
        })

# === SALVAR PREVISÃO NO BANCO ===
def save_forecast_to_db(forecast_df):
    try:
        engine = create_engine('mysql+mysqlconnector://u335174317_wazeportal:%40Ndre2025.@185.213.81.52/u335174317_wazeportal')
        with engine.begin() as connection:  # Abre uma conexão (transação implícita com begin)
            forecast_df.to_sql('forecast_history', con=connection, if_exists='append', index=False)
    except Exception as e:
        st.error(f"Erro ao salvar previsão no banco de dados: {e}")

# === GERAÇÃO DE INSIGHTS ===
def gerar_insights(df):
    insights = []
    media_geral = df['velocidade'].mean()

    # Dia mais lento (data específica)
    dia_mais_lento = df.groupby('data')['velocidade'].mean().idxmin()
    velocidade_dia_mais_lento = df[df['data'] == dia_mais_lento]['velocidade'].mean()

    # Dia da semana mais lento
    dia_da_semana_mais_lento = df.groupby('day_of_week')['velocidade'].mean().idxmin()

    # Hora mais lenta
    hora_mais_lenta = df.groupby('hour')['velocidade'].mean().idxmin()

    insights.append(f"📌 A velocidade média geral nesta rota foi de **{media_geral:.2f} km/h**.")
    insights.append(f"📅 O dia mais lento foi **{dia_mais_lento.strftime('%Y-%m-%d')}**, com uma velocidade média de **{velocidade_dia_mais_lento:.2f} km/h**.")
    insights.append(f"📅 O dia da semana mais lento foi **{dia_da_semana_mais_lento}**.")
    insights.append(f"🕒 A hora mais lenta foi **{hora_mais_lenta}:00**, com a menor média de velocidade.")

    return "\n\n".join(insights)

# === INTERFACE PRINCIPAL ===
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

    route_name = st.selectbox("Selecione a rota:", raw_df['route_name'].unique())
    route_id = raw_df[raw_df['route_name'] == route_name]['route_id'].unique()[0]
    processed_df = clean_data(raw_df, route_name)

    if len(processed_df) < 20:
        st.error("Dados insuficientes após limpeza (mínimo 20 registros).")
        st.stop()

    # === FILTROS ===
    st.subheader("🔎 Filtros")
    date_range = st.date_input("Intervalo de datas", [processed_df['data'].min(), processed_df['data'].max()])
    min_speed, max_speed = st.slider(
        "Faixa de velocidade (km/h)",
        min_value=int(processed_df['velocidade'].min()),
        max_value=int(processed_df['velocidade'].max()),
        value=(int(processed_df['velocidade'].min()), int(processed_df['velocidade'].max()))
    )

    date_start = pd.to_datetime(date_range[0])
    date_end = pd.to_datetime(date_range[1])

    processed_df = processed_df[(
        processed_df['data'] >= date_start) &
        (processed_df['data'] <= date_end) &
        (processed_df['velocidade'] >= min_speed) &
        (processed_df['velocidade'] <= max_speed)
    ]

    if processed_df.empty:
        st.warning("Nenhum dado disponível após aplicar os filtros.")
        st.stop()

    # === PREVISÃO ARIMA ===
    st.subheader("🔮 Previsão de Velocidade (ARIMA)")

    st.markdown("""Este gráfico mostra a previsão de velocidade para os próximos 30 minutos (10 passos de 3 minutos) usando o modelo ARIMA.
    A linha laranja representa a previsão, enquanto a faixa sombreada mostra o intervalo de confiança da previsão.
    """)

    st.markdown("### 🧠 Insights Gerados Automaticamente")
    st.markdown(gerar_insights(processed_df))

    display_limit = 200
    arima_df = processed_df[['data', 'velocidade']].rename(columns={'data': 'ds', 'velocidade': 'y'}).copy()

    # Garante que a coluna 'y' seja numérica antes de passar para o ARIMA
    arima_df['y'] = pd.to_numeric(arima_df['y'], errors='raise') # Usamos 'raise' para ver se ainda há erros

    arima_df_display = arima_df[-display_limit:] if len(arima_df) > display_limit else arima_df.copy()

    arima_forecast = create_arima_forecast(arima_df,route_id)

    # === GRÁFICO INTERATIVO COM PLOTLY
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=arima_df_display['ds'], y=arima_df_display['y'],
        mode='lines',
        name='Histórico',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=arima_forecast['ds'], y=arima_forecast['yhat'],
        mode='lines',
        name='Previsão',
        line=dict(color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=arima_forecast['ds'].tolist() + arima_forecast['ds'][::-1].tolist(),
        y=arima_forecast['yhat_upper'].tolist() + arima_forecast['yhat_lower'][::-1].tolist(),
        fill='toself',
        fillcolor='rgba(255,165,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Intervalo de Confiança'
    ))

    fig.update_layout(
        title="📈 Previsão de Velocidade com ARIMA",
        xaxis_title="Data e Hora",
        yaxis_title="Velocidade (km/h)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Salvar no banco
    save_forecast_to_db(arima_forecast)

    # === DECOMPOSIÇÃO DE SÉRIE TEMPORAL ===
    st.subheader("📉 Decomposição de Série Temporal")
    seasonal_decomposition_plot(processed_df)

    # === GRÁFICO INTERATIVO DE VELOCIDADE ===
    st.subheader("📊 Gráfico Interativo de Velocidade")
    plot_interactive_graph(processed_df, 'data', 'velocidade')

    # === DETECÇÃO DE ANOMALIAS ===
    st.subheader("🚨 Detecção de Anomalias")
    anomalies = detect_anomalies(processed_df)
    if not anomalies.empty:
        st.dataframe(
            anomalies[['data', 'velocidade', 'vel_diff']].sort_values('data', ascending=False).style.highlight_max(color='#ff6666')
        )
    else:
        st.success("Nenhuma anomalia significativa detectada.")
        
    # === DOWNLOAD DO GRÁFICO DE PREVISÃO ===
    buffer = BytesIO()
    fig.write_image(buffer, format='png')
    buffer.seek(0)

    st.subheader("📥 Baixar Gráfico de Previsão")
    st.download_button("Baixar gráfico", buffer, file_name="forecast_plot.png")

    st.markdown("### 📉 Tendência Geral de Velocidade")
    fig = px.scatter(processed_df, x='data', y='velocidade', trendline='ols', title="Velocidade com Linha de Tendência")
    st.plotly_chart(fig)

    # === EXPORTAR DADOS FILTRADOS ===
    st.subheader("📥 Exportar Dados Filtrados")
    csv = processed_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Baixar CSV",
        data=csv,
        file_name='dados_filtrados.csv',
        mime='text/csv',
    )

    # Atualizar a cada 5 minutos
    time.sleep(300)  # Atraso de 5 minutos
    st.experimental_rerun()  # Forçar o recarregamento da página

if __name__ == "__main__":
    main()