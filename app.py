import streamlit as st

# Configuração da página DEVE SER A PRIMEIRA CHAMADA
st.set_page_config(
    page_title="Análise de Rotas Inteligente",
    layout="wide",
    page_icon="📊"
)

# Restante das importações
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import mysql.connector
import pytz
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error

# Configurações de compatibilidade do numpy
if np.__version__.startswith('2.'):
    np.float_ = np.float64
    np.int_ = np.int_
    np.bool_ = np.bool_

TIMEZONE = pytz.timezone('America/Sao_Paulo')

# Tema personalizado
custom_theme = """
<style>
:root {
    --primary-color: #007BFF;
    --background-color: #000;
    --secondary-background-color: #FFFFFF;
    --accent-color: #17A2B8;
    --text-color: #343A40;
    --header-font: 'Segoe UI', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--header-font);
    color: var(--text-color);
    background-color: var(--background-color);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--primary-color);
    font-weight: 600;
}

.stApp {
    background-color: var(--background-color);
}

.stSidebar {
    background-color: var(--secondary-background-color) !important;
    color: var(--text-color);
}

.stButton>button {
    background-color: var(--primary-color);
    color: white;
    border-radius: 8px;
}

.stCheckbox>label {
    color: var(--text-color);
}

.stSelectbox>label {
    color: var(--text-color);
}
</style>
"""
st.markdown(custom_theme, unsafe_allow_html=True)

# faz conexxão com o banco de dados MySQL
def get_db_connection():
    return mysql.connector.connect(
        host="185.213.81.52",
        user="u335174317_wazeportal",
        password="@Ndre2025.",
        database="u335174317_wazeportal"
    )

def get_data(start_date=None, end_date=None, route_id=None):
    try:
        mydb = get_db_connection()
        mycursor = mydb.cursor()
        
        query = """
            SELECT hr.route_id, r.name AS route_name, hr.data, hr.velocidade 
            FROM historic_routes hr
            JOIN routes r ON hr.route_id = r.id
        """
        conditions = []
        params = []
        
        if start_date:
            conditions.append("hr.data >= %s")
            params.append(start_date)
        if end_date:
            conditions.append("hr.data <= %s")
            params.append(end_date)
        if route_id:
            conditions.append("hr.route_id = %s")
            params.append(route_id)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY hr.data ASC"

        mycursor.execute(query, params)
        results = mycursor.fetchall()
        col_names = [desc[0] for desc in mycursor.description]
        df = pd.DataFrame(results, columns=col_names)
        
        df['velocidade'] = pd.to_numeric(df['velocidade'], errors='coerce')
        return df, None
    except Exception as e:
        return None, str(e)
    finally:
        mycursor.close()
        mydb.close()

def get_route_coordinates(route_id):
    try:
        mydb = get_db_connection()
        mycursor = mydb.cursor()
        query = "SELECT x, y FROM route_lines WHERE route_id = %s ORDER BY id"
        mycursor.execute(query, (route_id,))
        results = mycursor.fetchall()
        df = pd.DataFrame(results, columns=['longitude', 'latitude'])
        mycursor.close()
        mydb.close()
        return df
    except Exception as e:
        st.error(f"Erro ao obter coordenadas: {e}")
        return pd.DataFrame()

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
    df = df.set_index('data').asfreq('3T')  # Definir frequência temporal
    decomposition = seasonal_decompose(df['velocidade'], model='additive', period=24)
    fig, ax = plt.subplots(3, 1, figsize=(12, 8))
    decomposition.observed.plot(ax=ax[0], title='Observado')
    decomposition.trend.plot(ax=ax[1], title='Tendência')
    decomposition.seasonal.plot(ax=ax[2], title='Sazonalidade')
    plt.tight_layout()
    st.pyplot(fig)

def create_arima_forecast(df, route_id, steps=10):
    try:
        model = auto_arima(df['y'], seasonal=True, m=480,
                          error_action='ignore', suppress_warnings=True)
        forecast = model.predict(n_periods=steps)
        
        last_date = df['ds'].max()
        future_dates = pd.date_range(start=last_date, periods=steps+1, freq='3min')[1:]
        
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast,
            'yhat_lower': forecast - 5,
            'yhat_upper': forecast + 5,
            'id_route': route_id
        })
    except Exception as e:
        st.error(f"Erro no modelo de previsão: {str(e)}")
        return pd.DataFrame()

def save_forecast_to_db(forecast_df):
    try:
        engine = create_engine('mysql+mysqlconnector://u335174317_wazeportal:%40Ndre2025.@185.213.81.52/u335174317_wazeportal')
        with engine.begin() as connection:
            forecast_df.to_sql('forecast_history', con=connection, if_exists='append', index=False)
    except Exception as e:
        st.error(f"Erro ao salvar previsão: {e}")

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
    insights.append(f"🕒 Hora mais lenta: **{hora_mais_lenta:02d}:00**")

    return "\n\n".join(insights)

def main():
    with st.sidebar:
        st.title("ℹ️ Painel de Controle")
        st.markdown("""
            **Configurações Avançadas:**
            - Compare múltiplas rotas
            - Ajuste parâmetros de análise
            - Visualize histórico de previsões
        """)

    st.title("🚀 Análise de Rotas Inteligente")
    
    # Carregamento de dados
    with st.spinner('Carregando dados básicos...'):
        raw_df_all_routes, error = get_data()
        if error:
            st.error(f"Erro: {error}")
            st.stop()

    # Seleção de rotas
    col1, col2 = st.columns(2)
    with col1:
        route_name = st.selectbox("Rota Principal:", raw_df_all_routes['route_name'].unique())
    with col2:
        compare_enabled = st.checkbox("Comparar com outra rota")
        if compare_enabled:
            second_route = st.selectbox("Rota Secundária:", 
                                      [r for r in raw_df_all_routes['route_name'].unique() if r != route_name])

    routes_to_process = [route_name]
    if compare_enabled:
        routes_to_process.append(second_route)

    all_data = {}
    for route in routes_to_process:
        with st.spinner(f'Processando {route}...'):
            route_id = raw_df_all_routes[raw_df_all_routes['route_name'] == route]['route_id'].unique()[0]
            
            date_range = st.date_input(f"Intervalo para {route}", 
                                     value=(pd.to_datetime('today') - pd.Timedelta(days=7), pd.to_datetime('today')),
                                     max_value=pd.to_datetime('today'),
                                     key=f"date_{route_id}")
            
            if date_range[0] > date_range[1]:
                st.error("Data final não pode ser anterior à data inicial")
                st.stop()

            raw_df, error = get_data(
                start_date=date_range[0].strftime('%Y-%m-%d'),
                end_date=date_range[1].strftime('%Y-%m-%d'),
                route_id=route_id
            )
            
            processed_df = clean_data(raw_df, route)
            all_data[route] = {
                'data': processed_df,
                'id': route_id
            }

    # Seção do Mapa
    st.header("🗺️ Visualização Geográfica")
    for route in routes_to_process:
        with st.expander(f"Mapa da Rota: {route}", expanded=True):
            route_coords = get_route_coordinates(all_data[route]['id'])
            
            if not route_coords.empty:
                max_lat = route_coords['latitude'].max() + 0.002
                min_lat = route_coords['latitude'].min() - 0.002
                max_lon = route_coords['longitude'].max() + 0.002
                min_lon = route_coords['longitude'].min() - 0.002
                
                fig = go.Figure(go.Scattermapbox(
                    mode="lines+markers",
                    lon=route_coords['longitude'],
                    lat=route_coords['latitude'],
                    marker={'size': 10, 'color': "#FF0000"},
                    line=dict(width=4, color='#1E90FF'),
                    hovertext=[f"Ponto {i+1}" for i in range(len(route_coords))],
                    hoverinfo="text+lat+lon"
                ))

                fig.update_layout(
                    mapbox={
                        'style': "open-street-map",
                        'center': {'lat': (max_lat + min_lat)/2, 'lon': (max_lon + min_lon)/2},
                        'zoom': 13,
                        'bounds': {
                            'west': min_lon,
                            'east': max_lon,
                            'south': min_lat,
                            'north': max_lat
                        }
                    },
                    margin={"r":0,"t":0,"l":0,"b":0},
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Nenhuma coordenada geográfica encontrada")

    # Seção de Análise
    st.header("📈 Análise Preditiva")
    for route in routes_to_process:
        with st.expander(f"Análise para {route}", expanded=True):
            processed_df = all_data[route]['data']
            
            st.subheader("🔮 Previsão de Velocidade")
            arima_df = processed_df[['data', 'velocidade']].rename(columns={
                'data': 'ds', 
                'velocidade': 'y'
            })
            
            steps = st.slider("⏱️ Passos de previsão (3min cada)", 5, 20, 10, key=f"steps_{route}")
            arima_forecast = create_arima_forecast(arima_df, all_data[route]['id'], steps)
            
            if not arima_forecast.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=arima_df['ds'], 
                    y=arima_df['y'],
                    mode='lines',
                    name='Histórico'
                ))
                fig.add_trace(go.Scatter(
                    x=arima_forecast['ds'],
                    y=arima_forecast['yhat'],
                    mode='lines',
                    name='Previsão'
                ))
                fig.update_layout(
                    title=f"Previsão para {route}",
                    xaxis_title="Data e Hora",
                    yaxis_title="Velocidade (km/h)"
                )
                st.plotly_chart(fig)
                
                save_forecast_to_db(arima_forecast)

            st.subheader("🧠 Insights Automáticos")
            st.markdown(gerar_insights(processed_df))

            st.subheader("📉 Decomposição Temporal")
            seasonal_decomposition_plot(processed_df)

            st.subheader("🔥 Heatmap Horário por Dia da Semana")
            pivot_table = processed_df.pivot_table(
                index='day_of_week',
                columns='hour',
                values='velocidade',
                aggfunc='mean'
            )

            # Reordenar dias da semana
            dias_ordenados = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            pivot_table = pivot_table.reindex(dias_ordenados)

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="coolwarm_r", ax=ax)
            ax.set_title("Velocidade Média por Dia da Semana e Hora")
            ax.set_xlabel("Hora do Dia")
            ax.set_ylabel("Dia da Semana")
            st.pyplot(fig)


    # Seção Técnica
    st.header("⚙️ Detalhes Técnicos")
    with st.expander("Relatório de Qualidade de Dados"):
        for route in routes_to_process:
            st.subheader(f"Qualidade dos Dados: {route}")
            report = {
                "total_registros": len(all_data[route]['data']),
                "registros_faltantes": all_data[route]['data']['velocidade'].isnull().sum(),
                "outliers_detectados": len(all_data[route]['data'][all_data[route]['data']['velocidade'] > 150]),
                "cobertura_temporal": f"{all_data[route]['data']['data'].min().date()} a {all_data[route]['data']['data'].max().date()}"
            }
            st.json(report)

if __name__ == "__main__":
    main()