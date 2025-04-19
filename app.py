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
import mysql.connector
import pytz
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error

# Configurações de cache e performance
if np.__version__.startswith('2.'):
    np.float_ = np.float64
    np.int_ = np.int_
    np.bool_ = np.bool_

TIMEZONE = pytz.timezone('America/Sao_Paulo')

# Configuração de tema personalizado
custom_theme = """
<style>
:root {
    --primary-color: #1E90FF;
    --background-color: #F0F2F6;
    --secondary-background-color: #FFFFFF;
    --text-color: #262730;
}

.stApp {
    background-color: var(--background-color);
    color: var(--text-color);
}

.stSidebar {
    background-color: var(--secondary-background-color) !important;
}

h1, h2, h3 {
    color: var(--primary-color) !important;
}
</style>
"""
st.markdown(custom_theme, unsafe_allow_html=True)

st.set_page_config(page_title="Análise de Rotas Inteligente", layout="wide")

# Cache para conexão de banco de dados
@st.cache_resource
def get_db_connection():
    return mysql.connector.connect(
        host="185.213.81.52",
        user="u335174317_wazeportal",
        password="@Ndre2025.",
        database="u335174317_wazeportal"
    )

# Cache para dados com atualização periódica
@st.cache_data(ttl=3600, show_spinner="Carregando dados...")
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

# Função para análise de qualidade dos dados
def data_quality_report(df):
    report = {
        "total_registros": len(df),
        "registros_faltantes": df['velocidade'].isnull().sum(),
        "outliers_velocidade": len(df[df['velocidade'] > 150]),
        "velocidade_media": df['velocidade'].mean(),
        "periodo_cobertura": f"{df['data'].min().date()} a {df['data'].max().date()}"
    }
    return report

# Função para carregar histórico de previsões
def load_historical_forecasts(route_id):
    try:
        mydb = get_db_connection()
        query = """
            SELECT f.ds AS data_previsao, f.yhat, h.data, h.velocidade
            FROM forecast_history f
            JOIN historic_routes h ON f.ds = h.data AND f.id_route = h.route_id
            WHERE f.id_route = %s
        """
        df = pd.read_sql(query, mydb, params=(route_id,))
        
        if not df.empty:
            df['erro'] = abs(df['yhat'] - df['velocidade'])
            df['erro_percentual'] = (df['erro'] / df['velocidade']) * 100
        return df
    except Exception as e:
        st.error(f"Erro ao carregar histórico: {e}")
        return pd.DataFrame()

def main():
    with st.sidebar:
        st.title("ℹ️ Painel de Controle")
        st.markdown("""
            **Configurações Avançadas:**
            - Selecione rotas para comparação
            - Ajuste parâmetros de análise
        """)

    st.title("🚀 Análise de Rotas Inteligente")
    
    # Carregamento de dados com feedback visual
    with st.spinner('Carregando dados básicos...'):
        raw_df_all_routes, error = get_data()
        if error:
            st.error(f"Erro: {error}")
            st.stop()

    # Seleção de rotas para comparação
    col1, col2 = st.columns(2)
    with col1:
        route_name = st.selectbox("Rota Principal:", raw_df_all_routes['route_name'].unique())
    with col2:
        compare_enabled = st.checkbox("Comparar com outra rota")
        if compare_enabled:
            second_route = st.selectbox("Rota Secundária:", 
                                      [r for r in raw_df_all_routes['route_name'].unique() if r != route_name])

    # Processamento paralelo para múltiplas rotas
    routes_to_process = [route_name]
    if compare_enabled:
        routes_to_process.append(second_route)

    all_data = {}
    for route in routes_to_process:
        with st.spinner(f'Processando {route}...'):
            route_id = raw_df_all_routes[raw_df_all_routes['route_name'] == route]['route_id'].unique()[0]
            
            # Validação de datas
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

    # Seção do Mapa da Rota com Zoom Automático
    st.subheader("🗺️ Mapa Inteligente da Rota")
    for route in routes_to_process:
        with st.expander(f"Mapa para {route}", expanded=True):
            route_coords = get_route_coordinates(all_data[route]['id'])
            
            if not route_coords.empty:
                # Cálculo de limites para zoom automático
                max_lat = route_coords['y'].max() + 0.002
                min_lat = route_coords['y'].min() - 0.002
                max_lon = route_coords['x'].max() + 0.002
                min_lon = route_coords['x'].min() - 0.002
                
                fig = go.Figure(go.Scattermapbox(
                    mode="lines+markers",
                    lon=route_coords['x'],
                    lat=route_coords['y'],
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
                st.warning("Nenhuma coordenada encontrada para esta rota")

    # Seção de Previsão com Auto-ARIMA
    st.header("🔮 Previsão Inteligente")
    for route in routes_to_process:
        with st.expander(f"Previsão para {route}", expanded=True):
            arima_df = all_data[route]['data'][['data', 'velocidade']].rename(columns={
                'data': 'ds', 
                'velocidade': 'y'
            })
            
            # Modelagem com Auto-ARIMA
            with st.spinner('Treinando modelo inteligente...'):
                try:
                    model = auto_arima(arima_df['y'], seasonal=True, m=24,
                                      error_action='ignore', suppress_warnings=True)
                    forecast = model.predict(n_periods=10)
                    
                    # Criação do dataframe de previsão
                    last_date = arima_df['ds'].max()
                    future_dates = pd.date_range(start=last_date, periods=11, freq='3min')[1:]
                    
                    forecast_df = pd.DataFrame({
                        'ds': future_dates,
                        'yhat': forecast,
                        'id_route': all_data[route]['id']
                    })
                    
                    # Exibição da previsão
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=arima_df['ds'], 
                        y=arima_df['y'],
                        mode='lines',
                        name='Histórico'
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast_df['ds'],
                        y=forecast_df['yhat'],
                        mode='lines',
                        name='Previsão'
                    ))
                    fig.update_layout(
                        title=f"Previsão para {route}",
                        xaxis_title="Data e Hora",
                        yaxis_title="Velocidade (km/h)"
                    )
                    st.plotly_chart(fig)
                    
                    # Análise de precisão histórica
                    hist_forecasts = load_historical_forecasts(all_data[route]['id'])
                    if not hist_forecasts.empty:
                        mae = mean_absolute_error(hist_forecasts['velocidade'], hist_forecasts['yhat'])
                        st.metric("Precisão Histórica (MAE)", f"{mae:.2f} km/h")
                    
                    save_forecast_to_db(forecast_df)
                    
                except Exception as e:
                    st.error(f"Erro no modelo de previsão: {str(e)}")

    # Seção de Auditoria de Dados
    st.header("🔍 Auditoria de Qualidade")
    with st.expander("Relatório Completo de Dados"):
        for route in routes_to_process:
            report = data_quality_report(all_data[route]['data'])
            st.subheader(f"Relatório para {route}")
            st.json(report)

    # Integração com dados externos (exemplo climático)
    if st.checkbox("Mostrar análise contextual (beta)"):
        st.info("Funcionalidade em desenvolvimento - integração com APIs externas")

if __name__ == "__main__":
    main()