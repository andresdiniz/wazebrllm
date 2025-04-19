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
import mysql.connector
import pytz
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error

# Configurações de compatibilidade do numpy
if np.__version__.startswith('2.'):
    np.float_ = np.float64
    np.int_ = np.int_
    np.bool_ = np.bool_

# Constantes e configurações
TIMEZONE = pytz.timezone('America/Sao_Paulo')
PRIMARY_COLOR = "#00AFFF"
BACKGROUND_COLOR = "#1E1E1E"
SECONDARY_BACKGROUND_COLOR = "#2D2D2D"
ACCENT_COLOR = "#FF4B4B"
TEXT_COLOR = "#FFFFFF"
HEADER_FONT = ('Segoe UI', 'sans-serif')

# Regras de alerta
ALERT_RULES = {
    "congestion": {
        "condition": lambda df: df['velocidade'].rolling(window=20, min_periods=1).mean() < 30,
        "message": "⚠️ Congestionamento detectado na rota {} (velocidade média abaixo de 30 km/h)"
    },
    "alta_velocidade": {
        "condition": lambda df: df['velocidade'].rolling(window=20, min_periods=1).mean() > 100,
        "message": "🚨 Velocidade excessiva detectada na rota {} (média acima de 100 km/h)"
    }
}

# Tema personalizado
custom_theme = f"""
<style>
:root {{
    --primary-color: {PRIMARY_COLOR};
    --background-color: {BACKGROUND_COLOR};
    --secondary-background-color: {SECONDARY_BACKGROUND_COLOR};
    --accent-color: {ACCENT_COLOR};
    --text-color: {TEXT_COLOR};
    --header-font: {', '.join(HEADER_FONT)};
}}

/* ... [restante do tema permanece igual] ... */
</style>
"""
st.markdown(custom_theme, unsafe_allow_html=True)

# Funções de Banco de Dados
@st.cache_resource(show_spinner=False)
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            database=st.secrets["mysql"]["database"]
        )
        return conn
    except Exception as e:
        st.error(f"Erro ao conectar ao banco de dados: {e}")
        st.stop()

@st.cache_data(show_spinner=False)
def get_all_route_names():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT name FROM routes")
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        st.error(f"Erro ao obter nomes das rotas: {e}")
        return []
    finally:
        if 'cursor' in locals(): cursor.close()

@st.cache_data(show_spinner=False)
def get_data(start_date=None, end_date=None, route_name=None):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT hr.route_id, r.name AS route_name, hr.data, hr.velocidade
            FROM historic_routes hr
            JOIN routes r ON hr.route_id = r.id
        """
        conditions = []
        params = []

        if route_name:
            conditions.append("r.name = %s")
            params.append(route_name)
        if start_date:
            conditions.append("hr.data >= %s")
            params.append(start_date)
        if end_date:
            end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1)
            conditions.append("hr.data < %s")
            params.append(end_datetime.strftime('%Y-%m-%d'))

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY hr.data ASC"
        cursor.execute(query, params)
        
        col_names = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(cursor.fetchall(), columns=col_names)
        
        df['data'] = pd.to_datetime(df['data']).dt.tz_localize(None)
        df['velocidade'] = pd.to_numeric(df['velocidade'], errors='coerce')
        
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)
    finally:
        if 'cursor' in locals(): cursor.close()

@st.cache_data(show_spinner=False)
def get_route_coordinates(route_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT x, y FROM route_lines WHERE route_id = %s ORDER BY id", (route_id,))
        return pd.DataFrame(cursor.fetchall(), columns=['longitude', 'latitude'])
    except Exception as e:
        st.error(f"Erro ao obter coordenadas: {e}")
        return pd.DataFrame()
    finally:
        if 'cursor' in locals(): cursor.close()

# Funções de Processamento
def clean_data(df):
    df = df.copy()
    df = df.sort_values('data')
    df['velocidade'] = (df['velocidade']
        .clip(upper=150)
        .interpolate(method='linear')
        .ffill()
        .bfill()
    )
    df['day_of_week'] = df['data'].dt.day_name()
    df['hour'] = df['data'].dt.hour
    return df.dropna(subset=['velocidade'])

def check_alerts(df):
    alerts = []
    for rule in ALERT_RULES.values():
        if not df.empty and rule['condition'](df).iloc[-1]:
            alerts.append(rule['message'].format(df['route_name'].iloc[0]))
    return alerts

def simulate_scenario(base_data, parameters):
    simulated = base_data.copy()
    simulated['velocidade'] = (simulated['velocidade'] * parameters.get('speed_factor', 1) + 
                              np.random.normal(0, parameters.get('noise', 0), size=len(simulated)))
    return simulated

def seasonal_decomposition_plot(df):
    if df.empty:
        return
    
    try:
        df_ts = df.set_index('data')['velocidade'].asfreq('3T')
        if len(df_ts) < 480*2:
            st.warning("Dados insuficientes para decomposição sazonal")
            return
            
        decomposition = seasonal_decompose(df_ts.dropna(), model='additive', period=480)
        fig, ax = plt.subplots(4, 1, figsize=(12, 10))
        
        decomposition.observed.plot(ax=ax[0], title='Observado', color=PRIMARY_COLOR)
        decomposition.trend.plot(ax=ax[1], title='Tendência', color=ACCENT_COLOR)
        decomposition.seasonal.plot(ax=ax[2], title='Sazonalidade', color=PRIMARY_COLOR)
        decomposition.resid.plot(ax=ax[3], title='Resíduo', color=ACCENT_COLOR)
        
        for axis in ax:
            axis.tick_params(colors=TEXT_COLOR)
            axis.title.set_color(TEXT_COLOR)
            axis.set_facecolor(SECONDARY_BACKGROUND_COLOR)
        
        fig.patch.set_facecolor(BACKGROUND_COLOR)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Erro na decomposição: {str(e)}")

def create_arima_forecast(df, route_id, steps=10):
    try:
        ts_data = df.set_index('data')['velocidade'].asfreq('3T').dropna()
        if len(ts_data) < 100:
            st.warning("Dados insuficientes para previsão")
            return pd.DataFrame()
            
        model = auto_arima(ts_data, seasonal=True, m=480, 
                          error_action='ignore', suppress_warnings=True,
                          stepwise=True, random_state=42)
        
        forecast, conf_int = model.predict(n_periods=steps, return_conf_int=True)
        future_dates = pd.date_range(ts_data.index[-1], periods=steps+1, freq='3T')[1:]
        
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast,
            'yhat_lower': conf_int[:,0],
            'yhat_upper': conf_int[:,1],
            'id_route': route_id
        })
    except Exception as e:
        st.error(f"Erro na previsão: {str(e)}")
        return pd.DataFrame()

def save_forecast_to_db(forecast_df):
    try:
        engine = create_engine(
            f"mysql+mysqlconnector://{st.secrets['mysql']['user']}:{st.secrets['mysql']['password']}"
            f"@{st.secrets['mysql']['host']}/{st.secrets['mysql']['database']}"
        )
        forecast_df.rename(columns={
            'ds': 'data',
            'yhat': 'previsao',
            'yhat_lower': 'limite_inferior',
            'yhat_upper': 'limite_superior',
            'id_route': 'route_id'
        }).to_sql('forecast_history', engine, if_exists='append', index=False)
        st.success("Previsão salva com sucesso!")
    except Exception as e:
        st.error(f"Erro ao salvar previsão: {str(e)}")

def gerar_insights(df):
    insights = []
    try:
        media = df['velocidade'].mean()
        hora_pico = df.groupby('hour')['velocidade'].mean().idxmin()
        dia_semana = df.groupby('day_of_week')['velocidade'].mean().idxmin()
        
        insights.append(f"📊 **Velocidade média:** {media:.1f} km/h")
        insights.append(f"⏱️ **Hora mais lenta:** {hora_pico:02d}:00")
        insights.append(f"📅 **Dia mais lento:** {dia_semana}")
    except Exception as e:
        insights.append("❌ Erro ao gerar insights")
    return "\n\n".join(insights)

# Interface Principal
def main():
    if "mysql" not in st.secrets:
        st.error("Configure as credenciais do banco de dados!")
        return

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Controles")
        route_names = get_all_route_names()
        
        main_route = st.selectbox(
            "Rota Principal",
            route_names,
            key="main_route"
        )
        
        compare = st.checkbox("Comparar com outra rota")
        second_route = st.selectbox(
            "Rota Secundária",
            [r for r in route_names if r != main_route],
            disabled=not compare
        ) if compare else None
        
        st.date_input("Período de Análise", 
                     value=(pd.to_datetime('today')-pd.Timedelta(days=7), 
                     key="date_range")

    # Carregamento de dados
    routes_data = {}
    for route in [main_route, second_route] if compare else [main_route]:
        if not route: continue
        
        start_date = st.session_state.date_range[0].strftime('%Y-%m-%d')
        end_date = st.session_state.date_range[1].strftime('%Y-%m-%d')
        
        df, error = get_data(start_date, end_date, route)
        if error:
            st.error(f"Erro na rota {route}: {error}")
            continue
            
        if df.empty:
            st.warning(f"Sem dados para {route}")
            continue
            
        routes_data[route] = {
            'data': clean_data(df),
            'id': df['route_id'].iloc[0]
        }

    # Seção de Mapas
    st.header("🗺️ Visualização Geográfica")
    cols = st.columns(2) if compare else [st.container()]
    for idx, (route, data) in enumerate(routes_data.items()):
        with cols[idx] if compare else st:
            coords = get_route_coordinates(data['id'])
            if not coords.empty:
                fig = px.line_mapbox(coords, 
                                    lat='latitude', 
                                    lon='longitude',
                                    zoom=12,
                                    mapbox_style="carto-darkmatter")
                fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                                height=400)
                st.plotly_chart(fig, use_container_width=True)

    # Análise Temporal
    st.header("📈 Análise Temporal")
    for route, data in routes_data.items():
        with st.expander(f"Análise para {route}", expanded=True):
            st.subheader("📊 Insights")
            st.markdown(gerar_insights(data['data']))
            
            st.subheader("📉 Decomposição Sazonal")
            seasonal_decomposition_plot(data['data'])
            
            st.subheader("🔥 Heatmap de Velocidade")
            pivot = data['data'].pivot_table(
                index='day_of_week',
                columns='hour',
                values='velocidade',
                aggfunc='mean'
            )
            fig, ax = plt.subplots(figsize=(12,6))
            sns.heatmap(pivot, cmap="viridis", ax=ax)
            ax.set_title("Velocidade Média por Hora e Dia")
            st.pyplot(fig)

    # Previsão
    st.header("🔮 Previsão ARIMA")
    for route, data in routes_data.items():
        with st.expander(f"Previsão para {route}", expanded=True):
            steps = st.slider("Horizonte de Previsão (3min)", 10, 120, 60, key=f"steps_{route}")
            
            if st.button(f"Gerar Previsão para {route}"):
                forecast = create_arima_forecast(data['data'], data['id'], steps)
                if not forecast.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data['data']['data'], 
                        y=data['data']['velocidade'], 
                        name='Histórico'
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'], 
                        y=forecast['yhat'], 
                        name='Previsão'
                    ))
                    fig.add_trace(go.Scatter(
                        x=forecast['ds'].tolist()+forecast['ds'][::-1].tolist(),
                        y=forecast['yhat_upper'].tolist()+forecast['yhat_lower'][::-1].tolist(),
                        fill='toself',
                        fillcolor='rgba(100,100,255,0.2)',
                        line_color='rgba(255,255,255,0)',
                        name='Intervalo'
                    ))
                    st.plotly_chart(fig)
                    
                    if st.button(f"Salvar Previsão para {route}"):
                        save_forecast_to_db(forecast)

if __name__ == "__main__":
    main()