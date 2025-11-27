import streamlit as st
import serial
import pandas as pd
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
import time
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="Ultimate Sensor Dashboard", page_icon="🛰️")

# --- Session State Initialization ---
if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=['Time', 'Temperature', 'Humidity'])
if 'last_3d_plot_time' not in st.session_state:
    st.session_state.last_3d_plot_time = 0
if 'last_3d_fig' not in st.session_state:
    st.session_state.last_3d_fig = None

# --- Sidebar for Navigation and Settings ---
st.sidebar.title("🛰️ Dashboard Controls")

# Sidebar: Navigation
page = st.sidebar.radio(
    "Navigate", 
    ["Live Dashboard", "Data Report & Statistics"],
    label_visibility="collapsed"
)

# Sidebar: Connection & Settings
with st.sidebar.expander("⚙️ Connection & Settings", expanded=True):
    serial_port = st.text_input("Serial Port", "COM8") 
    baud_rate = st.number_input("Baud Rate", value=9600, step=100) 
    history_length = st.slider("Data History Length", 50, 1000, 150) 
    
    st.caption("App must be refreshed if port/baud is changed while running.")

# Sidebar: Plot Settings
with st.sidebar.expander("📊 Plot Settings"):
    fast_update_ms = st.number_input("Fast Update (ms)", 1000, 5000, 2000) 
    # --- THIS IS THE CHANGED LINE ---
    slow_update_s = st.number_input("3D Plot Update (s)", 5, 60, 45) 
    # --- END OF CHANGE ---

# --- Auto-Refresh ---
st_autorefresh(interval=fast_update_ms, key="data_refresher")

# --- Serial Connection ---
@st.cache_resource
def get_serial_connection(port, baud):
    try:
        ser = serial.Serial(port=port, baudrate=baud, timeout=0.1)
        return ser
    except Exception as e:
        st.error(f"Error connecting to Arduino on {port}: {e}")
        st.write("Is the Arduino plugged in? Is the port correct? Is the Serial Monitor closed?")
        st.stop()

ser = get_serial_connection(serial_port, baud_rate)

# --- Helper Functions ---
def read_and_parse_serial(ser, history_len):
    """Reads all available lines from serial and updates session state."""
    while ser.in_waiting > 0:
        try:
            line = ser.readline().decode('utf-8').rstrip()

            if line.startswith("T:") and ",H:" in line:
                parts = line.split(',')
                temp_str = parts[0].split(':')[1]
                hum_str = parts[1].split(':')[1]
                
                t = float(temp_str)
                h = float(hum_str)
                now = datetime.now()
                
                new_row = pd.DataFrame({
                    'Time': [now], 
                    'Temperature': [t], 
                    'Humidity': [h]
                })
                
                st.session_state.data = pd.concat(
                    [st.session_state.data, new_row]
                ).tail(history_len) # Use the setting from the sidebar

        except Exception as e:
            pass # Silently ignore parse errors

def get_interpretation(temp, hum):
    """Provides real-time analysis of the current conditions."""
    if pd.isna(temp) or pd.isna(hum):
        return "Waiting for data..."
    analysis = []
    if temp > 35: analysis.append("🌡️ **Extreme Heat:** Very hot conditions. Risk of heat stress.")
    elif temp > 28: analysis.append("🌡️ **Hot:** Feels warm. Good for outdoor activities.")
    elif temp < 15: analysis.append("🌡️ **Cool:** Chilly. A sweater is recommended.")
    else: analysis.append("🌡️ **Comfortable:** Ideal temperature.")
    if hum > 70: analysis.append("💧 **High Humidity:** Feels muggy and damp. Risk of mold growth.")
    elif hum < 30: analysis.append("💧 **Low Humidity:** Air is very dry. May cause dry skin.")
    else: analysis.append("💧 **Good Humidity:** Ideal indoor humidity level (40-60%).")
    return "\n".join(f"* {item}" for item in analysis)

def create_plotly_3d_plot(df):
    """Generates an interactive 3D Plotly figure."""
    if df.shape[0] < 2: return None
    
    # Using the fix for older Plotly versions (update_layout)
    fig = px.line_3d(df, 
                     x='Time', 
                     y='Temperature', 
                     z='Humidity', 
                     title="Sensor Data Over Time",
                     color='Temperature',
                     markers=True)
    
    fig.update_layout(
        coloraxis_colorscale='Bluered', # Set color scale here
        scene=dict(
            xaxis_title='Time',
            yaxis_title='Temp (°C)',
            zaxis_title='Humidity (%)'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.update_traces(marker=dict(size=3), line=dict(width=5))
    return fig

# --- Main App Logic ---
read_and_parse_serial(ser, history_length)
df = st.session_state.data

# Get data for metrics
if df.shape[0] >= 2:
    latest = df.iloc[-1]; prev = df.iloc[-2]
    temp_delta = latest['Temperature'] - prev['Temperature']
    hum_delta = latest['Humidity'] - prev['Humidity']
    current_temp = latest['Temperature']; current_hum = latest['Humidity']
elif df.shape[0] == 1:
    latest = df.iloc[-1]
    temp_delta = None; hum_delta = None
    current_temp = latest['Temperature']; current_hum = latest['Humidity']
else:
    temp_delta = None; hum_delta = None
    current_temp = pd.NA; current_hum = pd.NA

# --- === Main Page: Live Dashboard === ---
if page == "Live Dashboard":
    st.title("🔴 Live Sensor Dashboard")
    st.caption(f"Displaying last {history_length} data points. Last update: {datetime.now().strftime('%H:%M:%S')}")
    
    # Metrics in a 2x2 grid
    st.subheader("Current Conditions")
    col1, col2 = st.columns(2)
    with col1:
        m1, m2 = st.columns(2)
        m1.metric("Temperature", 
                    f"{current_temp:.1f} °C" if not pd.isna(current_temp) else "---",
                    f"{temp_delta:+.1f} °C" if temp_delta is not None else None)
        m2.metric("Humidity", 
                    f"{current_hum:.1f} %" if not pd.isna(current_hum) else "---",
                    f"{hum_delta:+.1f} %" if hum_delta is not None else None)
    with col2:
        m3, m4 = st.columns(2)
        with m3:
            if not df.empty: st.metric("Avg. Temp", f"{df['Temperature'].mean():.1f} °C")
        with m4:
            if not df.empty: st.metric("Avg. Humidity", f"{df['Humidity'].mean():.1f} %")

    st.divider()

    # --- Second Row: Analysis and Plots ---
    plot_col, analysis_col = st.columns([3, 2])
    with plot_col:
        st.subheader("Data Plots")
        tab1, tab2, tab3 = st.tabs(["📈 2D Temp vs. Time", "💧 2D Humidity vs. Time", "🧊 Interactive 3D Plot"])
        with tab1:
            st.line_chart(df.set_index('Time')['Temperature'], use_container_width=True)
        with tab2:
            st.line_chart(df.set_index('Time')['Humidity'], use_container_width=True)
        with tab3:
            st.info(f"Interactive (drag to rotate). Updates every {slow_update_s} seconds.")
            plot_placeholder = st.empty()
            current_time = time.time()
            if current_time - st.session_state.last_3d_plot_time > slow_update_s:
                fig_3d = create_plotly_3d_plot(df)
                if fig_3d:
                    st.session_state.last_3d_fig = fig_3d
                    st.session_state.last_3d_plot_time = current_time
                    plot_placeholder.plotly_chart(fig_3d, use_container_width=True)
                else:
                    plot_placeholder.warning("Not enough data to generate 3D plot.")
            else:
                if st.session_state.last_3d_fig:
                    plot_placeholder.plotly_chart(st.session_state.last_3d_fig, use_container_width=True)
                else:
                    plot_placeholder.info("Waiting for initial 3D plot generation...")
    with analysis_col:
        st.subheader("Real-Time Interpretation")
        interpretation_text = get_interpretation(current_temp, current_hum)
        st.markdown(interpretation_text)

# --- === Main Page: Data Report === ---
elif page == "Data Report & Statistics":
    st.title("📋 Data Report & Statistics")
    
    st.subheader("Full Data Log")
    st.dataframe(df.sort_values(by="Time", ascending=False), use_container_width=True) 
    
    st.subheader("Summary Statistics")
    if not df.empty:
        st.dataframe(df[['Temperature', 'Humidity']].describe(), use_container_width=True) 
    else:
        st.warning("No data collected yet.")