import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
from sklearn.ensemble import RandomForestRegressor

# --- Page Configuration ---
st.set_page_config(
    page_title="Ground Water Level Predictor",
    page_icon="💧",
    layout="wide"
)
st.title("💧 Ground Water Level Predictor Dashboard")

# --- Load and Clean Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("groundwater_master_data.csv")
    except FileNotFoundError:
        st.error("Error: 'groundwater_master_data.csv' not found.")
        st.stop()
    df.rename(columns={'LATITUDE':'lat', 'LONGITUDE':'lon'}, inplace=True)
    numeric_cols = ['lat','lon','WL(mbgl)','Block Population','Rainfall(mm/Year)']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=numeric_cols+['Date'], inplace=True)
    return df

df = load_data()

# --- Train Random Forest Model ---
@st.cache_resource
def train_model():
    features = ['Rainfall(mm/Year)','Block Population','Predominant_Soil_Type','lat','lon']
    target = 'WL(mbgl)'
    X = df[features]
    y = df[target]
    X_encoded = pd.get_dummies(X, columns=['Predominant_Soil_Type'], drop_first=True)
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_encoded, y)
    return model, X_encoded.columns

model, trained_columns = train_model()

# --- Soil Visualization ---
def create_soil_visualization(soil_type, water_depth):
    soil_color_map = {
        'Black Cotton Soil': '#2E1B0F',
        'Deep Black Soil': '#1F0F07',
        'Red Loamy and Sandy Soil': '#B7410E',
        'Red Sandy to Loamy Soil': '#C46210',
        'Laterite and Alluvial Soil': '#C7A27F',
        'Mixed Red & Black Soil': '#6D4C41',
        'Lateritic Soil': '#945438',
        'Black Cotton & Red Sandy Loam': '#5D4037'
    }
    soil_layers = [s.strip() for s in soil_type.split('&')]
    colors = [soil_color_map.get(s, '#8D6E63') for s in soil_layers]
    max_depth = 30
    depth_pct = min(100, (water_depth / max_depth) * 100)
    layer_html = ""
    layer_height = 100 / len(colors)
    for c in colors:
        layer_html += f'<div style="height:{layer_height}%; width:100%; background:{c}; position:relative;"></div>'
    html = f"""
    <div style="font-family:sans-serif; width:250px; height:350px; border:2px solid #666; border-radius:10px; background:#E3F2FD; position:relative; overflow:hidden;">
        <div style="position:absolute; top:0; width:100%; height:50px; background:#81D4FA;"></div>
        <div style="position:absolute; top:50px; width:100%; height:20px; background:#66BB6A;"></div>
        <div style="position:absolute; top:70px; width:100%; height:calc(100% - 70px);">{layer_html}
            <div style="position:absolute; top:{depth_pct}%; width:100%; height:calc(100% - {depth_pct}%); background:linear-gradient(to bottom,#4FC3F7,#0288D1);"></div>
        </div>
        <div style="position:absolute; top:75px; left:5px; font-size:14px; color:#fff; text-shadow:1px 1px 2px black;">{soil_type}</div>
        <div style="position:absolute; top:calc(70px + {depth_pct}%); right:5px; font-size:14px; color:#fff; text-shadow:1px 1px 2px black;"><b>{water_depth:.2f} m</b></div>
        <div style="position:absolute; bottom:10px; left:50%; transform:translateX(-50%); font-size:14px; color:#fff; text-shadow:1px 1px 2px black;">Water Table</div>
    </div>
    """
    return html

# --- Sidebar Filters ---
st.sidebar.header("Filters")
state_list = ["Select State"] + sorted(df["STATE_UT"].unique())
selected_state = st.sidebar.selectbox("State/UT", state_list)

if selected_state != "Select State":
    state_df = df[df["STATE_UT"]==selected_state]
    district_list = ["Select District"] + sorted(state_df["DISTRICT"].unique())
    selected_district = st.sidebar.selectbox("District", district_list)
    
    if selected_district != "Select District":
        district_df = state_df[state_df["DISTRICT"]==selected_district]
        block_list = ["Select Block"] + sorted(district_df["BLOCK"].unique())
        selected_block = st.sidebar.selectbox("Block", block_list)
        
        if selected_block != "Select Block":
            block_df = district_df[district_df["BLOCK"]==selected_block]
            village_list = ["Select Village"] + sorted(block_df["VILLAGE_NA"].unique())
            selected_village = st.sidebar.selectbox("Village", village_list)
            
            if selected_village != "Select Village":
                selection_df = block_df[block_df["VILLAGE_NA"]==selected_village].reset_index()
                if selection_df.empty:
                    st.warning("No data available for this village.")
                else:
                    kpi_data = selection_df.iloc[0]
                    st.subheader(f"Dashboard for {selected_village}, {selected_block}, {selected_district}")
                    
                    # --- Layout ---
                    col1, col2 = st.columns([2,1])
                    
                    # --- Map ---
                    with col1:
                        st.subheader("Groundwater Heatmap")
                        map_theme = st.selectbox("Select Map Theme", ["light","dark","street"])
                        view_state = pdk.ViewState(latitude=kpi_data["lat"], longitude=kpi_data["lon"], zoom=10, pitch=45)
                        block_df['weight'] = block_df['WL(mbgl)'].max() - block_df['WL(mbgl)']
                        heatmap = pdk.Layer("HeatmapLayer", data=block_df, get_position="[lon, lat]", get_weight="weight", opacity=0.7, radius_pixels=40)
                        scatter = pdk.Layer("ScatterplotLayer", data=block_df, get_position="[lon, lat]", get_fill_color=[0,128,255,100], get_radius=200, pickable=True)
                        map_styles = {'light':'light','dark':'dark','street':'road'}
                        st.pydeck_chart(pdk.Deck(map_style=map_styles[map_theme], initial_view_state=view_state, layers=[heatmap, scatter], tooltip={"html":"<b>Village:</b> {VILLAGE_NA}<br/><b>WL:</b> {WL(mbgl)} mbgl"}))
                    
                    # --- KPIs + Prediction ---
                    with col2:
                        st.subheader("Details & Prediction")
                        st.metric("Actual WL", f"{kpi_data['WL(mbgl)']:.2f} mbgl")
                        st.metric("Soil Type", kpi_data['Predominant_Soil_Type'])
                        st.metric("Rainfall", f"{kpi_data['Rainfall(mm/Year)']:.0f} mm")
                        st.metric("Population", f"{int(kpi_data['Block Population']):,}")
                        
                        # Prediction
                        input_df = pd.DataFrame([kpi_data])
                        input_encoded = pd.get_dummies(input_df, columns=['Predominant_Soil_Type'])
                        input_aligned = input_encoded.reindex(columns=trained_columns, fill_value=0)
                        pred = model.predict(input_aligned)[0]
                        st.metric("Predicted WL", f"{pred:.2f} mbgl", delta=f"{pred-kpi_data['WL(mbgl)']:.2f} m vs Actual")
                        
                        st.subheader("Predict for Custom Input")
                        soil_input = st.selectbox("Soil Type", sorted(df["Predominant_Soil_Type"].unique()))
                        rainfall_input = st.number_input("Rainfall (mm)", value=1000)
                        pop_input = st.number_input("Population", value=50000)
                        lat_input = st.number_input("Latitude", value=float(kpi_data["lat"]))
                        lon_input = st.number_input("Longitude", value=float(kpi_data["lon"]))
                        if st.button("Predict"):
                            custom_df = pd.DataFrame([{'Rainfall(mm/Year)':rainfall_input,'Block Population':pop_input,'Predominant_Soil_Type':soil_input,'lat':lat_input,'lon':lon_input}])
                            custom_enc = pd.get_dummies(custom_df, columns=['Predominant_Soil_Type'])
                            custom_aligned = custom_enc.reindex(columns=trained_columns, fill_value=0)
                            custom_pred = model.predict(custom_aligned)[0]
                            st.success(f"Predicted WL: {custom_pred:.2f} mbgl")
                            st.components.v1.html(create_soil_visualization(soil_input, custom_pred), height=370)
                    
                    st.divider()
                    
                    # --- Graphs + Soil Viz ---
                    graph_col1, graph_col2 = st.columns([2,1])
                    with graph_col1:
                        st.subheader("Water Level Trend")
                        freq_option = st.radio("View Trend:", ["Yearly","Monthly"], horizontal=True)
                        freq = 'Y' if freq_option=="Yearly" else 'M'
                        trend_df = district_df.groupby(pd.Grouper(key='Date', freq=freq)).agg({'WL(mbgl)':'mean'}).reset_index()
                        if freq=='Y':
                            trend_df['Year'] = trend_df['Date'].dt.year
                            x_field = 'Year:O'
                            tooltip_field = 'Year:O'
                        else:
                            trend_df['Month'] = trend_df['Date'].dt.to_period('M').astype(str)
                            x_field = 'Month:O'
                            tooltip_field = 'Month:O'
                        line_chart = alt.Chart(trend_df).mark_line(point=True).encode(
                            x=alt.X(x_field,title='Time'),
                            y=alt.Y('WL(mbgl):Q',title='Avg Water Level'),
                            tooltip=[tooltip_field,'WL(mbgl):Q']
                        ).interactive()
                        st.altair_chart(line_chart,use_container_width=True)
                    
                    with graph_col2:
                        st.subheader("Groundwater Depth Profile")
                        st.components.v1.html(create_soil_visualization(kpi_data['Predominant_Soil_Type'],kpi_data['WL(mbgl)']), height=370)
                    
                    st.download_button("📥 Download Filtered Data", selection_df.to_csv(index=False), "filtered_data.csv", "text/csv")
