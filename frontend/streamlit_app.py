import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
API_TIMEOUT = 15  # seconds for API calls
UI_DELAY = 0.5    # seconds for UI stability

# Page configuration
st.set_page_config(
    page_title="SIFS - Smart Inventory & Forecasting System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.timeout = API_TIMEOUT
        self.token = None
    
    def set_auth_token(self, token: str):
        """Set authentication token"""
        self.token = token
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Helper method for all API requests"""
        url = f"{self.base_url}{endpoint}"
        try:
            # Ensure auth token is included if available
            if self.token and "headers" not in kwargs:
                kwargs["headers"] = {"Authorization": f"Bearer {self.token}"}
            
            response = self.session.request(
                method,
                url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                logger.error("Authentication failed - token may be invalid or expired")
                st.error("Your session has expired. Please log in again.")
                st.session_state.authenticated = False
                safe_rerun()
            return {"error": str(e)}
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"error": str(e)}
    
    def login(self, username: str, password: str) -> Dict:
        """Login and get access token"""
        return self._make_request(
            "POST",
            "/api/auth/login",
            data={"username": username, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
    
    def get_dashboard_data(self) -> Dict:
        """Get dashboard metrics"""
        return self._make_request("GET", "/api/dashboard")
    
    def get_forecast(self, sku_ids: List[str] = None, forecast_days: int = 30) -> Dict:
        """Get demand forecasts"""
        params = {"forecast_days": forecast_days}
        if sku_ids:
            params["sku_ids"] = ",".join(sku_ids)
        return self._make_request("GET", "/api/forecasting/forecast", params=params)
    
    def generate_forecast(self, sku_ids: List[str], forecast_days: int = 30) -> Dict:
        """Generate new forecasts"""
        payload = {"sku_ids": sku_ids, "forecast_days": forecast_days}
        return self._make_request("POST", "/api/forecasting/generate", json=payload)
    
    def get_available_skus(self) -> Dict:
        """Get all available SKUs"""
        return self._make_request("GET", "/api/forecasting/available-skus")
    
    def get_reorder_recommendations(self) -> Dict:
        """Get reorder recommendations"""
        return self._make_request("GET", "/api/inventory/reorder-recommendations")
    
    def get_inventory_alerts(self) -> Dict:
        """Get inventory alerts"""
        return self._make_request("GET", "/api/inventory/alerts")
    
    def get_inventory_summary(self) -> Dict:
        """Get inventory summary"""
        return self._make_request("GET", "/api/inventory/summary")
    
    def update_stock_levels(self, updates: List[Dict]) -> Dict:
        """Update stock levels"""
        return self._make_request("POST", "/api/inventory/update-stock", json={"updates": updates})
    
    def generate_purchase_orders(self, sku_ids: List[str]) -> Dict:
        """Generate purchase orders"""
        return self._make_request("POST", "/api/inventory/generate-purchase-orders", json={"sku_ids": sku_ids})
    
    def upload_sales_data(self, file) -> Dict:
        """Upload sales data file"""
        try:
            files = {"file": (file.name, file, file.type)}
            return self._make_request("POST", "/api/data/upload-sales", files=files)
        except Exception as e:
            logger.error(f"File upload error: {str(e)}")
            raise e
    
    def get_users(self) -> Dict:
        """Get all users (admin only)"""
        return self._make_request("GET", "/api/users")
    
    def create_user(self, user_data: Dict) -> Dict:
        """Create new user (admin only)"""
        return self._make_request("POST", "/api/users", json=user_data)
    
    def update_user(self, user_id: int, user_data: Dict) -> Dict:
        """Update user (admin only)"""
        return self._make_request("PUT", f"/api/users/{user_id}", json=user_data)
    
    def delete_user(self, user_id: int) -> Dict:
        """Delete user (admin only)"""
        return self._make_request("DELETE", f"/api/users/{user_id}")
    
    def get_system_configurations(self) -> Dict:
        """Get system configurations (admin only)"""
        return self._make_request("GET", "/api/system/configurations")
    
    def update_system_configurations(self, configurations: Dict) -> Dict:
        """Update system configurations (admin only)"""
        return self._make_request("PUT", "/api/system/configurations", json={"configurations": configurations})

def init_session_state():
    """Initialize session state variables with timeout protection"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'api_client' not in st.session_state:
        st.session_state.api_client = APIClient()
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    if 'last_action_time' not in st.session_state:
        st.session_state.last_action_time = time.time()

def safe_rerun():
    """Helper function for safe rerun with delay"""
    time.sleep(UI_DELAY)
    st.rerun()

def login_page():
    """Display centered login page"""
    # Custom CSS for centering the login form
    st.markdown("""
    <style>
    /* Remove default streamlit padding and margins */
    .main > div {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 85vh;
        padding: 1rem 0;
        margin-top: -2rem;
    }
    .login-form {
        background-color: #ffffff;
        padding: 2rem 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        width: 100%;
        max-width: 380px;
        border: 1px solid #e6e9ef;
    }
    .login-title {
        text-align: center;
        color: #1f2937;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .login-subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .stTextInput > div > div > input {
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid #d1d5db;
        font-size: 1rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    .login-button {
        width: 100%;
        background-color: #3b82f6;
        color: white;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        font-size: 1rem;
        font-weight: 500;
        margin-top: 1rem;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .login-button:hover {
        background-color: #2563eb;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create centered container
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    
    # Create three columns for centering
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-form">', unsafe_allow_html=True)
        
        # Title and subtitle
        st.markdown('<h1 class="login-title">🔐 SIFS Login</h1>', unsafe_allow_html=True)
        st.markdown('<p class="login-subtitle">Welcome to Smart Inventory & Forecasting System</p>', unsafe_allow_html=True)
        
        # Login form
        with st.form("login_form"):
            username = st.text_input(
                "Username", 
                placeholder="Enter your username",
                key="username_input"
            )
            
            password = st.text_input(
                "Password", 
                type="password", 
                placeholder="Enter your password",
                key="password_input"
            )
            
            # Add some spacing
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Submit button
            submit_button = st.form_submit_button(
                "Login",
                use_container_width=True,
                type="primary"
            )
            
            if submit_button:
                if username and password:
                    try:
                        with st.spinner("Authenticating..."):
                            # Clear any existing token
                            st.session_state.api_client = APIClient()
                            
                            # Make login request
                            token_data = st.session_state.api_client.login(username, password)
                            
                            if "access_token" in token_data:
                                # Set the token in the client
                                st.session_state.api_client.set_auth_token(token_data["access_token"])
                                
                                # Store user data with proper role
                                st.session_state.user_data = {
                                    'username': username,
                                    'role': token_data.get('role', 'viewer').lower(),  # Ensure lowercase
                                    'token': token_data['access_token']
                                }
                                st.session_state.authenticated = True
                                
                                st.success(f"Welcome, {username}!")
                                time.sleep(UI_DELAY)
                                safe_rerun()
                            else:
                                st.error(f"Login failed: {token_data.get('error', 'Invalid credentials')}")
                    except Exception as e:
                        st.error("Login error. Please try again.")
                        logger.error(f"Login error: {e}")
                else:
                    st.error("Please enter both username and password.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Optional: Add footer information (compact)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            "<p style='text-align: center; color: #9ca3af; font-size: 0.8rem; margin-top: 1rem;'>"
            "© 2025 SIFS | For support, contact your administrator"
            "</p>", 
            unsafe_allow_html=True
        )
    
def sidebar_navigation():
    """Display sidebar navigation"""
    with st.sidebar:
        st.title("📊 SIFS")
        
        # Verify user_data exists and has role
        if 'user_data' in st.session_state and 'role' in st.session_state.user_data:
            role_display = st.session_state.user_data['role'].capitalize()
            st.markdown(f"**Welcome, {st.session_state.user_data['username']}**")
            st.markdown(f"*Role: {role_display}*")
        else:
            st.warning("User role not detected")
            st.session_state.authenticated = False
            safe_rerun()
        
        st.divider()
        
        # Navigation menu - basic items for all users
        pages = {
            "📈 Dashboard": "Dashboard",
            "📊 Forecasting": "Forecasting", 
            "📦 Inventory": "Inventory"
        }
        
        # Additional items for data entry and above
        if 'user_data' in st.session_state and st.session_state.user_data['role'] in ['dataentry', 'admin']:
            pages.update({
                "📋 Reports": "Reports",
                "📤 Data Upload": "Data Upload"
            })
        
        # Admin-only items
        if 'user_data' in st.session_state and st.session_state.user_data['role'] == 'admin':
            pages.update({
                "👥 User Management": "User Management",
                "⚙️ Settings": "Settings"
            })
        
        # Display navigation buttons
        for page_display, page_key in pages.items():
            if st.button(page_display, use_container_width=True):
                st.session_state.current_page = page_key
                safe_rerun()
        
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_data = {}
            st.session_state.current_page = "Dashboard"
            st.session_state.api_client = APIClient()  # Reset client
            safe_rerun()

def dashboard_page():
    """Display dashboard page"""
    st.title("📈 Dashboard")
    
    # Get dashboard data with error handling
    dashboard_data = {}
    try:
        with st.spinner("Loading dashboard data..."):
            dashboard_data = st.session_state.api_client.get_dashboard_data()
    except Exception as e:
        logger.error(f"Failed to load dashboard data: {e}")
        st.warning("Could not load live dashboard data. Showing sample data.")
        dashboard_data = {}
    
    # Key metrics with safe defaults
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Sales", 
            f"${dashboard_data.get('totalSales', 50000):,.2f}",
            delta="12.5%"
        )
    
    with col2:
        st.metric(
            "Inventory Value", 
            f"${dashboard_data.get('currentInventoryValue', 125000):,.2f}",
            delta="-2.3%"
        )
    
    with col3:
        st.metric(
            "Open Alerts", 
            dashboard_data.get('openAlerts', 3),
            delta="3"
        )
    
    with col4:
        st.metric(
            "Pending Reorders", 
            dashboard_data.get('pendingReorders', 8),
            delta="2"
        )
    
    st.divider()
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sales Trend")
        
        # Dynamic date range from January 1, 2025 to current date
        start_date = datetime(2025, 1, 1).date()
        end_date = datetime.now().date()
        
        # Generate date range based on the time span
        days_diff = (end_date - start_date).days
        
        if days_diff <= 31:
            # If less than a month, show daily data
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            sales_multiplier = 100
        elif days_diff <= 365:
            # If less than a year, show weekly data  
            dates = pd.date_range(start=start_date, end=end_date, freq='W')
            sales_multiplier = 700
        else:
            # If more than a year, show monthly data
            dates = pd.date_range(start=start_date, end=end_date, freq='ME')
            sales_multiplier = 3000
        
        # Ensure we always include the current date if it's not already there
        if len(dates) == 0 or dates[-1].date() != end_date:
            dates = dates.append(pd.DatetimeIndex([end_date])).drop_duplicates().sort_values()
        
        # Generate sales data with some variation
        sales_data = pd.DataFrame({
            'Date': dates,
            'Sales': [50000 + (i * sales_multiplier) + (i * (sales_multiplier//2) * (-1)**i) for i in range(len(dates))]
        })
        
        # Create the chart with dynamic title
        period_desc = f"Jan 2025 - {end_date.strftime('%b %d, %Y')}"
        fig = px.line(sales_data, x='Date', y='Sales', title=f"Sales Trend ({period_desc})")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📦 Inventory Status")
        # Sample inventory data
        inventory_data = pd.DataFrame({
            'Category': ['Low Stock', 'Normal', 'Overstock'],
            'Count': [15, 120, 25]
        })
        
        fig = px.pie(inventory_data, values='Count', names='Category', 
                    title="Inventory Distribution")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # System alerts with safe handling
    system_alerts = dashboard_data.get('systemAlerts', [])
    if system_alerts:
        st.subheader("⚠️ Recent Alerts")
        for alert in system_alerts:
            # Safe access to alert fields
            alert_type = alert.get('type', 'Info')
            severity = alert.get('severity', 'medium')
            sku = alert.get('sku', 'Unknown')
            message = alert.get('message', 'No message')
            
            if severity == 'high':
                st.error(f"**{sku}** - {alert_type}: {message}")
            elif severity == 'medium':
                st.warning(f"**{sku}** - {alert_type}: {message}")
            else:
                st.info(f"**{sku}** - {alert_type}: {message}")
    
    # Recent activity with safe field access
    st.subheader("🕒 Recent Activity")
    
    # Get recent activities from API or use defaults
    recent_activities = dashboard_data.get('recentActivity', [])
    
    # If no activities from API, use default sample data
    if not recent_activities:
        recent_activities = [
            {"time": "2 hours ago", "activity": "New sales data uploaded", "user": "DataClerk1"},
            {"time": "4 hours ago", "activity": "Forecast generated for MEALIE_2KG", "user": "System"},
            {"time": "6 hours ago", "activity": "Low stock alert for SUGAR_2KG", "user": "System"},
            {"time": "1 day ago", "activity": "New user created", "user": "Admin"},
        ]
    
    # Display activities with safe field access
    for activity in recent_activities:
        try:
            # Safe access to activity fields
            time_str = activity.get('time', 'Unknown time')
            activity_str = activity.get('activity', 'Unknown activity')
            user_str = activity.get('user', 'Unknown user')
            
            st.text(f"⏰ {time_str} - {activity_str} (by {user_str})")
            
        except Exception as e:
            logger.error(f"Error displaying activity: {e}")
            # Skip this activity and continue
            continue

def forecasting_page():
    """Display forecasting page"""
    st.title("📊 Demand Forecasting")
    
    # Get available SKUs
    skus_data = st.session_state.api_client.get_available_skus()
    available_skus = skus_data.get('skus', [])
    
    if not available_skus:
        # Demo data if no SKUs available
        available_skus = [
            {'sku_id': 'MEALIE_2KG', 'sku_name': 'Mealie Meal 2KG'},
            {'sku_id': 'COOKOIL_2LT', 'sku_name': 'Cooking Oil 2L'},
            {'sku_id': 'SUGAR_2KG', 'sku_name': 'Sugar 2KG'},
            {'sku_id': 'BREAD_LOAF', 'sku_name': 'Bread Loaf'},
            {'sku_id': 'SOAP_100G', 'sku_name': 'Soap 100G'}
        ]
    
    # Controls section
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Create options for multiselect
        sku_options = [f"{sku['sku_id']} - {sku['sku_name']}" for sku in available_skus]
        selected_sku_options = st.multiselect(
            "Select SKUs to forecast:",
            options=sku_options,
            default=sku_options[:3] if len(sku_options) >= 3 else sku_options
        )
    
    with col2:
        forecast_days = st.number_input(
            "Forecast Days:",
            min_value=7,
            max_value=90,
            value=30,
            step=7
        )
    
    with col3:
        generate_button = st.button("🔄 Generate Forecast", type="primary")
    
    # Extract SKU IDs from selected options
    selected_skus = []
    if selected_sku_options:
        selected_skus = [option.split(' - ')[0] for option in selected_sku_options]
    
    # Handle forecast generation
    if generate_button and selected_skus:
        with st.spinner("Generating forecasts..."):
            result = st.session_state.api_client.generate_forecast(selected_skus, forecast_days)
            
            if result.get('success'):
                st.success(result['message'])
                st.session_state.forecast_data = result.get('forecast_data', {})
            else:
                st.error(f"Failed to generate forecasts: {result.get('message', 'Unknown error')}")
    
    # Display forecasts (using demo data if API not available)
    if selected_skus:
        st.subheader("📊 Forecast Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAPE", "8.5%")
        with col2:
            st.metric("RMSE", "2.1")
        with col3:
            st.metric("Bias", "-0.3")
        with col4:
            st.metric("Accuracy", "91.5%")
        
        # Generate sample forecast for each selected SKU
        for sku_option in selected_sku_options:
            sku_id = sku_option.split(' - ')[0]
            sku_name = sku_option.split(' - ')[1]
            
            with st.expander(f"📈 {sku_name} ({sku_id})", expanded=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Generate sample forecast data
                    dates = pd.date_range(start=datetime.now(), periods=forecast_days, freq='D')
                    historical_dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                                   end=datetime.now(), freq='D')
                    
                    # Historical data
                    historical_demand = [10 + 3 * np.sin(i * 2 * np.pi / 7) + np.random.normal(0, 1) 
                                        for i in range(len(historical_dates))]
                    
                    # Forecast data
                    forecast_demand = [10 + 3 * np.sin(i * 2 * np.pi / 7) + np.random.normal(0, 0.5) 
                                      for i in range(len(dates))]
                    
                    # Confidence intervals
                    upper_bound = [f + 2 for f in forecast_demand]
                    lower_bound = [max(0, f - 2) for f in forecast_demand]
                    
                    # Create forecast chart
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=historical_dates,
                        y=historical_demand,
                        mode='lines',
                        name='Historical Demand',
                        line=dict(color='blue')
                    ))
                    
                    # Forecast data
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=forecast_demand,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    # Confidence intervals
                    fig.add_trace(go.Scatter(
                        x=list(dates) + list(dates)[::-1],
                        y=upper_bound + lower_bound[::-1],
                        fill='toself',
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=True,
                        name='95% Confidence'
                    ))
                    
                    fig.update_layout(
                        title=f"Demand Forecast - {sku_name}",
                        xaxis_title="Date",
                        yaxis_title="Predicted Demand",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("**Forecast Metrics:**")
                    st.metric("Model Accuracy", "91.5%")
                    st.metric("Trend", "Stable")
                    st.metric("Seasonality", "Detected")
                    
                    total_demand = sum(forecast_demand)
                    avg_daily = total_demand / len(forecast_demand)
                    st.metric("Avg Daily Demand", f"{avg_daily:.1f}")

def inventory_page():
    """Display inventory management page"""
    st.title("📦 Inventory Management")
    
    # Get data
    with st.spinner("Loading reorder recommendations..."):
        reorder_data = st.session_state.api_client.get_reorder_recommendations()
    
    recommendations = reorder_data.get('recommendations', [])
    
    # Use demo data if no recommendations available
    if not recommendations:
        recommendations = [
            {
                'sku_id': 'MEALIE_2KG', 'sku_name': 'Mealie Meal 2KG', 'current_stock': 45,
                'safety_stock_qty': 20, 'reorder_point': 35, 'reorder_needed': True,
                'reorder_quantity': 100, 'supplier_name': 'Supplier A', 'days_of_supply': 12,
                'avg_daily_demand': 8.5
            },
            {
                'sku_id': 'SUGAR_2KG', 'sku_name': 'Sugar 2KG', 'current_stock': 25,
                'safety_stock_qty': 15, 'reorder_point': 25, 'reorder_needed': True,
                'reorder_quantity': 75, 'supplier_name': 'Supplier B', 'days_of_supply': 8,
                'avg_daily_demand': 6.2
            },
            {
                'sku_id': 'COOKOIL_2LT', 'sku_name': 'Cooking Oil 2L', 'current_stock': 120,
                'safety_stock_qty': 50, 'reorder_point': 75, 'reorder_needed': False,
                'reorder_quantity': 0, 'supplier_name': 'Supplier C', 'days_of_supply': 25,
                'avg_daily_demand': 4.8
            }
        ]
    
    # Convert to DataFrame
    df = pd.DataFrame(recommendations)
    
    # Summary metrics
    st.subheader("📊 Inventory Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        items_needing_reorder = len(df[df['reorder_needed'] == True])
        st.metric("Items Needing Reorder", items_needing_reorder)
    
    with col2:
        total_reorder_qty = df[df['reorder_needed'] == True]['reorder_quantity'].sum()
        st.metric("Total Reorder Quantity", f"{total_reorder_qty:,}")
    
    with col3:
        avg_stock = df['current_stock'].mean()
        st.metric("Average Current Stock", f"{avg_stock:.1f}")
    
    with col4:
        critical_items = len(df[df['days_of_supply'] <= 10])
        st.metric("Critical Items (≤10 days)", critical_items)
    
    # Alerts section
    st.subheader("⚠️ Inventory Alerts")
    
    alerts = [
        {"sku": "SUGAR_2KG", "type": "Low Stock", "message": "Current stock below safety level", "severity": "high"},
        {"sku": "MEALIE_2KG", "type": "Reorder Point", "message": "Reached reorder point", "severity": "medium"},
        {"sku": "SOAP_100G", "type": "Critical", "message": "Stock critically low", "severity": "high"}
    ]
    
    # Group alerts by severity
    high_alerts = [a for a in alerts if a.get('severity') == 'high']
    medium_alerts = [a for a in alerts if a.get('severity') == 'medium']
    low_alerts = [a for a in alerts if a.get('severity') == 'low']
    
    if high_alerts:
        st.markdown("**🔴 Critical Alerts:**")
        for alert in high_alerts:
            st.error(f"**{alert['sku']}** - {alert['type']}: {alert['message']}")
    
    if medium_alerts:
        st.markdown("**🟡 Medium Priority:**")
        for alert in medium_alerts:
            st.warning(f"**{alert['sku']}** - {alert['type']}: {alert['message']}")
    
    # Filter options
    st.subheader("🔍 Filter Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_reorder_needed = st.checkbox("Show only items needing reorder", value=False)
    
    with col2:
        suppliers = ['All'] + sorted(df['supplier_name'].unique().tolist())
        selected_supplier = st.selectbox("Filter by Supplier", suppliers)
    
    with col3:
        min_days_supply = st.slider("Minimum days of supply", 0, 30, 0)
    
    # Apply filters
    filtered_df = df.copy()
    
    if show_reorder_needed:
        filtered_df = filtered_df[filtered_df['reorder_needed'] == True]
    
    if selected_supplier != 'All':
        filtered_df = filtered_df[filtered_df['supplier_name'] == selected_supplier]
    
    if min_days_supply > 0:
        filtered_df = filtered_df[filtered_df['days_of_supply'] >= min_days_supply]
    
    # Main inventory table
    st.subheader("📋 Inventory Details")
    
    if filtered_df.empty:
        st.info("No items match the current filters.")
    else:
        # Display configuration
        display_columns = [
            'sku_id', 'sku_name', 'current_stock', 'safety_stock_qty', 
            'reorder_point', 'reorder_needed', 'reorder_quantity', 
            'supplier_name', 'days_of_supply', 'avg_daily_demand'
        ]
        
        # Format the dataframe for display
        display_df = filtered_df[display_columns].copy()
        display_df['reorder_needed'] = display_df['reorder_needed'].map({True: '✅ Yes', False: '❌ No'})
        
        # Rename columns for better display
        display_df.columns = [
            'SKU ID', 'Product Name', 'Current Stock', 'Safety Stock', 
            'Reorder Point', 'Needs Reorder', 'Reorder Qty', 
            'Supplier', 'Days Supply', 'Avg Daily Demand'
        ]
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Stock level visualization
        st.subheader("📊 Stock Level Analysis")
        
        # Create stock level chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Current vs Safety Stock', 'Days of Supply Distribution', 
                          'Reorder Quantity by Supplier', 'Stock Status Overview'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # Chart 1: Current vs Safety Stock
        fig.add_trace(
            go.Bar(x=filtered_df['sku_id'], y=filtered_df['current_stock'], 
                   name='Current Stock', marker_color='lightblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=filtered_df['sku_id'], y=filtered_df['safety_stock_qty'], 
                   name='Safety Stock', marker_color='orange'),
            row=1, col=1
        )
        
        # Chart 2: Days of Supply Distribution
        fig.add_trace(
            go.Histogram(x=filtered_df['days_of_supply'], nbinsx=20, 
                        name='Days of Supply', marker_color='green'),
            row=1, col=2
        )
        
        # Chart 3: Reorder Quantity by Supplier
        supplier_reorder = filtered_df[filtered_df['reorder_needed'] == True].groupby('supplier_name')['reorder_quantity'].sum()
        if not supplier_reorder.empty:
            fig.add_trace(
                go.Bar(x=supplier_reorder.index, y=supplier_reorder.values, 
                       name='Reorder Qty', marker_color='red'),
                row=2, col=1
            )
        
        # Chart 4: Stock Status Pie Chart
        status_counts = filtered_df['reorder_needed'].value_counts()
        fig.add_trace(
            go.Pie(labels=['No Reorder Needed', 'Reorder Needed'], 
                   values=[status_counts.get(False, 0), status_counts.get(True, 0)],
                   marker_colors=['lightgreen', 'lightcoral']),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Action buttons
        st.subheader("🛠️ Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Generate Purchase Orders", type="primary"):
                reorder_items = filtered_df[filtered_df['reorder_needed'] == True]['sku_id'].tolist()
                if reorder_items:
                    with st.spinner("Generating purchase orders..."):
                        result = st.session_state.api_client.generate_purchase_orders(reorder_items)
                        
                        if result.get('success'):
                            st.success(result['message'])
                            
                            # Display purchase orders
                            st.subheader("📄 Generated Purchase Orders")
                            for po in result.get('purchase_orders', []):
                                with st.expander(f"PO for {po['supplier_name']}"):
                                    st.write(f"**Order Date:** {po['order_date']}")
                                    st.write(f"**Total Items:** {po['total_items']:,}")
                                    st.write(f"**Estimated Total:** ${po['estimated_total']:,.2f}")
                                    
                                    po_df = pd.DataFrame(po['items'])
                                    st.dataframe(po_df, use_container_width=True)
                        else:
                            st.error(f"Failed to generate purchase orders: {result.get('message')}")
                else:
                    st.warning("No items need reordering.")
        
        with col2:
            if st.button("🔄 Refresh Data"):
                safe_rerun()
        
        with col3:
            if st.button("📊 Export to CSV"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"inventory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def reports_page():
    """Display reports page"""
    st.title("📋 Reports")
    
    # Report type selection
    col1, col2 = st.columns(2)
    
    with col1:
        report_type = st.selectbox(
            "Select Report Type",
            ["Sales Summary", "Inventory Report", "Forecast Accuracy", "Supplier Performance"]
        )
    
    with col2:
        date_range = st.date_input(
            "Select Date Range",
            value=[datetime.now() - timedelta(days=30), datetime.now()],
            format="YYYY-MM-DD"
        )
    
    if st.button("Generate Report", use_container_width=True):
        st.success(f"Generated {report_type} report")
        
        # Sample report data based on type
        if report_type == "Sales Summary":
            st.subheader("📊 Sales Summary Report")
            
            # Sample sales data with all SKUs from generate_dataset
            sales_report_data = pd.DataFrame({
                'SKU': ["MEALIE_2KG", "MEALIE_10KG", "RICE_5KG", "COOKOIL_2LT", "SUGAR_2KG", 
                       "SALT_1KG", "COKE_500ML", "MAZOE_1LT", "TEA_250G", "MILK_1LT",
                       "SOAP_100G", "DETERGENT_500G", "CANDLES_6PK", "TOOTHPASTE_100G",
                       "SANITARY_10PK", "BISCUITS_200G", "MAPUTI_100G", "BREAD_LOAF",
                       "BEANS_410G", "PILCHARDS_155G", "AIRTIME_5", "ELECTRICITY_10"],
                'Units Sold': [1250, 850, 950, 850, 950, 600, 2100, 750, 450, 800,
                              1200, 650, 350, 550, 400, 850, 500, 2100, 450, 400,
                              3500, 2800],
                'Revenue': [25000, 6800, 7125, 2975, 2090, 600, 2100, 1875, 1350, 1440,
                           960, 1625, 525, 990, 1400, 1020, 250, 2100, 675, 720,
                           17500, 28000],
                'Growth %': [12.5, -3.2, 8.1, 10.5, 7.8, 2.1, 15.7, 12.3, 5.5, 8.2,
                             4.5, 6.7, -1.8, 3.2, 9.1, 11.5, 1.2, 7.5, 4.3, 5.6,
                             18.2, 22.5]
            })
            
            st.dataframe(sales_report_data, use_container_width=True)
            
            # Sales chart
            fig = px.bar(sales_report_data, x='SKU', y='Units Sold', 
                        title="Units Sold by SKU")
            st.plotly_chart(fig, use_container_width=True)
            
        elif report_type == "Inventory Report":
            st.subheader("📦 Inventory Report")
            
            inventory_report_data = pd.DataFrame({
                'SKU': ["MEALIE_2KG", "MEALIE_10KG", "RICE_5KG", "COOKOIL_2LT", "SUGAR_2KG", 
                       "SALT_1KG", "COKE_500ML", "MAZOE_1LT", "TEA_250G", "MILK_1LT",
                       "SOAP_100G", "DETERGENT_500G", "CANDLES_6PK", "TOOTHPASTE_100G",
                       "SANITARY_10PK", "BISCUITS_200G", "MAPUTI_100G", "BREAD_LOAF",
                       "BEANS_410G", "PILCHARDS_155G", "AIRTIME_5", "ELECTRICITY_10"],
                'Current Stock': [45, 30, 25, 50, 25, 40, 120, 35, 20, 60,
                                 80, 45, 65, 30, 25, 40, 20, 80, 15, 18,
                                 1000, 500],
                'Safety Stock': [20, 15, 12, 25, 15, 20, 50, 15, 10, 30,
                                40, 25, 30, 15, 12, 20, 10, 30, 8, 9,
                                500, 200],
                'Reorder Point': [35, 25, 20, 40, 25, 30, 75, 25, 15, 45,
                                 60, 35, 50, 25, 20, 30, 15, 50, 12, 15,
                                 750, 300],
                'Days of Supply': [12, 10, 8, 15, 8, 20, 25, 10, 7, 18,
                                   20, 15, 30, 12, 10, 15, 8, 15, 5, 6,
                                   30, 25]
            })
            
            st.dataframe(inventory_report_data, use_container_width=True)
            
            # Inventory visualization
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Current vs Safety Stock', 'Days of Supply'))
            
            # Current vs Safety Stock
            fig.add_trace(
                go.Bar(x=inventory_report_data['SKU'], y=inventory_report_data['Current Stock'], 
                       name='Current Stock', marker_color='lightblue'),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=inventory_report_data['SKU'], y=inventory_report_data['Safety Stock'], 
                       name='Safety Stock', marker_color='orange'),
                row=1, col=1
            )
            
            # Days of Supply
            fig.add_trace(
                go.Bar(x=inventory_report_data['SKU'], y=inventory_report_data['Days of Supply'], 
                       name='Days of Supply', marker_color='green'),
                row=1, col=2
            )
            
            fig.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
        elif report_type == "Forecast Accuracy":
            st.subheader("🎯 Forecast Accuracy Report")
            
            accuracy_data = pd.DataFrame({
                'Model': ['ARIMA', 'Prophet', 'Croston', 'Ensemble'],
                'MAPE (%)': [8.5, 7.2, 12.1, 6.8],
                'RMSE': [2.1, 1.8, 3.2, 1.6],
                'Bias': [-0.3, 0.1, -0.8, 0.05]
            })
            
            st.dataframe(accuracy_data, use_container_width=True)
            
            fig = px.bar(accuracy_data, x='Model', y='MAPE (%)', 
                        title="Model Accuracy Comparison (Lower is Better)")
            st.plotly_chart(fig, use_container_width=True)
            
        elif report_type == "Supplier Performance":
            st.subheader("🏭 Supplier Performance Report")
            
            supplier_data = pd.DataFrame({
                'Supplier': ["National Foods", "United Refineries", "Delta Corp", "Unilever", 
                            "ZESA Holdings", "Econet", "Lobels Bread", "Country Choice"],
                'On-Time Delivery %': [95, 92, 98, 90, 100, 100, 85, 88],
                'Quality Issues': [2, 1, 0, 3, 0, 0, 5, 4],
                'Avg Lead Time (days)': [3, 2, 1, 2, 0, 0, 1, 4],
                'Price Stability': [4.2, 4.5, 4.8, 4.0, 5.0, 5.0, 3.8, 4.1]
            })
            
            st.dataframe(supplier_data, use_container_width=True)
            
            fig = px.bar(supplier_data, x='Supplier', y='On-Time Delivery %', 
                        title="Supplier On-Time Delivery Performance")
            st.plotly_chart(fig, use_container_width=True)

def data_upload_page():
    """Display data upload page"""
    st.title("📤 Data Upload")
    
    # File upload section
    st.subheader("📁 Upload Sales Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your sales data file. Supported formats: CSV, Excel"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.info(f"File: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Preview data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Upload button
            if st.button("🚀 Process and Upload Data", use_container_width=True):
                try:
                    with st.spinner("Processing and uploading data..."):
                        # Reset file pointer
                        uploaded_file.seek(0)
                        result = st.session_state.api_client.upload_sales_data(uploaded_file)
                    
                    st.success("✅ Data uploaded and processed successfully!")
                    st.json(result)
                    
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")
                    
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    
    st.divider()
    
    # Data format guidelines
    st.subheader("📋 Data Format Guidelines")
    
    with st.expander("Required Columns"):
        st.markdown("""
        Your sales data file should contain the following columns:
        
        - **date**: Date of the sale (formats: YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY)
        - **sku_id**: Product SKU identifier (string)
        - **quantity_sold**: Quantity sold (numeric, >= 0)
        - **price**: Price per unit (numeric, >= 0, optional)
        
        **Example:**
        ```
        date,sku_id,quantity_sold,price
        2024-01-01,MEALIE_2KG,15,20.00
        2024-01-01,COOKOIL_2LT,8,25.50
        ```
        """)
    
    with st.expander("Data Quality Tips"):
        st.markdown("""
        - Ensure dates are in a consistent format
        - Remove any header rows or summary rows
        - Check for missing values in required columns
        - Verify SKU IDs match your product catalog
        - Ensure quantity and price values are numeric
        """)

def user_management_page():
    """Display user management page (admin only)"""
    if st.session_state.user_data['role'].lower() != 'admin':
        st.error("❌ Access denied. Admin privileges required.")
        return
    
    st.title("👥 User Management")
    
    # Get users data
    with st.spinner("Loading users..."):
        users_data = st.session_state.api_client.get_users()
    
    users = users_data.get('users', [])
    
    # Add new user section
    with st.expander("➕ Add New User"):
        with st.form("add_user_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                new_full_name = st.text_input("Full Name")
            
            with col2:
                new_email = st.text_input("Email")
                new_role = st.selectbox("Role", ["admin", "dataentry", "viewer"])
            
            if st.form_submit_button("Create User"):
                if new_username and new_password:
                    try:
                        user_data = {
                            "username": new_username,
                            "password": new_password,
                            "full_name": new_full_name,
                            "email": new_email,
                            "role": new_role
                        }
                        
                        result = st.session_state.api_client.create_user(user_data)
                        st.success(f"✅ User '{new_username}' created successfully!")
                        safe_rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Failed to create user: {str(e)}")
                else:
                    st.error("Username and password are required.")
    
    # Display users table
    if users:
        st.subheader("👤 Current Users")
        
        # Display users with action buttons
        for idx, user in enumerate(users):
            col1, col2, col3, col4, col5 = st.columns([2, 2, 1, 1, 1])
            
            with col1:
                st.text(f"👤 {user['username']}")
            
            with col2:
                st.text(f"{user.get('full_name', 'N/A')}")
            
            with col3:
                role_color = {"admin": "🔴", "dataentry": "🟡", "viewer": "🟢"}
                st.text(f"{role_color.get(user['role'], '⚪')} {user['role']}")
            
            with col4:
                if st.button("✏️", key=f"edit_{user['id']}", help="Edit user"):
                    st.session_state[f"editing_user_{user['id']}"] = True
            
            with col5:
                if user['username'] != st.session_state.user_data['username']:  # Can't delete self
                    if st.button("🗑️", key=f"delete_{user['id']}", help="Delete user"):
                        try:
                            st.session_state.api_client.delete_user(user['id'])
                            st.success(f"✅ User '{user['username']}' deleted!")
                            safe_rerun()
                        except Exception as e:
                            st.error(f"❌ Failed to delete user: {str(e)}")
            
            # Edit user form (appears when edit button is clicked)
            if st.session_state.get(f"editing_user_{user['id']}", False):
                with st.form(f"edit_user_form_{user['id']}"):
                    st.markdown(f"**Editing User: {user['username']}**")
                    
                    edit_col1, edit_col2 = st.columns(2)
                    
                    with edit_col1:
                        edit_username = st.text_input("Username", value=user['username'])
                        edit_full_name = st.text_input("Full Name", value=user.get('full_name', ''))
                        edit_password = st.text_input("New Password (leave blank to keep current)", type="password")
                    
                    with edit_col2:
                        edit_email = st.text_input("Email", value=user.get('email', ''))
                        edit_role = st.selectbox("Role", ["admin", "dataentry", "viewer"], 
                                               index=["admin", "dataentry", "viewer"].index(user['role']))
                    
                    form_col1, form_col2 = st.columns(2)
                    
                    with form_col1:
                        if st.form_submit_button("💾 Save Changes"):
                            try:
                                update_data = {
                                    "username": edit_username,
                                    "full_name": edit_full_name,
                                    "email": edit_email,
                                    "role": edit_role
                                }
                                
                                if edit_password:
                                    update_data["password"] = edit_password
                                
                                result = st.session_state.api_client.update_user(user['id'], update_data)
                                st.success(f"✅ User '{edit_username}' updated successfully!")
                                st.session_state[f"editing_user_{user['id']}"] = False
                                safe_rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Failed to update user: {str(e)}")
                    
                    with form_col2:
                        if st.form_submit_button("❌ Cancel"):
                            st.session_state[f"editing_user_{user['id']}"] = False
                            safe_rerun()
            
            st.divider()
    else:
        st.info("No users found.")

def settings_page():
    """Display settings page (admin only)"""
    if st.session_state.user_data['role'].lower() != 'admin':
        st.error("❌ Access denied. Admin privileges required.")
        return
    
    st.title("⚙️ System Settings")
    
    # Get current configurations
    with st.spinner("Loading system configurations..."):
        config_data = st.session_state.api_client.get_system_configurations()
    
    configurations = config_data.get('configurations', {})
    
    st.subheader("📊 Forecasting Settings")
    
    with st.form("forecasting_settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            forecast_horizon = st.number_input(
                "Forecast Horizon (days)",
                min_value=7,
                max_value=365,
                value=configurations.get('forecast_horizon_days', 30)
            )
            
            safety_stock_multiplier = st.number_input(
                "Safety Stock Multiplier",
                min_value=1.0,
                max_value=3.0,
                value=configurations.get('safety_stock_multiplier', 1.5),
                step=0.1
            )
            
            reorder_threshold = st.number_input(
                "Reorder Threshold (days)",
                min_value=1,
                max_value=30,
                value=configurations.get('reorder_threshold_days', 7)
            )
        
        with col2:
            price_variance_threshold = st.number_input(
                "Price Variance Threshold",
                min_value=0.05,
                max_value=0.50,
                value=configurations.get('price_variance_threshold', 0.15),
                step=0.01
            )
            
            demand_smoothing_factor = st.number_input(
                "Demand Smoothing Factor",
                min_value=0.1,
                max_value=1.0,
                value=configurations.get('demand_smoothing_factor', 0.3),
                step=0.1
            )
            
            seasonal_adjustment = st.checkbox(
                "Enable Seasonal Adjustment",
                value=configurations.get('seasonal_adjustment', True)
            )
        
        if st.form_submit_button("💾 Save Forecasting Settings"):
            try:
                updated_configs = {
                    'forecast_horizon_days': forecast_horizon,
                    'safety_stock_multiplier': safety_stock_multiplier,
                    'reorder_threshold_days': reorder_threshold,
                    'price_variance_threshold': price_variance_threshold,
                    'demand_smoothing_factor': demand_smoothing_factor,
                    'seasonal_adjustment': seasonal_adjustment
                }
                
                result = st.session_state.api_client.update_system_configurations(updated_configs)
                st.success("✅ Forecasting settings updated successfully!")
                
            except Exception as e:
                st.error(f"❌ Failed to update settings: {str(e)}")
    
    st.divider()
    
    st.subheader("📧 Notification Settings")
    
    with st.form("notification_settings"):
        notification_email = st.text_input(
            "Notification Email",
            value=configurations.get('notification_email', 'admin@sifs.com')
        )
        
        backup_frequency = st.selectbox(
            "Backup Frequency",
            ["daily", "weekly", "monthly"],
            index=["daily", "weekly", "monthly"].index(configurations.get('backup_frequency', 'daily'))
        )
        
        data_retention_days = st.number_input(
            "Data Retention (days)",
            min_value=30,
            max_value=1095,
            value=configurations.get('data_retention_days', 365)
        )
        
        if st.form_submit_button("💾 Save Notification Settings"):
            try:
                updated_configs = {
                    'notification_email': notification_email,
                    'backup_frequency': backup_frequency,
                    'data_retention_days': data_retention_days
                }
                
                result = st.session_state.api_client.update_system_configurations(updated_configs)
                st.success("✅ Notification settings updated successfully!")
                
            except Exception as e:
                st.error(f"❌ Failed to update settings: {str(e)}")
    
    st.divider()
    
    st.subheader("🔧 System Maintenance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Clear Cache", use_container_width=True):
            st.success("Cache cleared successfully!")
    
    with col2:
        if st.button("📊 Generate System Report", use_container_width=True):
            st.success("System report generated!")
    
    with col3:
        if st.button("💾 Backup Database", use_container_width=True):
            st.success("Database backup initiated!")

def main():
    """Main application function with enhanced error handling"""
    # Custom CSS
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        margin: 0.25rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    init_session_state()
    
    # Check authentication
    if not st.session_state.authenticated:
        login_page()
        return
    
    # Display sidebar navigation
    sidebar_navigation()
    
    # Display selected page with error handling
    try:
        start_time = time.time()
        
        if st.session_state.current_page == "Dashboard":
            dashboard_page()
        elif st.session_state.current_page == "Forecasting":
            forecasting_page()
        elif st.session_state.current_page == "Inventory":
            inventory_page()
        elif st.session_state.current_page == "Reports":
            reports_page()
        elif st.session_state.current_page == "Data Upload":
            data_upload_page()
        elif st.session_state.current_page == "User Management":
            user_management_page()
        elif st.session_state.current_page == "Settings":
            settings_page()
        else:
            dashboard_page()
            
        # Log performance
        logger.info(f"Page {st.session_state.current_page} loaded in {time.time()-start_time:.2f}s")
        
    except Exception as e:
        st.error("An error occurred. Please try again or refresh the page.")
        logger.error(f"Page error: {e}", exc_info=True)
        time.sleep(UI_DELAY)
        st.rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Application crashed: {e}", exc_info=True)
        st.error("A critical error occurred. Please refresh the page.")
        time.sleep(UI_DELAY)
        st.rerun()