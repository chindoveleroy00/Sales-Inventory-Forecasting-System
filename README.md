# SIFS - Sales Inventory Forecasting System

A comprehensive inventory management and demand forecasting system designed for retail operations, with specific optimizations for Zimbabwean market conditions and business contexts.

## 🚀 Features

### Core Functionality
- **Demand Forecasting**: Advanced time-series forecasting using ARIMA, Prophet, Croston, and Ensemble models
- **Inventory Management**: Real-time stock tracking, reorder recommendations, and safety stock calculations
- **Data Pipeline**: Automated data cleaning, validation, and preprocessing
- **Alert System**: Intelligent alerts for low stock, reorder points, and anomalies
- **Multi-User System**: Role-based access control (Admin, Data Entry, Viewer)
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Interactive Dashboard**: Streamlit-based frontend with real-time visualizations

### Business Intelligence
- **Sales Analytics**: Trend analysis, performance metrics, and growth tracking
- **Supplier Management**: Performance tracking, lead time optimisation
- **Purchase Order Generation**: Automated PO creation based on forecasts
- **Price Control Validation**: Zimbabwe-specific price regulation compliance
- **Seasonal Analysis**: Holiday and event impact modeling

## 🏗️ Architecture

```
SIFS_Ultimate/
├── backend/
│   ├── api/                    # FastAPI application
│   ├── data_pipeline/          # Data processing and validation
│   ├── forecasting/           # ML models and prediction engine
│   └── inventory/             # Inventory management logic
├── config/                    # Configuration files
├── data/                      # Data storage and samples
├── database/                  # Database models and migrations
├── frontend/                  # Streamlit dashboard
├── docs/                      # Documentation
└── tests/                     # Test suites
```

## 📋 Prerequisites

- Python 3.8+
- SQLite (included) or PostgreSQL/MySQL (optional)
- 2GB+ RAM recommended
- Modern web browser for frontend

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd SIFS_Ultimate
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory:
```env
# Email Configuration (for notifications)
EMAIL_SENDER=your-email@gmail.com
EMAIL_PASSWORD=your-app-password

# Database Configuration (optional - defaults to SQLite)
DATABASE_URL=sqlite:///./data/sifs.db

# API Configuration
SECRET_KEY=your-secret-key-here
API_HOST=localhost
API_PORT=8000
```

### 5. Initialize Database
```bash
python database/scripts/init_db.py
```

### 6. Generate Sample Data (Optional)
```bash
python data/generate_dataset.py
```

## 🚀 Quick Start

### Option 1: Full System (Recommended)
```bash
# Start the API server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, start the frontend
streamlit run frontend/streamlit_app.py --server.port 8501
```

### Option 2: Frontend Only (Demo Mode)
```bash
streamlit run frontend/streamlit_app.py --server.port 8501
```

Access the application at:
- **Frontend Dashboard**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## 👤 Default User Accounts

| Username | Password | Role | Access Level |
|----------|----------|------|--------------|
| admin | adminpass | admin | Full system access |
| sales | salespass | sales | Sales data and reports |
| inventory | inventorypass | inventory | Inventory management |
| dataentry | datapass | dataentry | Data upload and entry |

⚠️ **Important**: Change default passwords in production!

## 📊 Data Format Requirements

### Sales Data Upload
Your CSV/Excel files should contain these columns:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| date | Date | Sale date (YYYY-MM-DD) | ✅ |
| sku_id | String | Product SKU identifier | ✅ |
| quantity_sold | Integer | Quantity sold (≥ 0) | ✅ |
| price | Float | Price per unit | ❌ |
| category | String | Product category | ❌ |

**Example CSV:**
```csv
date,sku_id,quantity_sold,price
2025-01-01,MEALIE_2KG,15,20.00
2025-01-01,COOKOIL_2LT,8,25.50
2025-01-02,SUGAR_2KG,12,4.50
```

### Inventory Data Format
```csv
sku_id,sku_name,current_stock,min_stock_level,supplier_lead_time
MEALIE_2KG,Mealie Meal 2KG,150,50,7
COOKOIL_2LT,Cooking Oil 2L,80,30,5
```

## 🤖 API Usage

### Authentication
```python
import requests

# Login
response = requests.post("http://localhost:8000/api/auth/login", 
                        data={"username": "admin", "password": "adminpass"})
token = response.json()["access_token"]

# Use token for subsequent requests
headers = {"Authorization": f"Bearer {token}"}
```

### Key Endpoints

#### Forecasting
```python
# Get demand forecast
response = requests.get("http://localhost:8000/api/forecasting/forecast?sku_ids=MEALIE_2KG,SUGAR_2KG&forecast_days=30", headers=headers)

# Generate new forecast
requests.post("http://localhost:8000/api/forecasting/generate", 
              json={"sku_ids": ["MEALIE_2KG"], "forecast_days": 30}, 
              headers=headers)
```

#### Inventory Management
```python
# Get reorder recommendations
response = requests.get("http://localhost:8000/api/inventory/reorder-recommendations", headers=headers)

# Update stock levels
requests.post("http://localhost:8000/api/inventory/update-stock",
              json={"updates": [{"sku_id": "MEALIE_2KG", "new_stock": 200}]},
              headers=headers)

# Generate purchase orders
requests.post("http://localhost:8000/api/inventory/generate-purchase-orders",
              json={"sku_ids": ["MEALIE_2KG", "SUGAR_2KG"]},
              headers=headers)
```

#### Data Upload
```python
# Upload sales data
files = {"file": ("sales_data.csv", open("sales_data.csv", "rb"), "text/csv")}
response = requests.post("http://localhost:8000/api/data/upload-sales", files=files, headers=headers)
```

## 🧠 Forecasting Models

### Available Models
1. **ARIMA**: Classical time-series model for trend and seasonality
2. **Prophet**: Facebook's robust forecasting model
3. **Croston TSB**: Specialized for intermittent demand
4. **Ensemble**: Weighted combination of multiple models

### Model Selection
- **High-volume, regular demand**: Prophet or Ensemble
- **Seasonal patterns**: ARIMA with seasonal components
- **Sporadic/intermittent demand**: Croston TSB
- **New products**: Ensemble with conservative weights

### Configuration
Edit `backend/forecasting/config/default.yaml`:
```yaml
forecast_horizon: 30  # days
models:
  arima:
    default_order: [1, 1, 1]
    seasonal_order: [1, 1, 1, 7]
  prophet:
    growth: 'linear'
    seasonality_mode: 'multiplicative'
```

## 📈 Key Metrics & KPIs

### Forecast Accuracy
- **MAPE** (Mean Absolute Percentage Error): < 15% target
- **RMSE** (Root Mean Square Error): Model comparison
- **Bias**: Systematic over/under-forecasting detection

### Inventory Performance
- **Days of Supply**: Current stock / daily demand
- **Stock-out Rate**: Percentage of zero-stock occurrences
- **Inventory Turnover**: Sales / average inventory value

### Business Intelligence
- **Fill Rate**: Orders fulfilled completely
- **Supplier Performance**: On-time delivery, quality scores
- **Demand Variability**: Coefficient of variation analysis

## 🛡️ Data Validation & Quality

### Automated Checks
- **Schema Validation**: Column presence and data types
- **Quality Checks**: Missing values, duplicates, outliers
- **Anomaly Detection**: Statistical outliers and unusual patterns
- **Policy Compliance**: Zimbabwe price control validation

### Error Handling
```python
# Example validation report
{
    "schema_errors": ["Missing required column: date"],
    "quality_issues": ["12 duplicate records found"],
    "anomalies": ["Price spike detected for SUGAR_2KG on 2025-01-15"],
    "policy_violations": ["MEALIE_2KG price exceeds regulated maximum"]
}
```

## 🔧 Configuration

### Database Configuration (`config/database.yaml`)
```yaml
sqlite:
  database: "sifs.db"
  timeout: 30
  detect_types: 1
  isolation_level: "IMMEDIATE"
```

### System Settings (Admin Panel)
- **Forecast Horizon**: 7-365 days
- **Safety Stock Multiplier**: 1.0-3.0x
- **Reorder Threshold**: Days until stockout
- **Price Variance Alerts**: Deviation tolerance
- **Seasonal Adjustments**: Enable/disable

## 📱 Frontend Features

### Dashboard
- **Real-time KPIs**: Sales, inventory value, alerts
- **Interactive Charts**: Trend analysis, inventory distribution
- **Alert Management**: Priority-based notifications

### Forecasting Interface
- **Model Selection**: Choose forecasting algorithms
- **Scenario Planning**: Adjust parameters and assumptions
- **Confidence Intervals**: Statistical uncertainty visualization
- **Export Capabilities**: Download forecasts as CSV/Excel

### Inventory Management
- **Stock Overview**: Current levels vs. safety stock
- **Reorder Workflow**: Automated PO generation
- **Supplier Analytics**: Performance tracking
- **Alert Dashboard**: Critical stock notifications

## 🔒 Security Features

### Authentication & Authorization
- **JWT Token-based**: Secure API authentication
- **Role-based Access**: Granular permission control
- **Session Management**: Automatic token refresh
- **Password Hashing**: Werkzeug secure hashing

### Data Protection
- **Input Validation**: XSS and injection prevention
- **CORS Configuration**: Cross-origin request control
- **Error Handling**: Sensitive information masking
- **Audit Logging**: User action tracking

## 🧪 Testing

### Run Test Suite
```bash
# Unit tests
python -m pytest tests/unit/

# Integration tests
python -m pytest tests/integration/

# Full test suite with coverage
python -m pytest tests/ --cov=backend --cov-report=html
```

### Test Data Generation
```python
python tests/fixtures/generate_test_data.py
```

## 🚀 Production Deployment

### Environment Variables
```env
# Production settings
DEBUG=False
SECRET_KEY=production-secret-key
DATABASE_URL=postgresql://user:pass@localhost/sifs_prod

# Email notifications
EMAIL_SENDER=notifications@yourcompany.com
EMAIL_PASSWORD=app-specific-password

# Security
CORS_ORIGINS=["https://yourdomain.com"]
ALLOWED_HOSTS=["yourdomain.com"]
```

### Docker Deployment (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Checklist
- [ ] Change default passwords
- [ ] Configure proper database (PostgreSQL/MySQL)
- [ ] Set up SSL certificates
- [ ] Configure email notifications
- [ ] Set up monitoring and logging
- [ ] Schedule regular backups
- [ ] Configure firewall rules

## 🔍 Troubleshooting

### Common Issues

#### Database Connection Errors
```bash
# Recreate database
python database/scripts/init_db.py

# Check database file permissions
ls -la data/sifs.db
```

#### Import Errors
```bash
# Ensure you're in the correct directory
pwd  # Should end with SIFS_Ultimate

# Verify virtual environment
which python  # Should point to venv

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

#### API Connection Issues
```bash
# Check if API is running
curl http://localhost:8000/

# Verify port availability
netstat -tulpn | grep 8000
```

#### Frontend Loading Problems
```bash
# Clear Streamlit cache
streamlit cache clear

# Restart with debugging
streamlit run frontend/streamlit_app.py --server.runOnSave true --logger.level debug
```

### Performance Optimization

#### Database
- Add indexes for frequently queried columns
- Regularly vacuum SQLite database
- Consider PostgreSQL for high-volume operations

#### Memory Usage
- Monitor forecast model memory consumption
- Implement data pagination for large datasets
- Clear old processed data regularly

#### API Response Times
- Enable FastAPI response caching
- Optimize database queries
- Use async operations for I/O-bound tasks

## 📚 Additional Resources

### Documentation
- **API Specification**: `/docs/API_Spec.md`
- **Entity Relationship Diagram**: `/docs/ERD.pdf`
- **Jupyter Notebooks**: `/notebooks/` (exploratory analysis)

### Sample Data
- **Sales Data**: `/data/raw/sales/sales_2025.csv`
- **Inventory Data**: `/data/raw/inventory/inventory_2025.csv`
- **Supplier Data**: `/data/raw/suppliers/suppliers.csv`

### Configuration Files
- **Forecasting Models**: `/backend/forecasting/config/`
- **Database Schema**: `/database/models/`
- **External Data Sources**: `/data/external/`

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `python -m pytest`
5. Commit changes: `git commit -m 'Add amazing feature'`
6. Push to branch: `git push origin feature/amazing-feature`
7. Open Pull Request

### Code Style
- Follow PEP 8 for Python code
- Use type hints where applicable
- Add docstrings to functions and classes
- Maintain test coverage above 80%

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Support

### Getting Help
- **Documentation**: Check this README and `/docs/` folder
- **Issues**: Open GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions

### Contact Information
- **Project Maintainer**: [Your Name]
- **Email**: [your-email@domain.com]
- **Organization**: [Your Organization]

---

## 📊 Quick Reference

### Default URLs
- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Key File Locations
- **Main API**: `backend/main.py`
- **Frontend App**: `frontend/streamlit_app.py`
- **Database Init**: `database/scripts/init_db.py`
- **Sample Data**: `data/generate_dataset.py`

### Important Commands
```bash
# Start API server
python -m uvicorn backend.main:app --reload

# Start frontend
streamlit run frontend/streamlit_app.py

# Initialize database
python database/scripts/init_db.py

# Run tests
python -m pytest tests/

# Generate sample data
python data/generate_dataset.py
```

---

**Built with ❤️ for modern retail inventory management**
