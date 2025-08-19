from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import date as DateType, datetime


# --- Authentication Models ---

class Token(BaseModel):
    """
    Represents an authentication token response.
    """
    access_token: str
    token_type: str
    username: str
    role: str  # Updated to match paste.txt


class TokenData(BaseModel):
    """
    Represents the data stored in a JWT token.
    """
    username: Optional[str] = None
    role: Optional[str] = None


class UserBase(BaseModel):
    """
    Base user model containing common fields.
    """
    username: str
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None  # Updated to match paste.txt structure
    role: Optional[str] = "Data Clerk"  # Updated default role from paste.txt


class UserCreate(UserBase):
    """
    Model for creating new users (includes password).
    """
    password: str


class UserInDB(UserBase):
    """
    Model representing a user as stored in the database.
    """
    hashed_password: str

    class Config:
        from_attributes = True


class UserOut(UserBase):
    """
    Model for returning user data (excludes sensitive information).
    """
    id: int
    created_at: Optional[datetime] = None  # Added from paste.txt
    is_active: Optional[bool] = True  # Made optional to match paste.txt structure

    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    """
    Model for login request payload.
    """
    username: str
    password: str


# --- Sales Data Models ---

class RawSalesRecord(BaseModel):
    """
    Represents a single raw sales record as it might be uploaded.
    Enhanced with additional fields from paste.txt.
    """
    date: str = Field(..., description="Date of the sale")
    sku_id: str = Field(..., description="Stock Keeping Unit identifier")
    sku_name: str = Field(..., description="Name of the SKU")  # Added from paste.txt
    quantity_sold: int = Field(..., ge=0, description="Quantity of the product sold")  # Changed to int
    price: float = Field(..., ge=0, description="Price of the product sold")  # Made required
    category: Optional[str] = None  # Added from paste.txt
    promotion_flag: Optional[bool] = False  # Added from paste.txt


class SalesDataUpload(BaseModel):
    """
    Model for uploading multiple sales records.
    """
    records: List[RawSalesRecord]


class CleanedSalesRecord(BaseModel):
    """
    Represents a single cleaned sales record after preprocessing.
    """
    date: DateType = Field(..., description="Date of the sale")
    sku_id: str = Field(..., description="Standardized SKU identifier")
    quantity_sold: float = Field(..., ge=0, description="Cleaned quantity of the product sold")
    price: Optional[float] = Field(None, ge=0, description="Cleaned price of the product sold")
    is_payday: bool = Field(False, description="True if the date is a payday")
    is_border_day: bool = Field(False, description="True if the date is a border day")
    is_public_holiday: bool = Field(False, description="True if the date is a public holiday")


class FeatureEngineeredRecord(BaseModel):
    """
    Represents a single record after feature engineering.
    """
    date: DateType = Field(..., description="Date of the sale")
    sku_id: str = Field(..., description="Standardized SKU identifier")
    quantity_sold: float = Field(..., ge=0, description="Cleaned quantity of the product sold")
    price: Optional[float] = Field(None, ge=0, description="Cleaned price of the product sold")
    is_payday: bool = Field(False, description="True if the date is a payday")
    is_border_day: bool = Field(False, description="True if the date is a border day")
    is_public_holiday: bool = Field(False, description="True if the date is a public holiday")

    # Time features
    year: int
    month: int
    day: int
    day_of_week: int
    week_of_year: int
    quarter: int
    day_of_year: int
    is_weekend: bool

    # Demand pattern features
    time_since_last_sale: Optional[float] = None
    school_term: Optional[str] = None
    lag_1: Optional[float] = None
    lag_7: Optional[float] = None
    lag_14: Optional[float] = None
    lag_30: Optional[float] = None
    rolling_avg: Optional[float] = None

    # Macro features
    exchange_rate: Optional[float] = None
    price_usd: Optional[float] = None

    # Unique identifier
    daily_sku_id_date: str


# --- Inventory Models ---

class CurrentInventoryInput(BaseModel):
    """
    Enhanced inventory input model with additional fields from paste.txt.
    """
    sku_id: str = Field(..., description="Stock Keeping Unit identifier")
    sku_name: str = Field(..., description="Name of the SKU")  # Added from paste.txt
    current_stock: float = Field(..., ge=0, description="Current quantity in stock")
    min_stock_level: Optional[float] = None  # Added from paste.txt
    supplier_lead_time: Optional[int] = 7  # Added from paste.txt
    expiry_date: Optional[str] = None  # Added from paste.txt


class ReorderRecommendation(BaseModel):
    """
    Enhanced reorder recommendation model with fields from paste.txt.
    """
    sku_id: str = Field(..., description="Stock Keeping Unit identifier")
    sku_name: str = Field(..., description="Name of the SKU")  # Added from paste.txt
    current_stock: float = Field(..., ge=0, description="Current stock level")
    safety_stock_qty: int = Field(..., ge=0, description="Calculated safety stock quantity")
    lead_time_days: int = Field(..., ge=0, description="Supplier lead time in days")
    min_order_qty: int = Field(..., ge=0, description="Minimum order quantity")
    reorder_point: int = Field(..., ge=0, description="Inventory level at which to reorder")
    reorder_needed: bool = Field(..., description="True if reorder is needed")
    reorder_quantity: int = Field(..., ge=0, description="Recommended quantity to reorder")
    supplier_name: str = Field(..., description="Name of the supplier")  # Updated from paste.txt
    days_of_supply: int = Field(..., description="Days of supply remaining")  # Added from paste.txt
    avg_daily_demand: float = Field(..., description="Average daily demand")  # Added from paste.txt
    supplier_id: Optional[str] = Field(None, description="ID of the primary supplier")  # Kept from original


class InventoryAlert(BaseModel):
    """
    Model for inventory alerts from paste.txt.
    """
    sku: str = Field(..., description="SKU identifier")
    type: str = Field(..., description="Alert type (Critical, Low Stock, Reorder Point)")
    message: str = Field(..., description="Alert message")
    severity: str = Field(..., description="Severity level (high, medium, low)")


class AlertOutput(BaseModel):
    """
    Enhanced alert output model (keeping original structure).
    """
    sku_id: str = Field(..., description="Stock Keeping Unit identifier")
    alert_type: str = Field(..., description="Type of alert")
    message: str = Field(..., description="Detailed alert message")
    current_stock: float = Field(..., ge=0, description="Current stock level")
    projected_demand_threshold: float = Field(..., ge=0, description="Projected demand threshold")
    timestamp: datetime = Field(..., description="Alert generation timestamp")


class StockUpdate(BaseModel):
    """
    Model for individual stock updates from paste.txt.
    """
    sku_id: str
    new_stock: float


class StockUpdateRequest(BaseModel):
    """
    Model for batch stock updates from paste.txt.
    """
    updates: List[StockUpdate]


class InventorySummary(BaseModel):
    """
    Model for inventory summary from paste.txt.
    """
    total_items: int
    total_value: float
    items_needing_reorder: int
    critical_items: int
    categories: Dict[str, Dict[str, Any]]


# --- Forecasting Models ---

class ForecastOutput(BaseModel):
    """
    Represents a single forecast point for an SKU.
    """
    date: DateType = Field(..., description="Forecast date")
    yhat: float = Field(..., ge=0, description="Predicted quantity sold")
    yhat_lower: float = Field(..., ge=0, description="Lower bound of prediction interval")
    yhat_upper: float = Field(..., ge=0, description="Upper bound of prediction interval")


class ForecastData(BaseModel):
    """
    Enhanced forecast data model from paste.txt.
    """
    sku_id: str
    sku_name: str
    forecast_dates: List[str]
    forecast_values: List[float]
    upper_bound: List[float]
    lower_bound: List[float]
    confidence_level: int = 95
    model_accuracy: float
    trend: str  # "increasing", "decreasing", "stable"
    seasonality_detected: bool


class MetricsOutput(BaseModel):
    """
    Model for forecasting performance metrics.
    """
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error")
    rmse: Optional[float] = Field(None, description="Root Mean Squared Error")
    bias: Optional[float] = Field(None, description="Bias of the predictions")
    accuracy: Optional[float] = Field(None, description="Accuracy (100 - MAPE)")


class ForecastResponse(BaseModel):
    """
    Enhanced forecast response model from paste.txt.
    """
    forecasts: List[ForecastData]  # Updated to use ForecastData
    forecast_horizon_days: int
    generated_at: str
    model_version: str
    # Keep original structure as well
    forecast_data: Optional[List[ForecastOutput]] = None
    metrics: Optional[Dict[str, MetricsOutput]] = None


class ForecastRequest(BaseModel):
    """
    Model for forecast requests from paste.txt.
    """
    sku_ids: List[str]
    forecast_days: int = 30


class SKUInfo(BaseModel):
    """
    Model for SKU information from paste.txt.
    """
    sku_id: str
    sku_name: str


class AvailableSKUsResponse(BaseModel):
    """
    Model for available SKUs response from paste.txt.
    """
    skus: List[SKUInfo]


# --- Purchase Order Models ---

class PurchaseOrderItem(BaseModel):
    """
    Model for purchase order items from paste.txt.
    """
    sku_id: str
    sku_name: str
    quantity: int
    estimated_unit_cost: float
    estimated_total: float


class PurchaseOrder(BaseModel):
    """
    Model for purchase orders from paste.txt.
    """
    supplier_name: str
    order_date: str
    items: List[PurchaseOrderItem]
    total_items: int
    estimated_total: float


class PurchaseOrderRequest(BaseModel):
    """
    Model for purchase order requests from paste.txt.
    """
    sku_ids: List[str]


class PurchaseOrderResponse(BaseModel):
    """
    Model for purchase order responses from paste.txt.
    """
    success: bool
    message: str
    purchase_orders: List[PurchaseOrder]


# --- Dashboard Models ---

class DashboardData(BaseModel):
    """
    Model for dashboard data from paste.txt.
    """
    totalSales: int
    currentInventoryValue: float
    openAlerts: int
    upcomingForecasts: int
    activeUsers: int
    pendingReorders: int
    systemAlerts: List[InventoryAlert]
    recentActivity: List[Dict[str, Any]]


# --- System Models ---

class StatusResponse(BaseModel):
    """
    Enhanced status response model.
    """
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """
    Enhanced error response model from paste.txt.
    """
    error: str  # Added from paste.txt
    message: str  # Added from paste.txt
    detail: Optional[str] = None  # Kept from original
    error_code: Optional[str] = None  # Kept from original
    details: Optional[Dict[str, Any]] = None  # Added from paste.txt


# --- Configuration Models ---

class SystemConfiguration(BaseModel):
    """
    Model for system configuration from paste.txt.
    """
    forecast_horizon_days: int = 30
    safety_stock_multiplier: float = 1.5
    reorder_threshold_days: int = 7
    price_variance_threshold: float = 0.15
    demand_smoothing_factor: float = 0.3
    seasonal_adjustment: bool = True
    notification_email: str = "admin@sifs.com"
    backup_frequency: str = "daily"
    data_retention_days: int = 365


class ConfigurationRequest(BaseModel):
    """
    Model for configuration requests from paste.txt.
    """
    configurations: SystemConfiguration


class ConfigurationResponse(BaseModel):
    """
    Model for configuration responses from paste.txt.
    """
    success: bool
    message: str
    updated_configurations: Optional[SystemConfiguration] = None