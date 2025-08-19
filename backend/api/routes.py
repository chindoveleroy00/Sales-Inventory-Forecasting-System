from datetime import datetime, timedelta
from typing import List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
import pandas as pd
import logging
from pathlib import Path
import numpy as np

# Import models
from .models import (
    UserOut, UserCreate,
    RawSalesRecord, CurrentInventoryInput,
    ForecastResponse, ReorderRecommendation,
    StatusResponse, ErrorResponse
)

try:
    from backend.database.database import get_db
    from backend.database.models.user_model import User
except ImportError:
    from database.database import get_db
    from database.models.user_model import User
from .auth import (
    get_current_user,
    get_password_hash
)

# Core components
from backend.data_pipeline.preprocessing.clean_sales import SalesPreprocessor
from backend.data_pipeline.preprocessing.transform import FeatureEngineer
from backend.inventory.reorder_logic import ReorderLogic
from backend.inventory.safety_stock_calc import SafetyStockCalculator
from backend.inventory.alert_engine import AlertEngine

# Configure router
logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize components
project_root = Path(__file__).resolve().parent.parent.parent
DATA_DIR = project_root / "data" / "raw"
sales_preprocessor = SalesPreprocessor(data_dir=DATA_DIR)
feature_engineer = FeatureEngineer(data_dir=DATA_DIR)

PRODUCT_SUPPLIER_MAPPING_PATH = project_root / 'data' / 'raw' / 'suppliers' / 'product_supplier_mapping.csv'
SUPPLIERS_PATH = project_root / 'data' / 'raw' / 'suppliers' / 'suppliers.csv'
SALES_DATA_PATH = project_root / 'data' / 'raw' / 'sales' / 'sales_2025.csv'
INVENTORY_DATA_PATH = project_root / 'data' / 'raw' / 'inventory' / 'inventory_2025.csv'

reorder_logic_instance = ReorderLogic(
    product_supplier_mapping_path=PRODUCT_SUPPLIER_MAPPING_PATH,
    suppliers_path=SUPPLIERS_PATH
)

# Initialize other components
# Initialize other components
safety_stock_calc = SafetyStockCalculator(
    product_supplier_mapping_path=PRODUCT_SUPPLIER_MAPPING_PATH,
    suppliers_path=SUPPLIERS_PATH
)
alert_engine = AlertEngine()


# --- Dashboard Endpoints ---
@router.get("/dashboard", tags=["Dashboard"])
async def get_dashboard(current_user: User = Depends(get_current_user)):
    try:
        # Load real data for dashboard metrics
        sales_df = pd.read_csv(SALES_DATA_PATH) if SALES_DATA_PATH.exists() else pd.DataFrame()
        inventory_df = pd.read_csv(INVENTORY_DATA_PATH) if INVENTORY_DATA_PATH.exists() else pd.DataFrame()
        
        # Calculate real metrics
        total_sales = sales_df['quantity_sold'].sum() if not sales_df.empty else 0
        inventory_value = (inventory_df['current_stock'] * 5.0).sum() if not inventory_df.empty else 0  # Assuming avg price $5
        
        # Get alerts
        alerts = await get_inventory_alerts_internal()
        open_alerts = len(alerts.get('alerts', []))
        
        # Get reorder recommendations
        reorder_data = await get_reorder_recommendations_internal()
        pending_reorders = len([r for r in reorder_data.get('recommendations', []) if r.get('reorder_needed')])
        
        return {
            "totalSales": int(total_sales),
            "currentInventoryValue": round(inventory_value, 2),
            "openAlerts": open_alerts,
            "upcomingForecasts": len(inventory_df) if not inventory_df.empty else 0,
            "activeUsers": 1,  # Current user
            "pendingReorders": pending_reorders,
            "systemAlerts": alerts.get('alerts', [])[:3],  # Top 3 alerts
            "recentActivity": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "action": "Dashboard accessed",
                    "user": current_user.username
                }
            ]
        }
    except Exception as e:
        logger.error(f"Error generating dashboard data: {e}")
        return {
            "totalSales": 0,
            "currentInventoryValue": 0,
            "openAlerts": 0,
            "upcomingForecasts": 0,
            "activeUsers": 1,
            "pendingReorders": 0,
            "systemAlerts": [],
            "recentActivity": []
        }


# --- Forecasting Endpoints ---
@router.get("/forecasting/forecast", tags=["Forecasting"])
async def get_forecast(
    sku_ids: Optional[str] = None,
    forecast_days: int = 30,
    current_user: User = Depends(get_current_user)
):
    """Generate demand forecasts for specified SKUs"""
    try:
        # Load sales data
        if not SALES_DATA_PATH.exists():
            raise HTTPException(status_code=404, detail="Sales data not found")
        
        sales_df = pd.read_csv(SALES_DATA_PATH)
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        
        # Parse SKU IDs
        if sku_ids:
            selected_skus = [sku.strip() for sku in sku_ids.split(',')]
        else:
            selected_skus = sales_df['sku_id'].unique().tolist()
        
        # Generate forecasts for each SKU
        forecasts = []
        current_date = datetime.now().date()
        
        for sku in selected_skus:
            sku_sales = sales_df[sales_df['sku_id'] == sku].copy()
            if sku_sales.empty:
                continue
                
            # Simple moving average forecast (replace with your actual model)
            sku_sales = sku_sales.sort_values('date')
            recent_sales = sku_sales.tail(30)  # Last 30 days
            avg_daily_demand = recent_sales['quantity_sold'].mean()
            
            # Generate forecast dates
            forecast_dates = []
            forecast_values = []
            
            for i in range(forecast_days):
                forecast_date = current_date + timedelta(days=i+1)
                # Add some seasonality and randomness
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 7)  # Weekly pattern
                forecast_value = max(1, int(avg_daily_demand * seasonal_factor * np.random.uniform(0.8, 1.2)))
                
                forecast_dates.append(forecast_date.isoformat())
                forecast_values.append(forecast_value)
            
            # Calculate confidence intervals
            std_dev = recent_sales['quantity_sold'].std()
            upper_bound = [max(v + 1.96 * std_dev, v * 1.2) for v in forecast_values]
            lower_bound = [max(1, v - 1.96 * std_dev) for v in forecast_values]
            
            forecasts.append({
                "sku_id": sku,
                "sku_name": sku_sales.iloc[0]['sku_name'],
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
                "confidence_level": 95,
                "model_accuracy": round(np.random.uniform(0.75, 0.95), 2),  # Mock accuracy
                "trend": "stable",  # Could be "increasing", "decreasing", "stable"
                "seasonality_detected": True
            })
        
        return {
            "forecasts": forecasts,
            "forecast_horizon_days": forecast_days,
            "generated_at": datetime.now().isoformat(),
            "model_version": "v1.0"
        }
        
    except Exception as e:
        logger.error(f"Error generating forecasts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate forecasts: {str(e)}")


@router.post("/forecasting/generate", tags=["Forecasting"])
async def generate_new_forecast(
    request_data: dict,
    current_user: User = Depends(get_current_user)
):
    """Trigger new forecast generation"""
    try:
        sku_ids = request_data.get('sku_ids', [])
        forecast_days = request_data.get('forecast_days', 30)
        
        # Convert list to comma-separated string
        sku_ids_str = ','.join(sku_ids) if sku_ids else None
        
        # Generate forecast
        forecast_result = await get_forecast(sku_ids_str, forecast_days, current_user)
        
        return {
            "success": True,
            "message": f"Generated forecasts for {len(forecast_result['forecasts'])} SKUs",
            "forecast_data": forecast_result
        }
        
    except Exception as e:
        logger.error(f"Error generating new forecast: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecasting/available-skus", tags=["Forecasting"])
async def get_available_skus(current_user: User = Depends(get_current_user)):
    """Get all available SKUs for forecasting"""
    try:
        if not SALES_DATA_PATH.exists():
            return {"skus": []}
        
        sales_df = pd.read_csv(SALES_DATA_PATH)
        skus = sales_df[['sku_id', 'sku_name']].drop_duplicates().to_dict('records')
        
        return {"skus": skus}
        
    except Exception as e:
        logger.error(f"Error fetching available SKUs: {e}")
        return {"skus": []}


# --- Enhanced Inventory Endpoints ---
async def get_reorder_recommendations_internal():
    """Internal function to get reorder recommendations"""
    try:
        # Load required data
        sales_df = pd.read_csv(SALES_DATA_PATH) if SALES_DATA_PATH.exists() else pd.DataFrame()
        inventory_df = pd.read_csv(INVENTORY_DATA_PATH) if INVENTORY_DATA_PATH.exists() else pd.DataFrame()
        suppliers_df = pd.read_csv(SUPPLIERS_PATH) if SUPPLIERS_PATH.exists() else pd.DataFrame()
        mapping_df = pd.read_csv(PRODUCT_SUPPLIER_MAPPING_PATH) if PRODUCT_SUPPLIER_MAPPING_PATH.exists() else pd.DataFrame()
        
        if inventory_df.empty:
            return {"recommendations": []}
        
        recommendations = []
        
        for _, item in inventory_df.iterrows():
            sku_id = item['sku_id']
            
            # Calculate demand metrics
            sku_sales = sales_df[sales_df['sku_id'] == sku_id] if not sales_df.empty else pd.DataFrame()
            avg_daily_demand = sku_sales['quantity_sold'].mean() if not sku_sales.empty else 5
            
            # Get supplier info
            supplier_info = mapping_df[mapping_df['sku_id'] == sku_id]
            if not supplier_info.empty:
                primary_supplier = supplier_info[supplier_info['is_primary_supplier'] == True].iloc[0]
                supplier_details = suppliers_df[suppliers_df['supplier_id'] == primary_supplier['supplier_id']]
                if not supplier_details.empty:
                    supplier_data = supplier_details.iloc[0]
                    lead_time = supplier_data['lead_time_days']
                    min_order_qty = supplier_data['min_order_qty']
                    supplier_name = supplier_data['name']
            else:
                lead_time = item.get('supplier_lead_time', 7)
                min_order_qty = 50
                supplier_name = "Unknown Supplier"
            
            # Calculate reorder metrics
            current_stock = item['current_stock']
            safety_stock = max(avg_daily_demand * 3, item.get('min_stock_level', 10))
            reorder_point = avg_daily_demand * lead_time + safety_stock
            
            # Determine if reorder is needed
            reorder_needed = current_stock <= reorder_point
            
            # Calculate reorder quantity
            if reorder_needed:
                target_stock = avg_daily_demand * (lead_time + 14)  # 2 weeks extra coverage
                reorder_quantity = max(min_order_qty, int(target_stock - current_stock))
            else:
                reorder_quantity = 0
            
            # Calculate days of supply
            days_of_supply = int(current_stock / avg_daily_demand) if avg_daily_demand > 0 else 999
            
            recommendations.append({
                "sku_id": str(sku_id),
                "sku_name": str(item['sku_name']),
                "current_stock": float(current_stock),
                "safety_stock_qty": int(safety_stock),
                "lead_time_days": int(lead_time),
                "min_order_qty": int(min_order_qty),
                "reorder_point": int(reorder_point),
                "reorder_needed": bool(reorder_needed),
                "reorder_quantity": int(reorder_quantity),
                "supplier_name": str(supplier_name),
                "days_of_supply": int(days_of_supply),
                "avg_daily_demand": round(avg_daily_demand, 2)
            })
        
        return {"recommendations": recommendations}
        
    except Exception as e:
        logger.error(f"Error generating reorder recommendations: {e}")
        return {"recommendations": []}


@router.get("/inventory/reorder-recommendations", response_model=Dict[str, List], tags=["Inventory"])
async def get_reorder_recommendations(current_user: User = Depends(get_current_user)):
    """Get reorder recommendations with real data"""
    return await get_reorder_recommendations_internal()


async def get_inventory_alerts_internal():
    """Internal function to get inventory alerts"""
    try:
        recommendations = await get_reorder_recommendations_internal()
        alerts = []
        
        for item in recommendations.get('recommendations', []):
            sku_id = item['sku_id']
            current_stock = item['current_stock']
            safety_stock = item['safety_stock_qty']
            days_of_supply = item['days_of_supply']
            
            # Generate alerts based on stock levels
            if current_stock <= 0:
                alerts.append({
                    "sku": sku_id,
                    "type": "Critical",
                    "message": "Out of stock - immediate reorder required",
                    "severity": "high"
                })
            elif current_stock <= safety_stock * 0.5:
                alerts.append({
                    "sku": sku_id,
                    "type": "Critical",
                    "message": f"Stock critically low ({int(current_stock)} units, {days_of_supply} days supply)",
                    "severity": "high"
                })
            elif current_stock <= safety_stock:
                alerts.append({
                    "sku": sku_id,
                    "type": "Low Stock",
                    "message": f"Below safety stock level ({int(current_stock)} units)",
                    "severity": "medium"
                })
            elif item['reorder_needed']:
                alerts.append({
                    "sku": sku_id,
                    "type": "Reorder Point",
                    "message": f"Reached reorder point ({days_of_supply} days supply remaining)",
                    "severity": "low"
                })
        
        return {"alerts": alerts}
        
    except Exception as e:
        logger.error(f"Error generating inventory alerts: {e}")
        return {"alerts": []}


@router.get("/inventory/alerts", tags=["Inventory"])
async def get_inventory_alerts(current_user: User = Depends(get_current_user)):
    """Get current inventory alerts"""
    return await get_inventory_alerts_internal()


@router.get("/inventory/summary", tags=["Inventory"])
async def get_inventory_summary(current_user: User = Depends(get_current_user)):
    """Get inventory value and status summary"""
    try:
        inventory_df = pd.read_csv(INVENTORY_DATA_PATH) if INVENTORY_DATA_PATH.exists() else pd.DataFrame()
        
        if inventory_df.empty:
            return {
                "total_items": 0,
                "total_value": 0,
                "items_needing_reorder": 0,
                "critical_items": 0,
                "categories": {}
            }
        
        # Get reorder recommendations for analysis
        reorder_data = await get_reorder_recommendations_internal()
        recommendations = reorder_data.get('recommendations', [])
        
        # Calculate summary metrics
        total_items = len(inventory_df)
        # Assuming average price of $5 per unit (you can load actual prices from sales data)
        total_value = inventory_df['current_stock'].sum() * 5.0
        
        items_needing_reorder = len([r for r in recommendations if r.get('reorder_needed')])
        critical_items = len([r for r in recommendations if r.get('days_of_supply', 999) <= 3])
        
        # Category breakdown (if you have category data)
        categories = {}
        if 'category' in inventory_df.columns:
            category_stats = inventory_df.groupby('category').agg({
                'current_stock': 'sum',
                'sku_id': 'count'
            }).to_dict('index')
            
            for cat, stats in category_stats.items():
                categories[cat] = {
                    "item_count": stats['sku_id'],
                    "total_stock": stats['current_stock']
                }
        
        return {
            "total_items": total_items,
            "total_value": round(total_value, 2),
            "items_needing_reorder": items_needing_reorder,
            "critical_items": critical_items,
            "categories": categories
        }
        
    except Exception as e:
        logger.error(f"Error generating inventory summary: {e}")
        return {
            "total_items": 0,
            "total_value": 0,
            "items_needing_reorder": 0,
            "critical_items": 0,
            "categories": {}
        }


@router.post("/inventory/update-stock", tags=["Inventory"])
async def update_stock_levels(
    request_data: dict,
    current_user: User = Depends(get_current_user)
):
    """Update stock levels manually"""
    try:
        updates = request_data.get('updates', [])
        
        if not INVENTORY_DATA_PATH.exists():
            raise HTTPException(status_code=404, detail="Inventory data not found")
        
        inventory_df = pd.read_csv(INVENTORY_DATA_PATH)
        
        updated_count = 0
        for update in updates:
            sku_id = update.get('sku_id')
            new_stock = update.get('new_stock')
            
            if sku_id and new_stock is not None:
                mask = inventory_df['sku_id'] == sku_id
                if mask.any():
                    inventory_df.loc[mask, 'current_stock'] = new_stock
                    inventory_df.loc[mask, 'last_restock_date'] = datetime.now().strftime('%Y-%m-%d')
                    updated_count += 1
        
        # Save updated inventory
        inventory_df.to_csv(INVENTORY_DATA_PATH, index=False)
        
        return {
            "success": True,
            "message": f"Updated stock levels for {updated_count} items",
            "updated_count": updated_count
        }
        
    except Exception as e:
        logger.error(f"Error updating stock levels: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/inventory/generate-purchase-orders", tags=["Inventory"])
async def generate_purchase_orders(
    request_data: dict,
    current_user: User = Depends(get_current_user)
):
    """Generate purchase orders from recommendations"""
    try:
        sku_ids = request_data.get('sku_ids', [])
        
        if not sku_ids:
            raise HTTPException(status_code=400, detail="No SKU IDs provided")
        
        # Get reorder recommendations
        reorder_data = await get_reorder_recommendations_internal()
        recommendations = reorder_data.get('recommendations', [])
        
        # Filter for selected SKUs
        selected_recommendations = [r for r in recommendations if r['sku_id'] in sku_ids]
        
        # Group by supplier
        purchase_orders = {}
        for rec in selected_recommendations:
            supplier = rec['supplier_name']
            if supplier not in purchase_orders:
                purchase_orders[supplier] = {
                    "supplier_name": supplier,
                    "order_date": datetime.now().strftime('%Y-%m-%d'),
                    "items": [],
                    "total_items": 0,
                    "estimated_total": 0
                }
            
            # Estimate cost (you can load actual prices from your data)
            estimated_cost = rec['reorder_quantity'] * 5.0  # $5 average cost per unit
            
            purchase_orders[supplier]["items"].append({
                "sku_id": rec['sku_id'],
                "sku_name": rec['sku_name'],
                "quantity": rec['reorder_quantity'],
                "estimated_unit_cost": 5.0,
                "estimated_total": estimated_cost
            })
            
            purchase_orders[supplier]["total_items"] += rec['reorder_quantity']
            purchase_orders[supplier]["estimated_total"] += estimated_cost
        
        return {
            "success": True,
            "message": f"Generated {len(purchase_orders)} purchase orders",
            "purchase_orders": list(purchase_orders.values())
        }
        
    except Exception as e:
        logger.error(f"Error generating purchase orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Data Endpoints ---
@router.post("/data/upload-sales", response_model=StatusResponse, tags=["Data"])
async def upload_sales_data(
        file: UploadFile = File(...),
        current_user: User = Depends(get_current_user)
):
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        cleaned_df = sales_preprocessor.clean_dataframe(df)
        feature_engineered_df = feature_engineer.engineer_features_dataframe(cleaned_df)

        # Save processed data
        processed_path = DATA_DIR / "sales" / f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        feature_engineered_df.to_csv(processed_path, index=False)

        return StatusResponse(
            status="success",
            message="Data processed successfully",
            details={"rows_processed": len(feature_engineered_df), "saved_to": str(processed_path)}
        )
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# --- User Management ---
@router.post("/users", response_model=UserOut, tags=["Users"])
async def create_user(
        user_data: UserCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    if current_user.role.lower() != "admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Username exists")

    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        username=user_data.username,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        email=user_data.email,
        role=user_data.role or "Data Clerk"
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


# --- System ---
@router.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@router.get("/users", response_model=Dict[str, List[UserOut]], tags=["Users"])
async def get_users(
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Get all system users - Admin only"""
    if current_user.role.lower() != "admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        users = db.query(User).all()
        return {"users": users}
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch users")


# --- System Configuration Endpoints ---
@router.get("/system/configurations", tags=["System"])
async def get_system_configurations(current_user: User = Depends(get_current_user)):
    """Get system configurations - Admin only"""
    if current_user.role.lower() != "admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    # For now, return default configurations
    # In a real system, these would be stored in database
    default_configs = {
        "forecast_horizon_days": 30,
        "safety_stock_multiplier": 1.5,
        "reorder_threshold_days": 7,
        "price_variance_threshold": 0.15,
        "demand_smoothing_factor": 0.3,
        "seasonal_adjustment": True,
        "notification_email": "admin@sifs.com",
        "backup_frequency": "daily",
        "data_retention_days": 365
    }

    return {"configurations": default_configs}


@router.put("/system/configurations", tags=["System"])
async def update_system_configurations(
        request_data: dict,
        current_user: User = Depends(get_current_user)
):
    """Update system configurations - Admin only"""
    if current_user.role.lower() != "admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    try:
        configurations = request_data.get("configurations", {})

        # Here you would typically save to database
        # For now, we'll just return success
        logger.info(f"Admin {current_user.username} updated configurations: {configurations}")

        return {
            "success": True,
            "message": "System configurations updated successfully",
            "updated_configurations": configurations
        }
    except Exception as e:
        logger.error(f"Error updating system configurations: {e}")
        raise HTTPException(status_code=500, detail="Failed to update configurations")


@router.put("/users/{user_id}", response_model=UserOut, tags=["Users"])
async def update_user(
        user_id: int,
        user_data: UserCreate,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Update an existing user - Admin only"""
    if current_user.role.lower() != "admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    # Find the user to update
    user_to_update = db.query(User).filter(User.id == user_id).first()
    if not user_to_update:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if username is taken by another user
    if user_data.username != user_to_update.username:
        existing_user = db.query(User).filter(
            User.username == user_data.username,
            User.id != user_id
        ).first()
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")

    # Update user fields
    user_to_update.username = user_data.username
    user_to_update.full_name = user_data.full_name
    user_to_update.email = user_data.email
    user_to_update.role = user_data.role or "Data Clerk"

    # Update password if provided
    if user_data.password:
        user_to_update.hashed_password = get_password_hash(user_data.password)

    try:
        db.commit()
        db.refresh(user_to_update)
        return user_to_update
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating user: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user")


@router.delete("/users/{user_id}", tags=["Users"])
async def delete_user(
        user_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    """Delete a user - Admin only"""
    if current_user.role.lower() != "admin":
        raise HTTPException(status_code=403, detail="Insufficient permissions")

    # Prevent admin from deleting themselves
    if current_user.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    # Find the user to delete
    user_to_delete = db.query(User).filter(User.id == user_id).first()
    if not user_to_delete:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        db.delete(user_to_delete)
        db.commit()
        return {"message": "User deleted successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete user")