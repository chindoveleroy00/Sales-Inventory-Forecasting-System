import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

# Zimbabwe-specific Configuration
CONFIG = {
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "output_dir": "data",
    "seed": 2025,
    "currency": "USD",
    "products": [
        # Basic Groceries & Staples
        {"sku_id": "MEALIE_2KG", "sku_name": "Mealie Meal 2KG", "base_price": 2.00, "category": "Staple", "expiry_days": 180, "seasonality": {"peak_months": [1,5,9], "peak_boost": 1.5}},
        {"sku_id": "MEALIE_10KG", "sku_name": "Mealie Meal 10KG", "base_price": 8.00, "category": "Staple", "expiry_days": 180, "seasonality": {"peak_months": [1,5,9], "peak_boost": 1.6}},
        {"sku_id": "RICE_5KG", "sku_name": "Rice 5KG", "base_price": 7.50, "category": "Staple", "expiry_days": 365, "seasonality": {"peak_months": [], "peak_boost": 1.0}},
        {"sku_id": "COOKOIL_2LT", "sku_name": "Cooking Oil 2L", "base_price": 3.50, "category": "Staple", "expiry_days": 540, "seasonality": {"peak_months": [11,12], "peak_boost": 1.4}},
        {"sku_id": "SUGAR_2KG", "sku_name": "Sugar 2KG", "base_price": 2.20, "category": "Staple", "expiry_days": None, "seasonality": {"peak_months": [6,7], "peak_boost": 1.3}},
        {"sku_id": "SALT_1KG", "sku_name": "Salt 1KG", "base_price": 1.00, "category": "Staple", "expiry_days": None, "seasonality": {"peak_months": [], "peak_boost": 1.0}},
        
        # Beverages
        {"sku_id": "COKE_500ML", "sku_name": "Coke 500ml", "base_price": 1.00, "category": "Beverage", "expiry_days": 180, "seasonality": {"peak_months": [11,12], "peak_boost": 1.8}},
        {"sku_id": "MAZOE_1LT", "sku_name": "Mazoe Orange 1L", "base_price": 2.50, "category": "Beverage", "expiry_days": 270, "seasonality": {"peak_months": [10,11,12], "peak_boost": 1.7}},
        {"sku_id": "TEA_250G", "sku_name": "Tea Leaves 250g", "base_price": 3.00, "category": "Beverage", "expiry_days": 365, "seasonality": {"peak_months": [6,7], "peak_boost": 1.4}},
        {"sku_id": "MILK_1LT", "sku_name": "Long-life Milk 1L", "base_price": 1.80, "category": "Beverage", "expiry_days": 180, "seasonality": {"peak_months": [], "peak_boost": 1.0}},
        
        # Household Items
        {"sku_id": "SOAP_100G", "sku_name": "Bathing Soap 100g", "base_price": 0.80, "category": "Household", "expiry_days": None, "seasonality": {"peak_months": [], "peak_boost": 1.0}},
        {"sku_id": "DETERGENT_500G", "sku_name": "Washing Powder 500g", "base_price": 2.50, "category": "Household", "expiry_days": None, "seasonality": {"peak_months": [], "peak_boost": 1.0}},
        {"sku_id": "CANDLES_6PK", "sku_name": "Candles 6-pack", "base_price": 1.50, "category": "Household", "expiry_days": None, "seasonality": {"peak_months": [5,6,7], "peak_boost": 1.5}},
        
        # Personal Care
        {"sku_id": "TOOTHPASTE_100G", "sku_name": "Toothpaste 100g", "base_price": 1.80, "category": "Personal Care", "expiry_days": 730, "seasonality": {"peak_months": [], "peak_boost": 1.0}},
        {"sku_id": "SANITARY_10PK", "sku_name": "Sanitary Pads 10-pack", "base_price": 3.50, "category": "Personal Care", "expiry_days": 1095, "seasonality": {"peak_months": [], "peak_boost": 1.0}},
        
        # Snacks & Confectionery
        {"sku_id": "BISCUITS_200G", "sku_name": "Biscuits 200g", "base_price": 1.20, "category": "Snack", "expiry_days": 180, "seasonality": {"peak_months": [11,12], "peak_boost": 1.6}},
        {"sku_id": "MAPUTI_100G", "sku_name": "Maputi 100g", "base_price": 0.50, "category": "Snack", "expiry_days": 90, "seasonality": {"peak_months": [], "peak_boost": 1.0}},
        
        # Bread & Baked Goods
        {"sku_id": "BREAD_LOAF", "sku_name": "Bread Loaf", "base_price": 1.00, "category": "Bakery", "expiry_days": 3, "seasonality": {"peak_months": [], "peak_boost": 1.0}},
        
        # Canned & Preserved Foods
        {"sku_id": "BEANS_410G", "sku_name": "Baked Beans 410g", "base_price": 1.50, "category": "Canned Food", "expiry_days": 730, "seasonality": {"peak_months": [], "peak_boost": 1.0}},
        {"sku_id": "PILCHARDS_155G", "sku_name": "Pilchards 155g", "base_price": 1.80, "category": "Canned Food", "expiry_days": 730, "seasonality": {"peak_months": [], "peak_boost": 1.0}},
        
        # Communication & Utilities
        {"sku_id": "AIRTIME_5", "sku_name": "Airtime $5", "base_price": 5.00, "category": "Utility", "expiry_days": None, "seasonality": {"peak_months": [], "peak_boost": 1.0}},
        {"sku_id": "ELECTRICITY_10", "sku_name": "Electricity $10", "base_price": 10.00, "category": "Utility", "expiry_days": None, "seasonality": {"peak_months": [5,6,7], "peak_boost": 1.3}}
    ],
    "events": [
        {"date": "2025-01-01", "name": "New Year", "impact": 1.5},
        {"date": "2025-04-18", "name": "Independence Day", "impact": 1.8},
        {"date": "2025-05-25", "name": "Africa Day", "impact": 1.3},
        {"date": "2025-08-12", "name": "Heroes' Day", "impact": 1.2},
        {"date": "2025-12-22", "name": "Unity Day", "impact": 1.4},
        {"date": "2025-12-25", "name": "Christmas", "impact": 2.0}
    ],
    "economic_factors": {
        "inflation_rate": 0.02,
        "payday_cycle": 30,
        "payday_impact": 1.6,
        "weekend_boost": 1.4,
        "base_demand": {
            "Staple": 15,
            "Beverage": 20,
            "Household": 8,
            "Personal Care": 5,
            "Snack": 12,
            "Bakery": 25,
            "Canned Food": 6,
            "Utility": 30
        }
    }
}

def generate_sales_data():
    """Generate sales data with Zimbabwean market patterns"""
    np.random.seed(CONFIG["seed"])
    dates = pd.date_range(CONFIG["start_date"], CONFIG["end_date"])
    records = []
    
    # Add paydays to events
    current_date = datetime.strptime(CONFIG["start_date"], "%Y-%m-%d")
    while current_date <= datetime.strptime(CONFIG["end_date"], "%Y-%m-%d"):
        CONFIG["events"].append({
            "date": current_date.strftime("%Y-%m-%d"),
            "name": "Payday",
            "impact": CONFIG["economic_factors"]["payday_impact"]
        })
        current_date += timedelta(days=CONFIG["economic_factors"]["payday_cycle"])
    
    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        month = date.month
        day_of_week = date.weekday()
        
        # Find active events
        active_events = [e for e in CONFIG["events"] if e["date"] == date_str]
        event_multiplier = max([e["impact"] for e in active_events]) if active_events else 1.0
        
        for product in CONFIG["products"]:
            # Base demand by category
            base_demand = CONFIG["economic_factors"]["base_demand"][product["category"]]
            
            # Apply seasonality
            if month in product["seasonality"]["peak_months"]:
                base_demand *= product["seasonality"]["peak_boost"]
            
            # Apply day-of-week effect
            if day_of_week in [4, 5]:  # Friday/Saturday
                base_demand *= CONFIG["economic_factors"]["weekend_boost"]
            
            # Final quantity with randomness
            quantity = int(base_demand * event_multiplier * np.random.uniform(0.8, 1.2))
            
            # Price with inflation
            months_passed = (date.year - 2025) * 12 + (date.month - 1)
            price = product["base_price"] * (1 + CONFIG["economic_factors"]["inflation_rate"]) ** months_passed
            
            records.append({
                "date": date_str,
                "sku_id": product["sku_id"],
                "sku_name": product["sku_name"],
                "quantity_sold": max(1, quantity),
                "price": round(price * np.random.uniform(0.98, 1.02), 2),
                "category": product["category"],
                "promotion_flag": np.random.random() < 0.2  # 20% promo chance
            })
    
    return pd.DataFrame(records)

def generate_inventory_data(sales_df):
    """Generate inventory data with Zimbabwean restocking patterns"""
    inventory = []
    current_date = datetime.strptime(CONFIG["end_date"], "%Y-%m-%d")
    
    for product in CONFIG["products"]:
        product_sales = sales_df[sales_df["sku_id"] == product["sku_id"]]
        avg_daily_sales = product_sales["quantity_sold"].mean()
        
        # Zimbabwean restocking logic
        if product["category"] in ["Bakery", "Beverage"]:
            # Fast-moving perishables
            coverage_days = np.random.randint(2, 5)
            lead_time = 1
        elif product["category"] == "Staple":
            # High-volume essentials
            coverage_days = np.random.randint(7, 14)
            lead_time = 3
        else:
            # Standard items
            coverage_days = np.random.randint(5, 10)
            lead_time = np.random.randint(2, 7)
        
        current_stock = int(avg_daily_sales * coverage_days)
        min_stock = int(avg_daily_sales * (coverage_days / 2))
        
        # Expiry dates
        expiry_date = (
            (current_date + timedelta(days=np.random.randint(1, product["expiry_days"]))).strftime("%Y-%m-%d")
            if product["expiry_days"] else None
        )
        
        # Fix the randint issue: ensure we have at least 1 day difference
        days_since_restock = np.random.randint(0, max(1, lead_time))
        
        inventory.append({
            "sku_id": product["sku_id"],
            "sku_name": product["sku_name"],
            "current_stock": current_stock,
            "min_stock_level": min_stock,
            "supplier_lead_time": lead_time,
            "expiry_date": expiry_date,
            "last_restock_date": (current_date - timedelta(days=days_since_restock)).strftime("%Y-%m-%d")
        })
    
    return pd.DataFrame(inventory)

def generate_supplier_data():
    """Generate supplier data with authentic Zimbabwean suppliers matched to products"""
    suppliers = [
        {
            "supplier_id": "SUP001",
            "name": "National Foods Zimbabwe",
            "contact": "+263 242 886 061",
            "lead_time_days": 3,
            "min_order_qty": 50,
            "primary_products": ["MEALIE_2KG", "MEALIE_10KG", "RICE_5KG", "BISCUITS_200G"],
            "location": "Harare"
        },
        {
            "supplier_id": "SUP002", 
            "name": "United Refineries Limited",
            "contact": "+263 242 621 881",
            "lead_time_days": 2,
            "min_order_qty": 24,
            "primary_products": ["COOKOIL_2LT"],
            "location": "Harare"
        },
        {
            "supplier_id": "SUP003",
            "name": "Delta Corporation Limited",
            "contact": "+263 242 486 981",
            "lead_time_days": 1,
            "min_order_qty": 20,
            "primary_products": ["COKE_500ML", "MAZOE_1LT"],
            "location": "Harare"
        },
        {
            "supplier_id": "SUP004",
            "name": "Unilever Zimbabwe",
            "contact": "+263 242 707 636",
            "lead_time_days": 2,
            "min_order_qty": 25,
            "primary_products": ["SOAP_100G", "DETERGENT_500G", "TOOTHPASTE_100G"],
            "location": "Harare"
        },
        {
            "supplier_id": "SUP005",
            "name": "ZESA Holdings",
            "contact": "+263 242 703 101",
            "lead_time_days": 0,
            "min_order_qty": 1,
            "primary_products": ["ELECTRICITY_10"],
            "location": "Harare"
        },
        {
            "supplier_id": "SUP006",
            "name": "Econet Wireless Zimbabwe",
            "contact": "+263 242 700 000",
            "lead_time_days": 0,
            "min_order_qty": 1,
            "primary_products": ["AIRTIME_5"],
            "location": "Harare"
        },
        {
            "supplier_id": "SUP007",
            "name": "Lobels Bread",
            "contact": "+263 242 621 773",
            "lead_time_days": 1,
            "min_order_qty": 20,
            "primary_products": ["BREAD_LOAF"],
            "location": "Harare"
        },
        {
            "supplier_id": "SUP008",
            "name": "Country Choice Foods",
            "contact": "+263 292 446 539",
            "lead_time_days": 4,
            "min_order_qty": 30,
            "primary_products": ["SUGAR_2KG"],
            "location": "Bulawayo"
        },
        {
            "supplier_id": "SUP009",
            "name": "Cairns Foods",
            "contact": "+263 242 621 454",
            "lead_time_days": 2,
            "min_order_qty": 50,
            "primary_products": ["BEANS_410G", "PILCHARDS_155G"],
            "location": "Harare"
        },
        {
            "supplier_id": "SUP010",
            "name": "Tanganda Tea Company",
            "contact": "+263 272 684 051",
            "lead_time_days": 3,
            "min_order_qty": 20,
            "primary_products": ["TEA_250G"],
            "location": "Mutare"
        },
        {
            "supplier_id": "SUP011",
            "name": "Dairibord Zimbabwe",
            "contact": "+263 242 700 951",
            "lead_time_days": 1,
            "min_order_qty": 24,
            "primary_products": ["MILK_1LT"],
            "location": "Harare"
        },
        {
            "supplier_id": "SUP012",
            "name": "Arenel Private Limited",
            "contact": "+263 242 621 987",
            "lead_time_days": 2,
            "min_order_qty": 25,
            "primary_products": ["MAPUTI_100G"],
            "location": "Harare"
        },
        {
            "supplier_id": "SUP013",
            "name": "Zimcandles Manufacturing",
            "contact": "+263 242 621 234",
            "lead_time_days": 3,
            "min_order_qty": 50,
            "primary_products": ["CANDLES_6PK"],
            "location": "Harare"
        },
        {
            "supplier_id": "SUP014",
            "name": "Refined Salt Company",
            "contact": "+263 242 621 456",
            "lead_time_days": 5,
            "min_order_qty": 100,
            "primary_products": ["SALT_1KG"],
            "location": "Harare"
        },
        {
            "supplier_id": "SUP015",
            "name": "Johnson & Johnson Zimbabwe",
            "contact": "+263 242 621 789",
            "lead_time_days": 7,
            "min_order_qty": 20,
            "primary_products": ["SANITARY_10PK"],
            "location": "Harare"
        }
    ]
    
    return pd.DataFrame(suppliers)

def generate_product_supplier_mapping():
    """Generate a mapping table between products and their suppliers"""
    suppliers_df = generate_supplier_data()
    
    # Create mapping based on primary products
    mappings = []
    for _, supplier in suppliers_df.iterrows():
        for product_sku in supplier["primary_products"]:
            mappings.append({
                "sku_id": product_sku,
                "supplier_id": supplier["supplier_id"],
                "supplier_name": supplier["name"],
                "is_primary_supplier": True,
                "contract_price_multiplier": np.random.uniform(0.85, 0.95),  # Wholesale discount
                "payment_terms": np.random.choice(["Net 30", "Net 15", "COD", "Net 45"])
            })
    
    # Add some secondary suppliers for key products
    key_products = ["MEALIE_2KG", "MEALIE_10KG", "COKE_500ML", "BREAD_LOAF", "MILK_1LT"]
    for product_sku in key_products:
        # Find suppliers that don't already supply this product
        current_suppliers = [m["supplier_id"] for m in mappings if m["sku_id"] == product_sku]
        available_suppliers = suppliers_df[~suppliers_df["supplier_id"].isin(current_suppliers)]
        
        if len(available_suppliers) > 0:
            secondary_supplier = available_suppliers.sample(1).iloc[0]
            mappings.append({
                "sku_id": product_sku,
                "supplier_id": secondary_supplier["supplier_id"],
                "supplier_name": secondary_supplier["name"],
                "is_primary_supplier": False,
                "contract_price_multiplier": np.random.uniform(0.90, 1.0),  # Less favorable terms
                "payment_terms": np.random.choice(["Net 30", "COD"])
            })
    
    return pd.DataFrame(mappings)

def save_datasets():
    """Generate and save all datasets with proper folder structure"""
    # Create directory structure
    (Path(CONFIG["output_dir"]) / "raw/sales").mkdir(parents=True, exist_ok=True)
    (Path(CONFIG["output_dir"]) / "raw/inventory").mkdir(parents=True, exist_ok=True)
    (Path(CONFIG["output_dir"]) / "raw/suppliers").mkdir(parents=True, exist_ok=True)
    (Path(CONFIG["output_dir"]) / "external").mkdir(exist_ok=True)
    
    print("Generating sales data...")
    sales_df = generate_sales_data()
    sales_df.to_csv(f"{CONFIG['output_dir']}/raw/sales/sales_2025.csv", index=False)
    
    print("Generating inventory data...")
    inventory_df = generate_inventory_data(sales_df)
    inventory_df.to_csv(f"{CONFIG['output_dir']}/raw/inventory/inventory_2025.csv", index=False)
    
    print("Generating supplier data...")
    supplier_df = generate_supplier_data()
    supplier_df.to_csv(f"{CONFIG['output_dir']}/raw/suppliers/suppliers.csv", index=False)
    
    print("Generating product-supplier mapping...")
    mapping_df = generate_product_supplier_mapping()
    mapping_df.to_csv(f"{CONFIG['output_dir']}/raw/suppliers/product_supplier_mapping.csv", index=False)
    
    print("Generating events data...")
    events_df = pd.DataFrame(CONFIG["events"])
    events_df.to_csv(f"{CONFIG['output_dir']}/external/events.csv", index=False)
    
    # Save configuration
    with open(f"{CONFIG['output_dir']}/external/config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)
    
    return {
        "sales_records": len(sales_df),
        "inventory_items": len(inventory_df),
        "suppliers": len(supplier_df),
        "product_supplier_mappings": len(mapping_df),
        "events": len(events_df)
    }

if __name__ == "__main__":
    print("Generating Zimbabwean SME datasets (USD)...")
    stats = save_datasets()
    
    # Generate sample data for display
    sample_sales = generate_sales_data()
    sample_inventory = generate_inventory_data(sample_sales)
    sample_suppliers = generate_supplier_data()
    sample_mapping = generate_product_supplier_mapping()
    
print(f"""
Successfully generated datasets in '{CONFIG["output_dir"]}' folder:
- raw/
  - sales/sales_2025.csv                    ({stats['sales_records']:,} records)
  - inventory/inventory_2025.csv            ({stats['inventory_items']} products)
  - suppliers/suppliers.csv                 ({stats['suppliers']} suppliers)
  - suppliers/product_supplier_mapping.csv ({stats['product_supplier_mappings']} mappings)
- external/
  - events.csv                              ({stats['events']} holidays)
  - config.json

Example Sales Record:
{sample_sales.iloc[0].to_dict()}

Example Inventory Record:
{sample_inventory.iloc[0].to_dict()}

Example Supplier Record:
{sample_suppliers.iloc[0].to_dict()}

Example Product-Supplier Mapping:
{sample_mapping.iloc[0].to_dict()}

Key Suppliers Added:
- National Foods Zimbabwe - Staple foods (mealie meal, rice, biscuits)
- Delta Corporation - Beverages (Coke, Mazoe)
- United Refineries - Cooking oil
- Unilever Zimbabwe - Personal care & household items
- ZESA Holdings - Electricity
- Econet Wireless - Airtime
- Lobels Bread - Fresh bread
- Country Choice Foods - Sugar
- Tanganda Tea Company - Tea
- Dairibord Zimbabwe - Milk
""")