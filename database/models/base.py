from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Boolean, Date, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Product(Base):
    __tablename__ = "products"

    sku_id = Column(String(20), primary_key=True)
    sku_name = Column(String(100), nullable=False)
    description = Column(Text)
    category = Column(String(30), nullable=False)
    base_price = Column(Float(precision=2), nullable=False)
    expiry_days = Column(Integer)
    peak_months = Column(String(20))
    peak_boost = Column(Float(precision=2))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    sales = relationship("Sale", back_populates="product")
    inventory = relationship("Inventory", back_populates="product")
    suppliers = relationship("ProductSupplier", back_populates="product")

class Sale(Base):
    __tablename__ = "sales"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    sku_id = Column(String(20), ForeignKey("products.sku_id"), nullable=False, index=True)
    quantity_sold = Column(Integer, nullable=False)
    price = Column(Float(precision=2), nullable=False)
    promotion_flag = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    product = relationship("Product", back_populates="sales")

class Inventory(Base):
    __tablename__ = "inventory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sku_id = Column(String(20), ForeignKey("products.sku_id"), nullable=False, index=True)
    current_stock = Column(Integer, nullable=False)
    min_stock_level = Column(Integer, nullable=False)
    supplier_lead_time = Column(Integer, nullable=False)
    expiry_date = Column(Date)
    last_restock_date = Column(Date, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    product = relationship("Product", back_populates="inventory")

class Supplier(Base):
    __tablename__ = "suppliers"

    supplier_id = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False)
    contact = Column(String(20), nullable=False)
    lead_time_days = Column(Integer, nullable=False)
    min_order_qty = Column(Integer, nullable=False)
    location = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    products = relationship("ProductSupplier", back_populates="supplier")

class ProductSupplier(Base):
    __tablename__ = "product_suppliers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sku_id = Column(String(20), ForeignKey("products.sku_id"), nullable=False)
    supplier_id = Column(String(20), ForeignKey("suppliers.supplier_id"), nullable=False)
    is_primary_supplier = Column(Boolean, default=True)
    contract_price_multiplier = Column(Float(precision=2), nullable=False)
    payment_terms = Column(String(20), nullable=False)

    product = relationship("Product", back_populates="suppliers")
    supplier = relationship("Supplier", back_populates="products")

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    name = Column(String(50), nullable=False)
    impact = Column(Float(precision=2), nullable=False)
    is_recurring = Column(Boolean, default=True)