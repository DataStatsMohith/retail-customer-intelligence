"""
Generates synthetic retail transaction data mimicking a pharmacy/beauty retailer (e.g. Boots).
Produces: transactions.csv, products.csv, customers.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)

def generate_data(n_customers=5000, n_products=200, n_days=365, seed=42):
    np.random.seed(seed)
    rng = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="D")

    # Products catalogue - pharmacy/beauty categories
    categories = ["Skincare", "Haircare", "Vitamins", "Pharmacy", "Fragrance", "Baby", "Electrical"]
    products = pd.DataFrame({
        "product_id": [f"P{str(i).zfill(4)}" for i in range(n_products)],
        "product_name": [f"Product_{i}" for i in range(n_products)],
        "category": np.random.choice(categories, n_products),
        "price": np.round(np.random.exponential(scale=12, size=n_products) + 2, 2),
    })

    # Customers
    customers = pd.DataFrame({
        "customer_id": [f"C{str(i).zfill(5)}" for i in range(n_customers)],
        "age": np.random.randint(18, 80, n_customers),
        "gender": np.random.choice(["F", "M", "Other"], n_customers, p=[0.60, 0.35, 0.05]),
        "loyalty_member": np.random.choice([True, False], n_customers, p=[0.65, 0.35]),
        "location": np.random.choice(["Nottingham","London","Manchester","Birmingham","Leeds"], n_customers),
    })

    # Transactions — power-law purchase frequency (mimics real retail)
    n_transactions = 80000
    transactions = pd.DataFrame({
        "transaction_id": [f"T{str(i).zfill(7)}" for i in range(n_transactions)],
        "customer_id": np.random.choice(customers["customer_id"], n_transactions,
                                         p=np.random.dirichlet(np.ones(n_customers)*0.3)),
        "product_id": np.random.choice(products["product_id"], n_transactions,
                                        p=np.random.dirichlet(np.ones(n_products)*0.5)),
        "date": np.random.choice(rng, n_transactions),
        "quantity": np.random.randint(1, 5, n_transactions),
    })
    transactions = transactions.merge(products[["product_id","price"]], on="product_id")
    transactions["revenue"] = transactions["quantity"] * transactions["price"]

    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(out / "transactions.csv", index=False)
    products.to_csv(out / "products.csv", index=False)
    customers.to_csv(out / "customers.csv", index=False)
    print(f"Generated {len(transactions):,} transactions | {n_customers:,} customers | {n_products} products")
    return transactions, products, customers

if __name__ == "__main__":
    cfg = load_config()
    generate_data(
        n_customers=cfg["data"]["n_customers"],
        n_products=cfg["data"]["n_products"],
        n_days=cfg["data"]["date_range_days"]
    )
