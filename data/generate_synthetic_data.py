"""
Generates synthetic retail transaction data mimicking a pharmacy/beauty retailer (e.g. Boots).
Produces: transactions.csv, products.csv, customers.csv
Uses realistic product names, brands and categories.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import yaml

def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)

# Realistic Boots-style product catalogue
PRODUCTS_CATALOGUE = {
    "Skincare": [
        "No7 Restore & Renew Face & Neck Serum", "No7 Protect & Perfect Intense Moisturiser",
        "Boots Cucumber Eye Gel", "Simple Kind to Skin Moisturiser SPF15",
        "Neutrogena Hydro Boost Water Gel", "Olay Regenerist Micro-Sculpting Cream",
        "No7 Lift & Luminate Day Cream", "Cerave Moisturising Cream",
        "Boots Tea Tree & Witch Hazel Toner", "La Roche-Posay Effaclar Purifying Gel",
        "Aveeno Daily Moisturising Lotion", "No7 Beautiful Skin Day Cream Dry Skin",
        "Boots Vitamin C Brightening Serum", "Garnier Micellar Cleansing Water",
        "Simple Refreshing Facial Wash Gel",
    ],
    "Haircare": [
        "Pantene Pro-V Repair & Protect Shampoo", "Head & Shoulders Classic Clean Shampoo",
        "Boots Coconut & Almond Conditioner", "TRESemmé Keratin Smooth Shampoo",
        "Elvive Total Repair 5 Shampoo", "Aussie Miracle Moist Shampoo",
        "Batiste Dry Shampoo Original", "Boots Argan Oil Hair Mask",
        "Herbal Essences Bio:Renew Shampoo", "Vo5 Nourishing Conditioner",
        "Schwarzkopf Gliss Hair Repair Conditioner", "Tigi Bed Head Urban Antidotes Shampoo",
        "OGX Moroccan Argan Oil Shampoo", "Lee Stafford Bleach Blondes Shampoo",
    ],
    "Vitamins": [
        "Boots Vitamin D 1000IU Tablets", "Seven Seas Omega-3 Fish Oil Capsules",
        "Boots Multivitamins & Minerals", "Centrum Advance Multivitamin Tablets",
        "Boots Vitamin C 1000mg Effervescent", "Holland & Barrett Iron Complex",
        "Boots Folic Acid 400mcg Tablets", "Wellwoman Original Vitamins",
        "Boots Biotin Hair Skin Nails", "Solgar Magnesium Citrate Tablets",
        "Boots Probiotic Daily Capsules", "Berocca Performance Orange Effervescent",
        "Boots Evening Primrose Oil Capsules", "Nature's Best Vitamin B12",
    ],
    "Pharmacy": [
        "Paracetamol 500mg Tablets 16s", "Ibuprofen 200mg Tablets 24s",
        "Boots Hayfever Relief Cetirizine", "Rennie Heartburn Relief Tablets",
        "Gaviscon Advance Mint Liquid", "Strepsils Honey & Lemon Lozenges",
        "Sudafed Blocked Nose Capsules", "Boots Antiseptic Cream 30g",
        "Savlon Antiseptic Wound Wash", "Piriton Allergy Tablets Chlorphenamine",
        "Boots Cold & Flu Max Strength", "Anadin Extra Aspirin Paracetamol",
        "Dioralyte Relief Sachets Blackcurrant", "Boots Travel Sickness Tablets",
    ],
    "Fragrance": [
        "HUGO BOSS Boss Bottled EDT 50ml", "Davidoff Cool Water EDT 40ml",
        "Boots No7 Peachy Rose Body Spray", "Impulse True Love Body Spray",
        "Lynx Africa Body Spray 150ml", "Sure Women Anti-Perspirant Roll-On",
        "Dove Original Anti-Perspirant Spray", "Boots Extracts Rose Body Spray",
        "Radox Feel Restored Shower Gel", "Imperial Leather Original Bar Soap",
        "Nivea Men Sensitive Shower Gel", "Simple Kind to Skin Shower Gel",
    ],
    "Baby": [
        "Sudocrem Antiseptic Healing Cream 125g", "Johnson's Baby Shampoo 500ml",
        "Boots Baby Gentle Fragrance Free Wipes", "Huggies Pure Baby Wipes 72s",
        "Infacol Wind Relief Drops", "Dentinox Cradle Cap Shampoo",
        "Boots Baby Moisturising Lotion", "Calpol Infant Suspension Paracetamol",
        "Gripe Water Original Formula", "Bepanthen Nappy Care Ointment",
        "Boots Baby Bedtime Bath", "Johnson's Baby Bedtime Lotion",
    ],
    "Electrical": [
        "Oral-B Pro 600 CrossAction Electric Toothbrush", "Philips Sonicare EasyClean Toothbrush",
        "Boots Smooth Results Straighteners", "Remington Keratin Therapy Hair Dryer",
        "Braun Series 3 Electric Shaver", "Boots Pro Curl Wand 25mm",
        "Philips Lumea Prestige IPL", "Boots Nail File & Buffer Set",
        "Scholl Velvet Smooth Pedi Electronic Foot File", "Remington Proluxe Hair Dryer",
    ],
}

def generate_data(n_customers=5000, n_days=365, seed=42):
    np.random.seed(seed)
    rng = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="D")

    # Build products dataframe from catalogue
    product_rows = []
    pid = 0
    for category, names in PRODUCTS_CATALOGUE.items():
        for name in names:
            # Realistic price by category
            price_ranges = {
                "Skincare": (4.99, 34.99), "Haircare": (2.49, 12.99),
                "Vitamins": (3.99, 19.99), "Pharmacy": (2.99, 9.99),
                "Fragrance": (8.99, 39.99), "Baby": (2.99, 14.99),
                "Electrical": (19.99, 129.99),
            }
            lo, hi = price_ranges[category]
            price = round(np.random.uniform(lo, hi), 2)
            product_rows.append({
                "product_id": f"P{str(pid).zfill(4)}",
                "product_name": name,
                "category": category,
                "price": price,
            })
            pid += 1

    products = pd.DataFrame(product_rows)
    n_products = len(products)

    # Realistic UK first names and locations
    first_names = ["Emma","Oliver","Amelia","George","Isla","Harry","Sophie","Jack","Lily","Charlie",
                   "Grace","Alfie","Chloe","Freddie","Emily","Thomas","Poppy","Oscar","Jessica","William",
                   "Mohit","Priya","Aisha","Mohammed","Fatima","James","Sarah","David","Laura","Michael"]
    last_names  = ["Smith","Jones","Williams","Taylor","Brown","Davies","Evans","Wilson","Thomas","Roberts",
                   "Johnson","Walker","Wright","Robinson","Thompson","White","Hughes","Edwards","Green","Hall"]

    customers = pd.DataFrame({
        "customer_id":   [f"C{str(i).zfill(5)}" for i in range(n_customers)],
        "first_name":    np.random.choice(first_names, n_customers),
        "last_name":     np.random.choice(last_names, n_customers),
        "age":           np.random.randint(18, 80, n_customers),
        "gender":        np.random.choice(["F","M","Other"], n_customers, p=[0.60, 0.35, 0.05]),
        "loyalty_member": np.random.choice([True, False], n_customers, p=[0.65, 0.35]),
        "location":      np.random.choice(
            ["Nottingham","London","Manchester","Birmingham","Leeds","Bristol","Sheffield","Liverpool"],
            n_customers
        ),
    })

    # Category purchase probabilities — mimic real pharmacy/beauty behaviour
    # Pharmacy & skincare bought most frequently
    cat_weights = {"Skincare":0.22,"Pharmacy":0.20,"Haircare":0.18,"Vitamins":0.15,
                   "Baby":0.10,"Fragrance":0.08,"Electrical":0.07}
    product_weights = np.array([cat_weights[r["category"]] / 
                                  sum(1 for x in product_rows if x["category"]==r["category"])
                                  for r in product_rows])
    product_weights /= product_weights.sum()

    n_transactions = 80000
    transactions = pd.DataFrame({
        "transaction_id": [f"T{str(i).zfill(7)}" for i in range(n_transactions)],
        "customer_id":    np.random.choice(customers["customer_id"], n_transactions,
                                           p=np.random.dirichlet(np.ones(n_customers)*0.3)),
        "product_id":     np.random.choice(products["product_id"], n_transactions, p=product_weights),
        "date":           np.random.choice(rng, n_transactions),
        "quantity":       np.random.randint(1, 5, n_transactions),
    })
    transactions = transactions.merge(products[["product_id","price"]], on="product_id")
    transactions["revenue"] = transactions["quantity"] * transactions["price"]

    out = Path("data/raw")
    out.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(out / "transactions.csv", index=False)
    products.to_csv(out / "products.csv", index=False)
    customers.to_csv(out / "customers.csv", index=False)
    print(f"Generated {len(transactions):,} transactions | {n_customers:,} customers | {n_products} products")
    print(f"Categories: {products['category'].value_counts().to_dict()}")
    return transactions, products, customers

if __name__ == "__main__":
    cfg = load_config()
    generate_data(
        n_customers=cfg["data"]["n_customers"],
        n_days=cfg["data"]["date_range_days"]
    )