"""
Product Recommendation Engine:
  1. Collaborative Filtering (user-item matrix, cosine similarity)
  2. Content-Based Filtering (product category/price features)
  3. Hybrid approach (weighted combination)
Designed to personalise the shopping experience — a key Boots DS use case.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from src.utils.logger import get_logger

logger = get_logger(__name__)

class CollaborativeFilteringRecommender:
    """User-based collaborative filtering using cosine similarity."""

    def __init__(self, min_interactions: int = 3, n_recommendations: int = 10):
        self.min_interactions  = min_interactions
        self.n_recommendations = n_recommendations
        self.user_item_matrix  = None
        self.similarity_matrix = None
        self.customer_index    = None
        self.product_index     = None

    def fit(self, transactions: pd.DataFrame):
        logger.info("Building user-item matrix for collaborative filtering...")

        # Filter customers with minimum interactions
        active = transactions.groupby("customer_id")["transaction_id"].count()
        active = active[active >= self.min_interactions].index
        df = transactions[transactions["customer_id"].isin(active)]

        # Aggregate to customer-product purchase count
        interactions = df.groupby(["customer_id","product_id"])["quantity"].sum().reset_index()

        self.customer_index = {c: i for i, c in enumerate(interactions["customer_id"].unique())}
        self.product_index  = {p: i for i, p in enumerate(interactions["product_id"].unique())}
        self.idx_to_product = {v: k for k, v in self.product_index.items()}

        rows = interactions["customer_id"].map(self.customer_index)
        cols = interactions["product_id"].map(self.product_index)
        data = interactions["quantity"].values

        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.customer_index), len(self.product_index))
        )
        # Compute user-user cosine similarity
        self.similarity_matrix = cosine_similarity(self.user_item_matrix)
        logger.info(f"CF model fitted | {len(self.customer_index):,} customers | {len(self.product_index):,} products")
        return self

    def recommend(self, customer_id: str, n: int = None) -> pd.DataFrame:
        n = n or self.n_recommendations
        if customer_id not in self.customer_index:
            logger.warning(f"Customer {customer_id} not in training set — returning popular items")
            return self._popular_fallback(n)

        user_idx   = self.customer_index[customer_id]
        sim_scores = self.similarity_matrix[user_idx]
        top_users  = np.argsort(sim_scores)[::-1][1:21]  # top 20 similar users

        # Aggregate products purchased by similar users, exclude already purchased
        purchased = set(self.user_item_matrix[user_idx].indices)
        scores = np.zeros(self.user_item_matrix.shape[1])
        for u in top_users:
            weight = sim_scores[u]
            scores += weight * self.user_item_matrix[u].toarray().flatten()

        scores[list(purchased)] = 0  # exclude already bought
        top_products = np.argsort(scores)[::-1][:n]

        return pd.DataFrame({
            "product_id": [self.idx_to_product[i] for i in top_products],
            "score":      scores[top_products],
            "method":     "collaborative_filtering"
        })

    def _popular_fallback(self, n: int) -> pd.DataFrame:
        """Cold-start fallback: return most popular products."""
        totals = np.asarray(self.user_item_matrix.sum(axis=0)).flatten()
        top    = np.argsort(totals)[::-1][:n]
        return pd.DataFrame({
            "product_id": [self.idx_to_product[i] for i in top],
            "score":      totals[top],
            "method":     "popularity_fallback"
        })


class ContentBasedRecommender:
    """Content-based filtering using product category and price features."""

    def __init__(self, n_recommendations: int = 10):
        self.n_recommendations = n_recommendations
        self.product_features  = None
        self.similarity_matrix = None

    def fit(self, products: pd.DataFrame):
        logger.info("Building product feature matrix for content-based filtering...")
        # One-hot encode category, normalise price
        features = pd.get_dummies(products[["product_id","category"]], columns=["category"])
        features["price_norm"] = (products["price"] - products["price"].min()) / (products["price"].max() - products["price"].min())
        self.product_ids       = features["product_id"].values
        self.product_index     = {p: i for i, p in enumerate(self.product_ids)}
        X = features.drop(columns=["product_id"]).values.astype(float)
        self.similarity_matrix = cosine_similarity(X)
        logger.info(f"Content-based model fitted | {len(self.product_ids)} products")
        return self

    def recommend_similar(self, product_id: str, n: int = None) -> pd.DataFrame:
        n = n or self.n_recommendations
        if product_id not in self.product_index:
            return pd.DataFrame()
        idx    = self.product_index[product_id]
        scores = self.similarity_matrix[idx]
        top    = np.argsort(scores)[::-1][1:n+1]
        return pd.DataFrame({
            "product_id": self.product_ids[top],
            "score":      scores[top],
            "method":     "content_based"
        })


class HybridRecommender:
    """Weighted hybrid of collaborative and content-based approaches."""

    def __init__(self, cf_weight: float = 0.7, cb_weight: float = 0.3, n_recommendations: int = 10):
        self.cf = CollaborativeFilteringRecommender(n_recommendations=n_recommendations * 2)
        self.cb = ContentBasedRecommender(n_recommendations=n_recommendations * 2)
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.n         = n_recommendations

    def fit(self, transactions: pd.DataFrame, products: pd.DataFrame):
        self.cf.fit(transactions)
        self.cb.fit(products)
        return self

    def recommend(self, customer_id: str, last_purchased_product: str = None) -> pd.DataFrame:
        cf_recs = self.cf.recommend(customer_id).set_index("product_id")["score"]
        cb_recs = pd.Series(dtype=float)
        if last_purchased_product:
            cb_df   = self.cb.recommend_similar(last_purchased_product)
            if not cb_df.empty:
                cb_recs = cb_df.set_index("product_id")["score"]

        all_products = set(cf_recs.index) | set(cb_recs.index)
        hybrid_scores = {}
        for p in all_products:
            cf_s = cf_recs.get(p, 0) * self.cf_weight
            cb_s = cb_recs.get(p, 0) * self.cb_weight
            hybrid_scores[p] = cf_s + cb_s

        result = pd.DataFrame({"product_id": list(hybrid_scores.keys()),
                                "score": list(hybrid_scores.values()),
                                "method": "hybrid"})
        return result.sort_values("score", ascending=False).head(self.n).reset_index(drop=True)
