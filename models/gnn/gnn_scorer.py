"""
FraudTrap — Graph Neural Network (Tier 3)
Fraud ring detection via entity relationship graphs.
Architecture: GraphSAGE with temporal edge features.
Uses Lambda architecture: pre-computed embeddings for known entities (fast path)
+ real-time inference for new entities (slow path, ~50ms).
"""
from __future__ import annotations
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ── Lightweight GraphSAGE (no PyG dependency required for inference) ──────────

class SAGEConv(nn.Module):
    """
    Simplified GraphSAGE convolution.
    Aggregates neighbour features via mean pooling, then concatenates with self.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self   = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_neigh  = nn.Linear(in_dim, out_dim, bias=False)
        self.bias       = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor, adj_sparse: torch.Tensor) -> torch.Tensor:
        """
        x: (N, in_dim) node features
        adj_sparse: (N, N) sparse adjacency (normalised)
        """
        agg = torch.sparse.mm(adj_sparse, x)          # mean neighbour aggregation
        out = self.lin_self(x) + self.lin_neigh(agg) + self.bias
        return F.relu(out)


class FraudGNN(nn.Module):
    """
    Two-layer GraphSAGE classifier for fraud ring detection.
    Inputs: node feature matrix + adjacency matrix of entity graph.
    Output: per-node fraud probability.
    Supports Monte Carlo Dropout for uncertainty estimation.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv1   = SAGEConv(in_dim, hidden_dim)
        self.conv2   = SAGEConv(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim // 2, out_dim)
        self._mc_dropout_enabled = False

    def enable_mc_dropout(self, enabled: bool = True):
        """Enable Monte Carlo Dropout at inference time for uncertainty estimation."""
        self._mc_dropout_enabled = enabled

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Apply dropout during training OR when MC dropout is enabled
        dropout_active = self.training or self._mc_dropout_enabled
        h = self.dropout(self.conv1(x, adj)) if dropout_active else self.conv1(x, adj)
        h = self.dropout(self.conv2(h, adj)) if dropout_active else self.conv2(h, adj)
        return self.classifier(h)

    def predict_proba(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x, adj)
            return F.softmax(logits, dim=-1)[:, 1]

    def predict_proba_mc(self, x: torch.Tensor, adj: torch.Tensor, n_samples: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout prediction for uncertainty estimation.
        Returns (mean_prob, std_prob) for each node.
        """
        self.enable_mc_dropout(True)
        probs = []
        for _ in range(n_samples):
            probs.append(self.predict_proba(x, adj))
        self.enable_mc_dropout(False)
        probs_stack = torch.stack(probs, dim=0)  # (n_samples, N)
        return probs_stack.mean(dim=0), probs_stack.std(dim=0)


# ── Entity graph builder ──────────────────────────────────────────────────────

class EntityGraphBuilder:
    """
    Builds a sparse entity graph from transaction records.
    Nodes: account_ids, device_ids, ip_hashes, merchant_ids.
    Edges: shared attributes (same device → account, same IP → account, etc.)
    Edge weight: frequency of co-occurrence in the time window.
    """

    def build(
        self,
        transactions: list[dict],
        node_feature_dim: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """
        Returns:
          x:        (N, feature_dim) node feature matrix
          adj:      (N, N) normalised sparse adjacency tensor
          node_ids: list of entity identifiers (index → id)
        """
        node_set: set[str] = set()
        edges: list[tuple[str, str, float]] = []
        stats: dict[str, dict] = {}

        def add_node(node_id: str, entity_type: str, txn: dict, neighbour: Optional[str] = None) -> None:
            node_set.add(node_id)
            amount = float(txn.get("amount", 0.0) or 0.0)
            is_fraud = float(txn.get("is_fraud", txn.get("label", 0.0)) or 0.0)
            row = stats.setdefault(
                node_id,
                {
                    "type": entity_type,
                    "count": 0.0,
                    "sum": 0.0,
                    "sum_sq": 0.0,
                    "max": 0.0,
                    "fraud": 0.0,
                    "neighbours": set(),
                },
            )
            row["count"] += 1.0
            row["sum"] += amount
            row["sum_sq"] += amount * amount
            row["max"] = max(row["max"], amount)
            row["fraud"] += is_fraud
            if neighbour:
                row["neighbours"].add(neighbour)

        def get_value(txn: dict, *names: str) -> Optional[str]:
            extra = txn.get("extra_fields") or {}
            for name in names:
                value = txn.get(name)
                if value is None and isinstance(extra, dict):
                    value = extra.get(name)
                if value:
                    return str(value)
            return None

        for txn in transactions:
            acct = get_value(txn, "account_id")
            if not acct:
                continue

            acct_node = f"account:{acct}"
            add_node(acct_node, "account", txn)

            related = [
                ("device", get_value(txn, "device_id"), 1.0),
                ("ip", get_value(txn, "ip_address_hash"), 0.7),
                ("merchant", get_value(txn, "merchant_id"), 0.6),
                ("email", get_value(txn, "email_hash", "email_id"), 0.8),
                ("phone", get_value(txn, "phone_hash", "phone_id"), 0.8),
                ("address", get_value(txn, "address_hash", "shipping_address_hash"), 0.6),
                ("bank", get_value(txn, "bank_id", "issuer_bank_id"), 0.4),
                ("card", get_value(txn, "card_hash", "card_id"), 0.9),
                ("cookie", get_value(txn, "cookie_hash", "cookie_id"), 0.7),
                ("browser", get_value(txn, "user_agent_hash", "browser_fingerprint"), 0.7),
                ("session", get_value(txn, "session_id"), 0.5),
                ("recipient", get_value(txn, "counterparty_account_id", "recipient_id"), 0.9),
                ("wallet", get_value(txn, "wallet_id"), 0.9),
                ("beneficiary", get_value(txn, "beneficiary_id"), 0.9),
            ]

            for entity_type, value, weight in related:
                if not value:
                    continue
                node = f"{entity_type}:{value}"
                add_node(node, entity_type, txn, acct_node)
                stats[acct_node]["neighbours"].add(node)
                edges.append((acct_node, node, weight))

        node_ids = sorted(node_set)
        node_idx  = {nid: i for i, nid in enumerate(node_ids)}
        N = len(node_ids)

        if N == 0:
            return torch.zeros(1, node_feature_dim), torch.eye(1).to_sparse(), ["dummy"]

        # Build adjacency
        rows, cols, vals = [], [], []
        for src, dst, w in edges:
            if src in node_idx and dst in node_idx:
                i, j = node_idx[src], node_idx[dst]
                rows += [i, j]
                cols += [j, i]
                vals += [w, w]

        if rows:
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values  = torch.tensor(vals, dtype=torch.float)
            adj_raw = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
            # Row-normalise
            deg = torch.sparse.sum(adj_raw, dim=1).to_dense().clamp(min=1)
            norm_vals = values / deg[rows]
            adj = torch.sparse_coo_tensor(indices, norm_vals, (N, N)).coalesce()
        else:
            adj = torch.eye(N).to_sparse()

        x = torch.tensor(
            self._build_node_features(node_ids, stats, node_feature_dim).astype(np.float32)
        )

        return x, adj, node_ids

    @staticmethod
    def _build_node_features(
        node_ids: list[str],
        stats: dict[str, dict],
        node_feature_dim: int,
    ) -> np.ndarray:
        type_names = [
            "account", "device", "ip", "merchant", "email", "phone", "address",
            "bank", "card", "cookie", "browser", "session", "recipient",
            "wallet", "beneficiary",
        ]
        type_index = {name: i for i, name in enumerate(type_names)}
        features = np.zeros((len(node_ids), node_feature_dim), dtype=np.float32)

        for row_idx, node_id in enumerate(node_ids):
            row = stats.get(node_id, {})
            entity_type = str(row.get("type", "unknown"))
            if entity_type in type_index and type_index[entity_type] < node_feature_dim:
                features[row_idx, type_index[entity_type]] = 1.0

            count = float(row.get("count", 0.0))
            total = float(row.get("sum", 0.0))
            sum_sq = float(row.get("sum_sq", 0.0))
            max_amt = float(row.get("max", 0.0))
            fraud = float(row.get("fraud", 0.0))
            neighbours = row.get("neighbours", set())
            degree = float(len(neighbours)) if isinstance(neighbours, set) else 0.0
            avg = total / max(count, 1.0)
            variance = max(sum_sq / max(count, 1.0) - avg * avg, 0.0)
            engineered = [
                np.log1p(count),
                np.log1p(total),
                np.log1p(avg),
                np.log1p(np.sqrt(variance)),
                np.log1p(max_amt),
                fraud / max(count, 1.0),
                np.log1p(degree),
            ]
            start = min(len(type_names), node_feature_dim)
            end = min(start + len(engineered), node_feature_dim)
            features[row_idx, start:end] = engineered[: end - start]

        return features


# ── GNN scorer wrapper ────────────────────────────────────────────────────────

class GNNScorer:
    """
    Lambda-architecture GNN scorer.
    - Batch path: pre-computed entity embeddings stored in Redis (< 1ms lookup).
    - Real-time path: online GNN inference for new/unseen entities (~50ms).
    Falls back to 0.5 (neutral score) if graph is too small for meaningful inference.
    Supports Monte Carlo Dropout for uncertainty estimation and temperature scaling for calibration.
    """

    MIN_GRAPH_SIZE = 10       # don't run GNN on trivially small graphs

    def __init__(self, in_dim: int = 32, hidden_dim: int = 64):
        self.model = FraudGNN(in_dim, hidden_dim)
        self.graph_builder = EntityGraphBuilder()
        self.in_dim = in_dim
        self.is_fitted = False
        # Temperature scaling for calibration
        self.temperature: float = 1.0

    def fit(
        self,
        transactions: list[dict],
        labels: np.ndarray,
        epochs: int = 50,
        lr: float = 1e-3,
        early_stopping_patience: int = 5,
        val_split: float = 0.2,
    ) -> "GNNScorer":
        logger.info("GNNScorer.fit: {} transactions", len(transactions))
        x, adj, node_ids = self.graph_builder.build(transactions, self.in_dim)
        N = x.shape[0]

        if N < self.MIN_GRAPH_SIZE:
            logger.warning("Graph too small ({} nodes) for GNN training; skipping", N)
            return self

        # Align labels to node order (account nodes only)
        y = torch.zeros(N, dtype=torch.long)
        txn_map = {f"account:{t.get('account_id', '')}": i for i, t in enumerate(transactions)}
        for i, nid in enumerate(node_ids):
            if nid in txn_map and txn_map[nid] < len(labels):
                y[i] = int(labels[txn_map[nid]])

        # Train/val split for early stopping
        indices = torch.randperm(N)
        val_size = int(N * val_split)
        train_idx, val_idx = indices[val_size:], indices[:val_size]

        optimiser = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        # Class weights to handle imbalance
        n_fraud = y[train_idx].sum().item()
        n_legit = len(train_idx) - n_fraud
        if n_fraud > 0:
            w = torch.tensor([1.0, n_legit / n_fraud])
        else:
            w = torch.ones(2)
        criterion = nn.CrossEntropyLoss(weight=w)

        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            optimiser.zero_grad()
            logits = self.model(x[train_idx], adj[train_idx][:, train_idx])
            loss   = criterion(logits, y[train_idx])
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimiser.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(x[val_idx], adj[val_idx][:, val_idx])
                val_loss = criterion(val_logits, y[val_idx]).item()
            
            if (epoch + 1) % 10 == 0:
                logger.debug("GNN epoch {}/{} train_loss={:.4f} val_loss={:.4f}", 
                             epoch + 1, epochs, loss.item(), val_loss)
            
            # Early stopping
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping at epoch {} (best val_loss: {:.4f})", 
                                epoch + 1, best_val_loss)
                    self.model.load_state_dict(best_state)
                    break
            
            self.model.train()

        # Temperature scaling calibration on validation set
        self.model.eval()
        with torch.no_grad():
            val_logits = self.model(x[val_idx], adj[val_idx][:, val_idx])
            self.temperature = self._fit_temperature(val_logits, y[val_idx])
        
        self.is_fitted = True
        logger.info("GNNScorer training complete (temperature={:.4f})", self.temperature)
        return self

    def _fit_temperature(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Fit temperature scaling parameter on validation set."""
        # Simple grid search for temperature
        best_temp = 1.0
        best_nll = float('inf')
        for temp in np.linspace(0.5, 3.0, 26):
            scaled_logits = logits / temp
            probs = F.softmax(scaled_logits, dim=-1)
            nll = F.cross_entropy(probs, labels).item()
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        return float(best_temp)

    def score_transaction(
        self,
        account_id: str,
        recent_transactions: list[dict],
    ) -> float:
        """
        Score a single account's fraud probability via graph inference.
        Returns float in [0, 1]. Falls back to 0.5 if graph is too small.
        """
        if not self.is_fitted:
            return 0.5

        x, adj, node_ids = self.graph_builder.build(
            recent_transactions, self.in_dim
        )
        N = x.shape[0]
        if N < self.MIN_GRAPH_SIZE:
            return 0.5

        probs = self.model.predict_proba(x, adj)

        # Find the node index for the target account
        account_node = f"account:{account_id}"
        if account_node in node_ids:
            idx = node_ids.index(account_node)
            return float(probs[idx].item())
        return 0.5

    def score_transaction_with_uncertainty(
        self,
        account_id: str,
        recent_transactions: list[dict],
        mc_samples: int = 20,
    ) -> tuple[float, float]:
        """
        Score with Monte Carlo Dropout uncertainty estimation.
        Returns (mean_prob, std_prob). Higher std = more uncertain.
        """
        if not self.is_fitted:
            return 0.5, 0.0

        x, adj, node_ids = self.graph_builder.build(
            recent_transactions, self.in_dim
        )
        N = x.shape[0]
        if N < self.MIN_GRAPH_SIZE:
            return 0.5, 0.0

        mean_probs, std_probs = self.model.predict_proba_mc(x, adj, n_samples=mc_samples)

        account_node = f"account:{account_id}"
        if account_node in node_ids:
            idx = node_ids.index(account_node)
            return float(mean_probs[idx].item()), float(std_probs[idx].item())
        return 0.5, 0.0

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / "gnn.pt")
        meta = {"in_dim": self.in_dim, "is_fitted": self.is_fitted, "temperature": self.temperature}
        with open(path / "gnn_meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        logger.info("GNNScorer saved → {}", path)

    @classmethod
    def load(cls, path: Path) -> "GNNScorer":
        path = Path(path)
        with open(path / "gnn_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        obj = cls(in_dim=meta["in_dim"])
        obj.model.load_state_dict(torch.load(path / "gnn.pt", map_location="cpu"))
        obj.model.eval()
        obj.is_fitted = meta["is_fitted"]
        obj.temperature = meta.get("temperature", 1.0)
        logger.info("GNNScorer loaded from {}", path)
        return obj
