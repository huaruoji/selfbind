"""
Identity Store: editable identity facts for injection.

Each fact is a (key, value) pair. Facts are converted to a dense vector
by encoding as text and averaging token embeddings. This vector is then
injected into the model's hidden states via the IdentityInjector.
"""

import torch
import yaml
from pathlib import Path
from typing import Dict, Optional


class IdentityStore:
    """Stores model self-identity facts as editable key-value pairs.

    Usage:
        store = IdentityStore()
        store.add("model_name", "ResearchKB")
        store.add("creator", "ResearchKB Team")
        store.set_identity(store.to_text())

    Or load from YAML:
        store = IdentityStore.from_yaml("configs/identity.yaml")
    """

    def __init__(self, name: str = "default"):
        self.name = name
        self.facts: Dict[str, str] = {}

    def add(self, key: str, value: str) -> "IdentityStore":
        self.facts[key] = value
        return self

    def remove(self, key: str) -> "IdentityStore":
        self.facts.pop(key, None)
        return self

    def to_text(self, prefix: str = "I am") -> str:
        """Convert facts to a natural-language identity claim.

        Example:
            "I am ResearchKB. I was created by ResearchKB Team.
             My purpose is research assistance.
             I can help with: reading papers, organizing notes."
        """
        if not self.facts:
            return ""

        parts = []
        for key, value in self.facts.items():
            parts.append(f"{value}")

        statement = f"{prefix} {'. '.join(parts)}."
        return statement

    def to_vector(
        self, model, tokenizer, device: str = "cuda:0"
    ) -> Optional[torch.Tensor]:
        """Encode identity text → hidden state vector via token embeddings.

        Returns a [1, hidden_dim] tensor suitable for residual injection.
        """
        text = self.to_text()
        if not text:
            return None

        tokens = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.model.embed_tokens(tokens.input_ids).to(
                device=device, dtype=model.dtype
            )
        # Average pool: single identity vector
        return emb.mean(dim=1, keepdim=False)  # [1, hidden_dim]

    @classmethod
    def from_yaml(cls, path: str) -> "IdentityStore":
        """Load identity facts from a YAML config."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Identity config not found: {path}")

        data = yaml.safe_load(path.read_text())
        store = cls(name=data.get("name", path.stem))

        for fact in data.get("facts", []):
            store.add(fact["key"], fact["value"])

        return store

    def __repr__(self) -> str:
        return f"IdentityStore({self.name}, facts={len(self.facts)})"
