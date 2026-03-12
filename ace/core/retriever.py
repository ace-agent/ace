from playbook_utils import parse_playbook_line, format_playbook_line
from sentence_transformers import SentenceTransformer
import numpy as np


class Retriever:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large", top_k: int = 5):
        self.model = SentenceTransformer(model_name, device="cpu")
        self.top_k = top_k
        self.bullets_with_sections: list = []
        self.passage_embeddings: np.ndarray | None = None

    def index_playbook(self, playbook: str) -> None:
        """Parse playbook bullets and pre-compute their embeddings."""
        self.bullets_with_sections = self._extract_bullets_with_sections(playbook)
        if not self.bullets_with_sections:
            self.passage_embeddings = None
            return
        passage_texts = [
            "passage: " + b["content"] for b in self.bullets_with_sections
        ]
        self.passage_embeddings = self.model.encode(passage_texts, normalize_embeddings=True)

    def retrieve(self, question: str, context: str = "", top_k: int | None = None) -> str:
        """Return a mini-playbook string containing the top-k most relevant bullets."""
        if not self.bullets_with_sections or self.passage_embeddings is None:
            return ""

        top_k = top_k if top_k is not None else self.top_k
        top_k = min(top_k, len(self.bullets_with_sections))

        query_text = "query: " + question + " " + context
        query_embedding = self.model.encode(query_text, normalize_embeddings=True)

        similarities = np.dot(self.passage_embeddings, query_embedding)
        top_k_indices = set(np.argsort(similarities)[-top_k:])

        section_order = list(dict.fromkeys(
            b["section"] for b in self.bullets_with_sections
        ))

        section_bullets: dict[str, list[str]] = {s: [] for s in section_order}
        for i, b in enumerate(self.bullets_with_sections):
            if i in top_k_indices:
                line = format_playbook_line(b["id"], b["helpful"], b["harmful"], b["content"])
                section_bullets[b["section"]].append(line)

        lines = []
        for section in section_order:
            if section_bullets[section]:
                lines.append(section)
                lines.extend(section_bullets[section])
                lines.append("")

        return "\n".join(lines).rstrip()

    def _extract_bullets_with_sections(self, playbook: str) -> list:
        results = []
        current_section = ""
        for line in playbook.strip().split("\n"):
            if line.strip().startswith("##"):
                current_section = line.strip()
            else:
                parsed = parse_playbook_line(line)
                if parsed:
                    parsed["section"] = current_section
                    results.append(parsed)
        return results
