"""Literature monitor for Alz_face_analyze.

Auto-fetches papers across 4 directions (embedding / asymmetry / emotion / age)
× 4 sources (arXiv / Semantic Scholar / PubMed / OpenAlex), downloads PDFs to
references/waiting_review/, and (optionally) git-pushes results.
"""

__version__ = "0.1.0"
