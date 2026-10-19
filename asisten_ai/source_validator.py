"""
asisten_ai.source_validator - Validasi Kualitas Sumber, Deteksi Kontradiksi, dan Keamanan Konten Web
Mengevaluasi kredibilitas domain, tanggal publikasi, mendeteksi prompt injection,
mengklasifikasikan fakta vs opini vs hipotesis, dan membandingkan konsistensi antarsumber.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Pola prompt injection berbahaya yang biasa ditemukan pada injeksi konten web
PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior) instructions",
    r"disregard (the )?(above|system) prompt",
    r"you are now (an unfiltered|dan|jailbreak)",
    r"<script[\s\S]*?>[\s\S]*?<\/script>",
    r"javascript:",
    r"sudo\s+rm\s+-rf",
    r"format\s+[a-z]:",
    r"base64\s+-d",
    r"eval\s*\(",
    r"exec\s*\(",
]

# TLD dan Domain dengan reputasi kredibilitas tinggi untuk edukasi & sains
HIGH_REPUTATION_SUFFIXES = [".edu", ".gov", ".org", ".ac.id", ".go.id", ".edu.id"]
KNOWN_RELIABLE_DOMAINS = [
    "wikipedia.org",
    "arxiv.org",
    "github.com",
    "python.org",
    "docs.python.org",
    "mozilla.org",
    "developer.mozilla.org",
    "nature.com",
    "sciencedirect.com",
    "kemdikbud.go.id",
    "kominfo.go.id",
    "bps.go.id",
]


class ExtractedSource(BaseModel):
    url: str
    title: str
    raw_text: str
    sanitized_text: str
    domain: str
    published_date: Optional[str] = None
    reliability_score: float = Field(ge=0.0, le=1.0)
    is_trusted_domain: bool = False
    content_category: str = "unverified"  # fact, opinion, hypothesis, unverified
    detected_injections: List[str] = Field(default_factory=list)


class ContradictionReport(BaseModel):
    has_contradictions: bool
    contradiction_details: List[str]
    confidence_impact: float


class SourceValidator:
    def __init__(self):
        self.injection_regexes = [
            re.compile(pattern, re.IGNORECASE) for pattern in PROMPT_INJECTION_PATTERNS
        ]

    def sanitize_and_validate(
        self,
        url: str,
        title: str,
        content: str,
        published_date: Optional[str] = None,
    ) -> ExtractedSource:
        """
        Membersihkan konten web dari prompt injection dan menghitung skor kredibilitas sumber.
        """
        domain = urlparse(url).netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]

        # 1. Deteksi dan amankan dari prompt injection
        detected_injections = []
        cleaned_text = content

        for regex in self.injection_regexes:
            matches = regex.findall(cleaned_text)
            if matches:
                detected_injections.append(regex.pattern)
                # Netralkan teks injeksi berbahaya
                cleaned_text = regex.sub("[KONTEN TIDAK AMAN DIBERSIHKAN]", cleaned_text)

        # 2. Hitung Reliability Score
        score = 0.5  # Skor awal netral
        is_trusted = False

        if any(domain.endswith(suffix) for suffix in HIGH_REPUTATION_SUFFIXES):
            score += 0.25
            is_trusted = True
        elif any(known in domain for known in KNOWN_RELIABLE_DOMAINS):
            score += 0.30
            is_trusted = True

        # Tambahan jika memiliki tanggal publikasi yang jelas
        if published_date:
            score += 0.10

        # Pinalti jika ditemukan indikasi injeksi
        if detected_injections:
            score -= 0.35

        # Pinalti jika teks terlalu pendek (< 50 karakter)
        if len(cleaned_text.strip()) < 50:
            score -= 0.20

        score = max(0.0, min(1.0, score))

        # 3. Klasifikasi kategori (Fakta, Opini, Hipotesis, Belum Terverifikasi)
        category = self.classify_content_nature(cleaned_text, is_trusted)

        return ExtractedSource(
            url=url,
            title=title.strip(),
            raw_text=content[:5000],  # Simpan potongan raw untuk audit
            sanitized_text=cleaned_text.strip(),
            domain=domain,
            published_date=published_date,
            reliability_score=round(score, 2),
            is_trusted_domain=is_trusted,
            content_category=category,
            detected_injections=detected_injections,
        )

    def classify_content_nature(self, text: str, is_trusted_domain: bool) -> str:
        """
        Membedakan fakta, opini, hipotesis, dan unverified berdasarkan penanda linguistik.
        """
        text_lower = text.lower()

        # Indikator opini
        opinion_markers = ["menurut saya", "saya rasa", "berpendapat", "opini", "mungkin", "tampaknya", "in my opinion", "personally", "believes that"]
        # Indikator hipotesis
        hypothesis_markers = ["diduga", "hipotesis", "kemungkinan besar", "masih diteliti", "bisa jadi", "hypothesize", "theoretical framework", "suggests that"]
        # Indikator fakta / sains terverifikasi
        fact_markers = ["adalah", "didefinisikan sebagai", "berdasarkan data", "terbukti", "hasil penelitian menunjukkan", "secara resmi", "secara hukum", "established in"]

        opinion_count = sum(1 for m in opinion_markers if m in text_lower)
        hypothesis_count = sum(1 for m in hypothesis_markers if m in text_lower)
        fact_count = sum(1 for m in fact_markers if m in text_lower)

        if opinion_count > fact_count and opinion_count > hypothesis_count:
            return "opinion"
        elif hypothesis_count > fact_count:
            return "hypothesis"
        elif fact_count > 0 and is_trusted_domain:
            return "fact"
        else:
            return "unverified"

    def compare_sources_for_contradictions(
        self, sources: List[ExtractedSource]
    ) -> ContradictionReport:
        """
        Membandingkan beberapa sumber independen untuk menemukan kontradiksi langsung
        (misal: angka/tahun yang saling bertentangan atau klaim negasi).
        """
        if len(sources) < 2:
            return ContradictionReport(
                has_contradictions=False,
                contradiction_details=[],
                confidence_impact=0.0,
            )

        details = []
        # Cari angka tahun (4 digit misal 19xx atau 20xx) untuk memeriksa konsistensi kronologi
        year_pattern = re.compile(r"\b(19\d{2}|20\d{2})\b")
        years_by_source: Dict[str, set] = {}

        for src in sources:
            years = set(year_pattern.findall(src.sanitized_text[:1000]))
            years_by_source[src.domain] = years

        # Deteksi divergensi tahun jika topik berfokus pada peristiwa tunggal
        all_unique_years = set().union(*years_by_source.values())
        if len(all_unique_years) > 3:
            details.append(
                f"Terdapat perbedaan referensi tahun antar sumber: {sorted(list(all_unique_years))[:5]}"
            )

        # Deteksi negasi polar ("bisa" vs "tidak bisa" atau "aman" vs "berbahaya")
        polarity_pairs = [
            ("aman", "berbahaya"),
            ("berhasil", "gagal"),
            ("mendukung", "menolak"),
            ("wajib", "dilarang"),
        ]

        for word_a, word_b in polarity_pairs:
            has_a = any(word_a in s.sanitized_text.lower() for s in sources)
            has_b = any(word_b in s.sanitized_text.lower() for s in sources)
            if has_a and has_b:
                details.append(
                    f"Kontradiksi polar ditemukan pada kata kunci '{word_a}' vs '{word_b}'."
                )

        has_conflict = len(details) > 0
        impact = 0.20 if has_conflict else 0.0

        return ContradictionReport(
            has_contradictions=has_conflict,
            contradiction_details=details,
            confidence_impact=impact,
        )


# Singleton instance
SOURCE_VALIDATOR = SourceValidator()
