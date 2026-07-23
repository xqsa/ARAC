"""Research benchmark generators kept separate from the HCC/AOB truth source."""

from .chen2018_binary import (
    CHEN2018_SCHEMA_VERSION,
    Chen2018BinaryProblem,
    Chen2018Spec,
)
from .wang2025_overlapping import (
    WANG2025_MAX_SHARED_MEMBERSHIPS,
    WANG2025_SCHEMA_VERSION,
    Wang2025OverlappingProblem,
    Wang2025OverlappingSpec,
)
from .wang2025_local_escape import (
    WANG2025_LOCAL_ESCAPE_CASES,
    WANG2025_LOCAL_ESCAPE_SUITE_VERSION,
    Wang2025LocalEscapeCase,
    get_wang2025_local_escape_case,
)

__all__ = [
    "CHEN2018_SCHEMA_VERSION",
    "Chen2018BinaryProblem",
    "Chen2018Spec",
    "WANG2025_MAX_SHARED_MEMBERSHIPS",
    "WANG2025_LOCAL_ESCAPE_CASES",
    "WANG2025_LOCAL_ESCAPE_SUITE_VERSION",
    "WANG2025_SCHEMA_VERSION",
    "Wang2025LocalEscapeCase",
    "Wang2025OverlappingProblem",
    "Wang2025OverlappingSpec",
    "get_wang2025_local_escape_case",
]
