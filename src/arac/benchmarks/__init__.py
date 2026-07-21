"""Research benchmark generators kept separate from the HCC/AOB truth source."""

from .chen2018_binary import (
    CHEN2018_SCHEMA_VERSION,
    Chen2018BinaryProblem,
    Chen2018Spec,
)

__all__ = [
    "CHEN2018_SCHEMA_VERSION",
    "Chen2018BinaryProblem",
    "Chen2018Spec",
]
