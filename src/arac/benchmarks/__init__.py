"""Benchmark adapters kept outside the ARAC method implementation."""

from arac.benchmarks.aob import AobBenchmark, OptimizationProblem
from arac.benchmarks.ioh_bbob import IohBbobBenchmark

__all__ = ["AobBenchmark", "IohBbobBenchmark", "OptimizationProblem"]
