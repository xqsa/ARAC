"""Benchmark adapters kept outside the ARAC method implementation."""

from arac.benchmarks.aob import AobBenchmark, OptimizationProblem
from arac.benchmarks.ioh_bbob import IohBbobBenchmark
from arac.benchmarks.overlap24 import Overlap24Benchmark

__all__ = ["AobBenchmark", "IohBbobBenchmark", "OptimizationProblem", "Overlap24Benchmark"]
