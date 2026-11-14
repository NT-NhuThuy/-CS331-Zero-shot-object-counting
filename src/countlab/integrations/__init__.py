"""Explicit backends; importing countlab never loads GPU dependencies.

Runtime adapters always receive GT-free image/query inputs.  Model weights and
upstream source are supplied by users and must be fingerprinted per run.
"""
