"""
TruthCheck Database Models
Import all models here so they are registered with SQLAlchemy when the package is imported.
"""
from models.user import User
from models.report import Report

__all__ = ['User', 'Report']
