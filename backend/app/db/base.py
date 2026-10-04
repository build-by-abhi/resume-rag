"""
Declarative base + metadata shared by every model.

Notes for beginners:
* `Mapped[...]` is the modern SQLAlchemy 2.0 way to describe a column's type
  and Python type together. `mapped_column(...)` turns it into a real column.
* Importing a model module is enough for its table to be registered on
  `Base.metadata`; `create_all` later reads that metadata.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    # You can set a shared naming convention here so Alembic autogenerate
    # produces stable constraint names (useful in Phase 5).
    pass