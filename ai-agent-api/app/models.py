"""
SQLAlchemy Models for AI Agent Stack

IMPORTANT: Do NOT use "metadata" as a column name - it's reserved by SQLAlchemy's DeclarativeBase.
Use "metainfo", "task_metadata", etc. instead.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import String, Text, Integer, Boolean, DateTime, JSON, ForeignKey, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.database import Base


class Task(Base):
    """Stores task execution history for learning and retry logic."""
    __tablename__ = "tasks"
    
    # Task identification
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    task_type: Mapped[str] = mapped_column(String(50), default="general")
    
    # Execution details
    goal: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    steps_taken: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Status tracking
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending, running, completed, failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Site association
    site_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    site_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("sites.id"), nullable=True)
    
    # NOTE: Using "metainfo" instead of "metadata" - metadata is RESERVED in SQLAlchemy!
    metainfo: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    site: Mapped[Optional["Site"]] = relationship("Site", back_populates="tasks")


class Site(Base):
    """Stores site-specific information and learned patterns."""
    __tablename__ = "sites"
    
    # Site identification
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    domain: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Authentication
    login_required: Mapped[bool] = mapped_column(Boolean, default=False)
    login_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    auth_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Site structure (learned patterns)
    form_structure: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    navigation_patterns: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    selectors: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # NOTE: Using "site_metadata" instead of "metadata" - RESERVED word!
    site_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_visited: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_updated: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    tasks: Mapped[list["Task"]] = relationship("Task", back_populates="site")


class Preference(Base):
    """Stores user preferences as key-value pairs."""
    __tablename__ = "preferences"
    
    # Key-value storage
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    value_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    value_type: Mapped[str] = mapped_column(String(20), default="string")  # string, json, int, bool
    
    # Categorization
    category: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Opportunity(Base):
    """Stores discovered opportunities (grants, funding, etc.) for jspriggins.com."""
    __tablename__ = "opportunities"
    
    # Basic info
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    opportunity_type: Mapped[str] = mapped_column(String(50), default="grant")  # grant, residency, fellowship, etc
    
    # Source
    source_url: Mapped[str] = mapped_column(String(1000), nullable=False)
    source_site: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Details
    deadline: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    budget: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    budget_amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    currency: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    eligibility: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Categorization
    tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    categories: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    
    # NOTE: Using "opportunity_metadata" instead of "metadata" - RESERVED!
    opportunity_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="new")  # new, reviewed, published, archived
    is_eligible: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    relevance_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # WordPress integration
    wp_post_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Timestamps
    discovered_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AgentMemory(Base):
    """Stores agent memories and learned information for context."""
    __tablename__ = "agent_memories"
    
    # Memory content
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    memory_type: Mapped[str] = mapped_column(String(50), nullable=False)  # fact, preference, skill, context
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # NOTE: Using "memory_metadata" instead of "metadata" - RESERVED!
    memory_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Embedding for vector similarity (optional - for pgvector/Qdrant integration)
    embedding_vector: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string of vector
    
    # Importance/relevance
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Source tracking
    source_task_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("tasks.id"), nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class LLMRouterLog(Base):
    """Logs LLM router decisions for analysis and optimization."""
    __tablename__ = "llm_router_logs"
    
    # Request info
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_type: Mapped[str] = mapped_column(String(50), nullable=False)  # chat, code, research, etc.
    prompt_preview: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    estimated_complexity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Routing decision
    selected_provider: Mapped[str] = mapped_column(String(50), nullable=False)
    selected_model: Mapped[str] = mapped_column(String(100), nullable=False)
    routing_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Performance
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tokens_input: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tokens_output: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cost_estimate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # NOTE: Using "request_metadata" instead of "metadata" - RESERVED!
    request_metadata: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Status
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ScheduledJob(Base):
    """Stores scheduled/recurring job configurations."""
    __tablename__ = "scheduled_jobs"
    
    # Job identification
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    job_type: Mapped[str] = mapped_column(String(50), nullable=False)  # scrape, monitor, report, etc
    
    # Schedule (cron-style)
    cron_expression: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    interval_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    
    # Job configuration
    config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_run: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    next_run: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    run_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
