from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from urllib.parse import quote_plus

password = quote_plus("jstamo")
SQLALCHEMY_DATABASE_URL = f"postgresql://postgres:{password}@localhost:5432/sesa_db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
