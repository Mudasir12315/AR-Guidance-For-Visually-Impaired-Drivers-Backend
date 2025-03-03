from sqlalchemy.orm import declarative_base,sessionmaker
from sqlalchemy import create_engine
from SensitiveData import username,password,database

Base=declarative_base()
# MySQL URI
DATABASE_URI = f"mysql+pymysql://{username}:{password}@localhost/{database}"

engine=create_engine(DATABASE_URI,echo=True)
Session=sessionmaker(bind=engine)