from database import Base, engine
from models.User import User
from models.UserLog import UserLog
from models.UserPreference import UserPreference

def create_tables():
    Base.metadata.create_all(bind=engine)  # Create the tables
