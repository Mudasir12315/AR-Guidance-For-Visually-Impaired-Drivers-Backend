from sqlalchemy import String, Integer, Column
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = 'user'

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(100), nullable=False)
    password = Column(String(100), nullable=False)
    email = Column(String(100), unique=True)
    isLogOut=Column(Integer,nullable=False)

    # Relationships
    preference = relationship('UserPreference', back_populates='user',uselist=False)
    logs = relationship('UserLog', back_populates='user')

    def __init__(self,full_name,password,email,islogout):
        self.full_name=full_name
        self.password=password
        self.email=email
        self.isLogOut=islogout