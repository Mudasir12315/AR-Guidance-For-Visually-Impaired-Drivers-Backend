from sqlalchemy import String, Integer, Column, Date, Time, ForeignKey, Float,Boolean
from sqlalchemy.orm import relationship
from database import Base

class UserLog(Base):
    __tablename__ = 'user_log'

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('user.user_id'), nullable=False)
    detected_object = Column(String(100))
    alert = Column(String(50))
    distance = Column(Float)
    date = Column(Date)
    time = Column(Time)
    img_path = Column(String(255))
    camera_mode=Column(Integer) #0 for left mirror and 1 for right mirror
    is_displayed = Column(Boolean, default=False)  # New column to track if displayed

    # Relationships
    user = relationship("User", back_populates="logs")
