from sqlalchemy import String, Integer, Column, Float, ForeignKey,Boolean
from sqlalchemy.orm import relationship
from database import Base


class UserPreference(Base):
    __tablename__ = 'user_preference'

    pre_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('user.user_id'),unique=True, nullable=False)
    peripheral_threshold = Column(Float)
    distance_threshold = Column(Float)
    distance_status= Column(Boolean)
    peripheral_status = Column(Boolean)
    color_status = Column(Boolean)

    user=relationship('User',back_populates='preference')

    def __init__(self,user_id,peripheral_threshold,distance_threshold,distance_status,peripheral_status,color_status):
        self.user_id=user_id
        self.peripheral_threshold=peripheral_threshold
        self.distance_threshold=distance_threshold
        self.distance_status=distance_status
        self.peripheral_status=peripheral_status
        self.color_status=color_status