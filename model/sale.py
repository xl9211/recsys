# coding=utf-8
u"""
User: xulin
Date: 13-11-21
"""
from sqlalchemy import Column, Integer, String
from recommendation.model.base import Base


class Sale(Base):
    __tablename__ = 'marketing_sale'

    id = Column(Integer, primary_key=True)
    user = Column(String(64), index=True)
    brand = Column(String(64), index=True)
    product = Column(String(64), index=True)


