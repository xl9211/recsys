# coding=utf-8
u"""
User: xulin
Date: 13-6-6
Time: 下午3:23
"""
from recommendation.model.base import Base
from recommendation.model.session import engine
from recommendation.model.sale import *


Base.metadata.create_all(engine)


