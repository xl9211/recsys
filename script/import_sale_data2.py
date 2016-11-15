# coding=utf-8
u"""
User: xulin
Date: 13-11-21
"""
from xlrd import open_workbook
from recommendation.model.sale import Sale
from recommendation.model.session import get_session


wb = open_workbook('../data/sales.xls')
s = wb.sheet_by_index(1)
print 'Sheet:', s.name

with get_session() as session:
    for row in range(s.nrows):
        sale = Sale(
            user=str(int(s.cell_value(row, 7))),
            brand=s.cell_value(row, 12),
            product=s.cell_value(row, 13)
        )

        session.add(sale)
    session.commit()




