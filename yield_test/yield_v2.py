#encoding=utf8

"""
通过一段实际代码理解yield的用法
"""
class Bank():
    crisis = False
    def create_atm(self):
        while not self.crisis:
            yield "$100"

hsbc = Bank()

corner_street_atm = hsbc.create_atm()
print(corner_street_atm.next())
print(corner_street_atm.next())
print([corner_street_atm.next() for cash in range(5)])
hsbc.crisis = True
try:
    print(corner_street_atm.next())
except Exception, e:
    print e
wall_street_atm = hsbc.create_atm()
try:
    print(wall_street_atm.next())
except Exception, e:
    print e
hsbc.crisis = False
try:
    print(corner_street_atm.next())
except Exception, e:
    print e
brand_new_atm = hsbc.create_atm()
# for cash in brand_new_atm:
    # print cash
