
import dis

def test():
    for _ in [1,2,3]:
        pass

dis.dis(test)
