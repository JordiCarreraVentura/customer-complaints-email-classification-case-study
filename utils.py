
def get_credential(path):
    with open(path) as rd:
        return rd.read().strip()

