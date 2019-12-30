import types

class InputProvider(object):
    def __init__(self, message: str, type, infinite: bool = True):
        self.message = message
        self.type = type
        self.infinite = infinite
    
    def input(self):
        while True:
            inp = input(self.message)

            try:
                if isinstance(self.type(inp), self.type):
                    return inp
            except ValueError:
                pass

            if self.infinite:
                print('Provide correct value!\n')
            else:
                return -1
        
    