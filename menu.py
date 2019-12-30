from typing import Sequence, Union
import types
from input_provider import InputProvider
import sys

class Menu(object):
    def __init__(self, options: Sequence, functions: Union[callable, Sequence[callable]], menu_text: str = None, has_exit: bool = False, has_back: bool = False):
        self.additional_optional_count = 0
        self.options = options
        self.functions = functions
        self.menu_text = menu_text
        self.has_back = has_back
        self.has_exit = has_exit

        if has_back:
            self.additional_optional_count += 1
        if has_exit:
            self.additional_optional_count += 1
    
    def select(self) -> int:

        while True:
            selected = int(InputProvider('Choice: ', int, False).input())

            if selected > len(self.options) + self.additional_optional_count or selected <= 0:
                print('Such option does not exist!\n')
                self.display()
                continue
            elif selected > len(self.options):
                if self.has_back:
                    return 
                if self.has_exit:
                    sys.exit()
            
            if callable(self.functions):
                self.functions(selected)
            else:
                self.functions[selected - 1](selected)

            return selected
    
    def display(self):
        if self.menu_text != None:
            print(self.menu_text)

        for i in range(len(self.options)):
            print('[{}] - {}'.format(i + 1, self.options[i]))
        
        count = 0
        if self.has_back:
                print('[{}] - {}'.format(len(self.options) + 1, 'Back'))
                count += 1
        if self.has_exit:
                print('[{}] - {}'.format(len(self.options) + count + 1, 'Exit'))