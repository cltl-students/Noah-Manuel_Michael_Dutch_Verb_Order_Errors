# Noah-Manuel Michael
# Created: 20.06.2023
# Last updated: 20.06.2023
# Define better print method


def pyprint(element):
    """
    Pass function or variable. Print function or variable name followed by value of return statement of function or 
    value of variable.
    :param element: function or variable
    :return: None
    """
    try:
        assert all([type(element) != str, type(element) != bool, type(element) != int, type(element) != float,
                    type(element) != list, type(element) != dict, type(element) != set])
        variable_name: str = [element for element in str(element).split()][1]
        print(f'{variable_name}: {str(element())}')
    except AssertionError:
        variable_name: str = [name for name, value in locals().items() if value == element][0]
        print(f'{variable_name}: {element}')


def pynprint(element):
    """
    Pass function or variable. Print function or variable name followed by new line followed by value of return 
    statement of function or value of variable.
    :param element: function or variable
    :return: None
    """
    try:
        assert all([type(element) != str, type(element) != bool, type(element) != int, type(element) != float,
                    type(element) != list, type(element) != dict, type(element) != set])
        variable_name: str = [element for element in str(element).split()][1]
        print(f'{variable_name}:\n{str(element())}')
    except AssertionError:
        variable_name: str = [name for name, value in locals().items() if value == element][0]
        print(f'{variable_name}:\n{element}')


# def test():
#     def allo():
#         hallo = 'this is a function'
#         return hallo
# 
#     ballo = 'this is a variable'
#     callo = 0
#     dallo = 0.12
#     fallo = True
#     
#     pyprint(allo)
#     pyprint(ballo)
#     pyprint(callo)
#     pyprint(dallo)
#     pyprint(fallo)
#     
#     pynprint(allo)
#     pynprint(ballo)
#     pynprint(callo)
#     pynprint(dallo)
#     pynprint(fallo)
