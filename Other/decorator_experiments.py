# Noah-Manuel Michael
# Created: 06.05.2023
# Last updated: 12.05.2023
# Experiments with decorators

def my_first_decorator(original_function):
    def wrapper(*args, **kwargs):
        my_list = ['0th_list_space']
        original_result = original_function(*args, **kwargs)
        my_list.extend([original_result, '2nd_list_space'])
        return my_list
    return wrapper


@my_first_decorator
def easy_original_function(str_input: str = '1st_list_space'):
    list_space = str_input
    return list_space


print(easy_original_function())
print(easy_original_function('XXX'))

########################################################################################################################
proper_measurements = {}
print('proper measurements before adding stupid measurements with decorators:')
print(proper_measurements)


def add_transformed_measurement_to_proper_measurements(original_function):
    def wrapper(*args, **kwargs):
        feature_value = original_function(*args, **kwargs)
        proper_measurements[original_function.__name__.lstrip('get_')] = feature_value
    return wrapper


@add_transformed_measurement_to_proper_measurements
def get_height_in_cm(height_in_km: float):
    height_in_cm = height_in_km * 100000
    return height_in_cm


@add_transformed_measurement_to_proper_measurements
def get_weight_in_kg(weight_in_g: int):
    weight = weight_in_g / 1000
    return weight


stupid_measurements = {'height_in_km': 0.00183, 'weight_in_g': 82400}
print('stupid measurements:')
print(stupid_measurements)

get_height_in_cm(stupid_measurements['height_in_km'])
get_weight_in_kg(stupid_measurements['weight_in_g'])

print('proper measurements after adding stupid measurements with decorators:')
print(proper_measurements)
