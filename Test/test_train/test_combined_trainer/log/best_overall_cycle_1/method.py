from ATF.methods.decorators import method_profile

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group='math',
    weight=1.0,
    vectorized=False
)
def add_one(x):
    return [x + 1]

@method_profile(
    output_count=1,
    input_types=['scalar', 'scalar'],
    output_types=['scalar'],
    group='math',
    weight=1.0,
    vectorized=False
)
def multiply(a, b):
    return [a * b]

@method_profile(
    output_count=1,
    input_types=['scalar'],
    output_types=['scalar'],
    group='math',
    weight=1.0,
    vectorized=False
)
def identity(x):
    return [x]