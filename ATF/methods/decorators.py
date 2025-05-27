def output_count(n):
    def decorator(func):
        func.output_count = n
        return func
    return decorator