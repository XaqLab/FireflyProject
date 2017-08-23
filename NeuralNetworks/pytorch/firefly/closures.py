# A few useful closures.

def closure(function, data, *args, **kwargs):
    """ A generic closure for remembering data between function calls. """
    enclosed_data = data
    return lambda *args, **kwargs: function(enclosed_data, *args, **kwargs)


def in_a_row(f, count):
    """ A closure that keeps track of the number of times in a row f evaluates
    to True. """
    counter = count
    def g(*args, **kwargs):
        if f(*args, **kwargs):
            counter[0] += 1
        else:
            counter[0] = 0
        #print(counter[0], ", ", end="")
        return counter[0]
    return g


def increment(x):
    # modify x so this function can be used in a closure
    x[0] = x[0] + 1
    return x[0]


def append(x,y):
    # modify x so this function can be used in a closure
    x[0] = x[0] + [y]
    return x[0]


if __name__ == '__main__':
    # Example: counting 
    count = closure(increment, [0])
    print(count())
    print(count())
    print(count())
    
    # Example: accumulate arguments
    accumulate = closure(append, [[]], 0)
    print(accumulate(1))
    print(accumulate(2))
    print(accumulate(3))
    
    # Example: count number of consecutive ones
    one = lambda x : x == 1
    ones_in_a_row = in_a_row(one, [0], 0)
    print(ones_in_a_row(1))
    print(ones_in_a_row(1))
    print(ones_in_a_row(1))
    print(ones_in_a_row(0))
    print(ones_in_a_row(1))
    print(ones_in_a_row(1))
