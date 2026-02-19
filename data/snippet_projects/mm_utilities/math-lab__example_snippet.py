def derivative_central(f, x, h=1e-6):
    """Central difference derivative approximation."""
    return (f(x + h) - f(x - h)) / (2 * h)
