def projectile_range(v0, theta, g=9.81):
    """Range of projectile (no air drag), theta in radians."""
    import math

    return (v0**2 * math.sin(2 * theta)) / g
