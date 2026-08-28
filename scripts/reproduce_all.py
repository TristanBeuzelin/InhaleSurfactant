from itertools import product

import newton_raphson
import nodes_alim
import single_injection
import splitting_factors


def main():
    """Reproduces all figures at once."""
    print("Single injection...")
    figures = ["a", "b", "c", "d", "e", "f"]
    viscosities = [3e-2, 3e-1, 1.0]
    TYPES = ["infant", "adult"]
    for figure, (type, viscosity) in zip(figures, product(TYPES, viscosities)):
        single_injection.main(figure=figure, type=type, viscosity=viscosity)
    print("Single injection done. Figure 1 plot reproduced.")
    print("Spltting factors graph...")
    splitting_factors.main(TYPE="infant")
    splitting_factors.main(TYPE="adult")
    print("Figure 2 reproduced.")
    print("Final nodes alimentation...")
    nodes_alim.main()
    print("Figure 3.a reproduced.")
    print("Newton-Raphson...")
    newton_raphson.main()
    print("Figure 3.b reproduced.")


if __name__ == "__main__":
    main()
