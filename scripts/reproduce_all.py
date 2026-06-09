import single_injection
import splitting_factors
import nodes_alim
import newton_raphson


def main():
    """Reproduces all figures at once."""
    single_injection.main()
    splitting_factors.main()
    nodes_alim.main()
    newton_raphson.main()


if __name__ == "__main__":
    main()
