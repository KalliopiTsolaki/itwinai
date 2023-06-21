import argparse
from torchvision.datasets import MNIST

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocessing of MNIST dataset.")
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Path where to store preprocessed datasets.",
        default=None
    )
    # Syntactic sugar
    parser.add_argument(
        "-s", "--stage",
        type=str,
        help="Kind of dataset split to use.",
        default='train',
        choices=('train', 'test')
    )
    args = parser.parse_args()
    print(
        """
    ******************************
    * Called GAN preprocessing *
    ******************************

    - Fake preprocessing step
    - nothing is actually done

    """
    )
