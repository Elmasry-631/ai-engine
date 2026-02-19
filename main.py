import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI engine CLI")
    parser.add_argument("--build", action="store_true", help="Build FAISS index")
    parser.add_argument("--predict", type=str, help="Path to image for prediction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.build:
        from core.build_index import build_index

        build_index()

    if args.predict:
        from services.predict import predict

        predict(args.predict)




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI engine CLI")
    parser.add_argument("--build", action="store_true", help="Build FAISS index")
    parser.add_argument("--predict", type=str, help="Path to image for prediction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.build:
        from core.build_index import build_index

        build_index()

    if args.predict:
        from services.predict import predict

        predict(args.predict)



from core.build_index import build_index
from services.predict import predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI engine CLI")
    parser.add_argument("--build", action="store_true", help="Build FAISS index")
    parser.add_argument("--predict", type=str, help="Path to image for prediction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.build:
        build_index()

    if args.predict:
        predict(args.predict)


if __name__ == "__main__":
    main()
