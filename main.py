import argparse
from services.predict import predict
from core.build_index import build_index

parser = argparse.ArgumentParser()
parser.add_argument("--build", action="store_true")
parser.add_argument("--predict", type=str)

args = parser.parse_args()

if args.build:
    build_index()

if args.predict:
    predict(args.predict)
