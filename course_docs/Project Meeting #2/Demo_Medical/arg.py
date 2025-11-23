import argparse


argparser = argparse.ArgumentParser(description="Demo for argument parsing")


argparser.add_argument(
    "--example_arg",
    type=str,
    required=True,
    # default="default_value",
)

argparser.add_argument(
    "--another_arg",
    type=int,
    required=True,
)


def get_arguments():
    return argparser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    print(f"Example Arg: {args.example_arg}, {type(args.example_arg)}")
    print(f"Another Arg: {args.another_arg}, {type(args.another_arg)}")