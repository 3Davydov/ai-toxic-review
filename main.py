import argparse
from pathlib import Path

from toxic_clf.data import load_dataset, prepare, save_dataset
from toxic_clf.models import classifier


def main():
    args = parse_args()
    args.func(args)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')

    default_raw_train_data_path = Path('toxic_clf/models/code-review-dataset-full.xlsx')
    default_raw_test_data_path = Path('toxic_clf/models/test.xlsx')
    default_prepared_train_data_path = Path('toxic_clf/models/prepared-train-code-review-dataset-full')
    default_prepared_test_data_path = Path('toxic_clf/models/prepared-test-code-review-dataset-full')
    prepare_data_parser = subparsers.add_parser('prepare-data')
    prepare_data_parser.set_defaults(func=prepare_data)
    prepare_data_parser.add_argument(
        'input_train',
        nargs='?',
        help='Path to load raw dataset',
        type=Path,
		default=default_raw_train_data_path,
    )
    prepare_data_parser.add_argument(
        'input_test',
        nargs='?',
        help='Path to load raw dataset',
        type=Path,
		default=default_raw_test_data_path,
    )
    prepare_data_parser.add_argument(
        '--output_train',
        nargs='?',
        help='Path to save prepared dataset to',
        type=Path,
        default=default_prepared_train_data_path,
    )
    prepare_data_parser.add_argument(
        '--output_test',
        nargs='?',
        help='Path to save prepared dataset to',
        type=Path,
        default=default_prepared_test_data_path,
    )

    predict_parser = subparsers.add_parser('classify')
    predict_parser.set_defaults(func=classify)
    predict_parser.add_argument(
        '--train_dataset',
        nargs='?',
        help='Path to prepared dataset',
        type=Path,
        default=default_prepared_train_data_path,
    )
    predict_parser.add_argument(
        '--test_dataset',
        nargs='?',
        help='Path to prepared dataset',
        type=Path,
        default=default_prepared_test_data_path,
    )
    predict_parser.add_argument(
        '-m',
        '--model',
        choices=['classic_ml', 'roberta'],
        default='classic_ml',
    )

    return parser.parse_args()


def prepare_data(args):
    train_dataset = prepare(args.input_train)
    test_dataset = prepare(args.input_test)
    save_dataset(train_dataset, args.output_train)
    save_dataset(test_dataset, args.output_test)


def classify(args):
    train_dataset = load_dataset(args.train_dataset)
    test_dataset = load_dataset(args.test_dataset)
    classifier(train_dataset, test_dataset, args.model)


if __name__ == '__main__':
    main()
