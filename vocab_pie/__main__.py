from configargparse import ArgumentParser

from vocab_pie.creator import DEFAULT_IGNORE_CASE, DEFAULT_OUTPUT_FILE, \
    DEFAULT_TITLE, DEFAULT_MAX_LAYERS, DEFAULT_MIN_DISPLAY_PERCENTAGE, \
    DEFAULT_MIN_LABEL_PERCENTAGE, DEFAULT_LABEL_FONT_SIZE, create_from_file, \
    VocabPieError


def main():
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--ignore-case", type=bool, default=DEFAULT_IGNORE_CASE)
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--title", type=str, default=DEFAULT_TITLE)
    parser.add_argument("--max-layers", type=int, default=DEFAULT_MAX_LAYERS)
    parser.add_argument(
        "--min-display-percentage",
        type=float,
        default=DEFAULT_MIN_DISPLAY_PERCENTAGE)
    parser.add_argument(
        "--min-label-percentage",
        type=float,
        default=DEFAULT_MIN_LABEL_PERCENTAGE)
    parser.add_argument(
        "--label-font-size", type=int, default=DEFAULT_LABEL_FONT_SIZE)
    args = parser.parse_args()

    try:
        create_from_file(
            args.file,
            args.ignore_case,
            args.output_file,
            args.prefix,
            args.title,
            args.max_layers,
            args.min_display_percentage,
            args.min_label_percentage,
            args.label_font_size)
    except VocabPieError as e:
        e.display()


if __name__ == "__main__":
    main()
