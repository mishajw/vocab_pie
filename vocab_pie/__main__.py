import itertools
import operator
import os
import sys
from typing import List, Tuple, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from configargparse import ArgumentParser

DEFAULT_IGNORE_CASE = True
DEFAULT_OUTPUT_FILE = "vocab-pie.png"
COLOR_MAP = plt.get_cmap("tab10")


def __main():
    parser = ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--ignore-case", type=bool, default=DEFAULT_IGNORE_CASE)
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE)
    args = parser.parse_args()

    try:
        create_from_file(args.file)
    except VocabPieError as e:
        e.display()


def create_from_file(
        file_path: str,
        ignore_case: bool = DEFAULT_IGNORE_CASE,
        output_file: str = DEFAULT_OUTPUT_FILE):
    if not os.path.isfile(file_path):
        raise VocabPieError(f"File does not exist: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise VocabPieError(f"Can not read file: {file_path}")

    with open(file_path, "r") as f:
        sentences = [line.strip() for line in f if line != ""]

    create_from_sentences(sentences, ignore_case, output_file)


def create_from_sentences(
        sentences: List[str],
        ignore_case: bool = DEFAULT_IGNORE_CASE,
        output_file: str = DEFAULT_OUTPUT_FILE):
    # Apply ignore case flag
    if ignore_case:
        sentences = [s.lower() for s in sentences]

    # Create sentence hierarchy and pie chart layers
    hierarchy = Hierarchy.from_sentences(sentences)
    layers = hierarchy.get_layers()

    figure, axes = plt.subplots()
    axes.axis('equal')

    # Calculate where the layers will start and end
    radius_start, radius_end = 0.2, 1.0
    num_layers = len(layers)
    layer_width = (radius_end - radius_start) / num_layers
    layer_starting_points = np.arange(
        radius_start, radius_end + layer_width, layer_width)

    for layer, layer_start in zip(layers, layer_starting_points):
        widths = [cell.percentage for cell in layer.cells]
        labels = [cell.label for cell in layer.cells]
        colors = [__get_color(cell, num_layers) for cell in layer.cells]

        # `matplotlib` is awful, so this is how you get the cell labels to be
        # centered...
        label_distance = layer_start / (layer_start + (layer_width * 0.6))

        # Create the pie chart layer
        pie, texts = axes.pie(
            widths,
            radius=layer_start + layer_width,
            labels=labels,
            colors=colors,
            labeldistance=label_distance,
            pctdistance=1)
        plt.setp(pie, width=layer_width, edgecolor='white')

        # Set the font size for all labels
        for text in texts:
            text.set_fontsize(8)

    figure.savefig(output_file)


class Hierarchy:
    def __init__(self, root: str, children: List[Tuple["Hierarchy", float]]):
        self.__root = root
        self.__children = children

        # Normalize values to sum to one, creating percentages
        total_percentages = sum(p for _, p in self.__children)
        self.__children = [
            (c, p / total_percentages) for c, p in self.__children]

        # Sort children by their percentage
        self.__children = sorted(self.__children, key=lambda t: t[1])

    @classmethod
    def from_sentences(cls, sentences: List[str]):
        # Split sentences into (non-empty) words
        sentences = [[w for w in s.split() if w != ""] for s in sentences]

        # Delegate to private method
        return cls.__from_sentences(sentences, "")

    @classmethod
    def __from_sentences(cls, sentences: List[List[str]], root: str):
        # Remove empty sentences
        sentences = [s for s in sentences if s]

        # Group sentences by their first word
        grouped = itertools.groupby(
            sorted(sentences, key=operator.itemgetter(0)),
            key=operator.itemgetter(0))

        # Recurse with the grouped sentences, with the first word taken off
        children = []
        num_sentences = len(sentences)
        for first_word, grouped_sentences in grouped:
            grouped_sentences = [s[1:] for s in grouped_sentences]
            children.append((
                Hierarchy.__from_sentences(grouped_sentences, first_word),
                len(grouped_sentences) / num_sentences))

        return Hierarchy(root, children)

    def get_layers(self) -> List["Layer"]:
        """
        Get the layers for making a pie chart

        These layers do not contain the root node.

        Returns a list of layers, with each layer containing a list of slices
        with a label and a percentage. The percentages in each layer sum to 1.
        """

        layers: List["Layer"] = [Layer([])]
        for i, (child, percentage) in enumerate(self.__children):
            # Add a cell for this child
            layers[0].cells.append(LayerCell(child.__root, percentage, [i]))

            # Get the child's layers, and modify them to fit with the rest of
            # the layers
            child_layers = child.get_layers()
            for layer in child_layers:
                for cell in layer.cells:
                    # Reduce the size of the child's layers so that they fit in
                    # the child's percentage
                    cell.percentage *= percentage
                    # Add the child's color index to the child's layer's color
                    # chain
                    cell.color_chain.insert(0, i)

            # Add to our layers
            for j, layer in enumerate(child_layers):
                if len(layers) <= j + 1:
                    layers.append(Layer([]))
                layers[j + 1].cells.extend(layer.cells)

        return layers


def __get_color(cell: "LayerCell", max_layers: int) -> np.array:
    # The first category determines the base color to use
    base_color = np.array(COLOR_MAP(cell.color_chain[0]))

    # On a scale of 0 to 1, how "pastelly" should the color be?

    # Colors become more "pastelly" as they go to outer layers
    layer_pastel_scale = (len(cell.color_chain) - 1) / max_layers

    # Colors become more "pastelly" as they go across a layer inside a segment
    segment_pastel_scale = min(cell.color_chain[-1] / 3, 1)

    # The final pastel scale is the layer scale, with the segment scale added on
    # but never exceeding the next layer's scale
    pastel_scale = layer_pastel_scale + segment_pastel_scale * (1 / max_layers)

    # We make colors more "pastelly" by squashing the color space from `[0, 1]`
    # to `[new_zero, 1]`, where `new_zero` is between `[0, 0.5]`
    new_zero = pastel_scale

    # We apply this new color space to the color
    return (base_color * (1 - new_zero)) + new_zero


class LayerCell:
    def __init__(self, label: str, percentage: float, color_chain: List[int]):
        self.label = label
        self.percentage = percentage
        self.color_chain = color_chain


class Layer(NamedTuple):
    cells: List[LayerCell]


class VocabPieError(BaseException):
    def __init__(self, message: str):
        self.__message = message

    def display(self):
        print("Error: " + self.__message, file=sys.stderr)


if __name__ == "__main__":
    __main()
