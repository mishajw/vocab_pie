import itertools
import operator
import os
import sys
import hashlib
from typing import List, Tuple, NamedTuple, Optional

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_IGNORE_CASE = True
DEFAULT_OUTPUT_FILE = "vocab-pie.png"
DEFAULT_TITLE = "Vocabulary"
DEFAULT_MAX_LAYERS = 5
DEFAULT_MIN_DISPLAY_PERCENTAGE = 0.01
DEFAULT_MIN_LABEL_PERCENTAGE = 0.01
DEFAULT_LABEL_FONT_SIZE = 10
COLOR_MAP = plt.get_cmap("tab20")
DPI = 100


def create_from_file(
        file_path: str,
        ignore_case: bool = DEFAULT_IGNORE_CASE,
        output_file: str = DEFAULT_OUTPUT_FILE,
        prefix: str = None,
        title: str = DEFAULT_TITLE,
        max_layers: int = DEFAULT_MAX_LAYERS,
        min_display_percentage: float = DEFAULT_MIN_DISPLAY_PERCENTAGE,
        min_label_percentage: float = DEFAULT_MIN_LABEL_PERCENTAGE,
        label_font_size: int = DEFAULT_LABEL_FONT_SIZE):
    if not os.path.isfile(file_path):
        raise VocabPieError(f"File does not exist: {file_path}")
    if not os.access(file_path, os.R_OK):
        raise VocabPieError(f"Can not read file: {file_path}")

    with open(file_path, "r") as f:
        sentences = [line.strip() for line in f if line != ""]

    create_from_sentences(
        sentences, ignore_case, output_file, prefix, title, max_layers,
        min_display_percentage, min_label_percentage, label_font_size)


def create_from_sentences(
        sentences: List[str],
        ignore_case: bool = DEFAULT_IGNORE_CASE,
        output_file: str = DEFAULT_OUTPUT_FILE,
        prefix: str = None,
        title: str = DEFAULT_TITLE,
        max_layers: int = DEFAULT_MAX_LAYERS,
        min_display_percentage: float = DEFAULT_MIN_DISPLAY_PERCENTAGE,
        min_label_percentage: float = DEFAULT_MIN_LABEL_PERCENTAGE,
        label_font_size: int = DEFAULT_LABEL_FONT_SIZE):
    # Apply ignore case flag
    if ignore_case:
        sentences = [s.lower() for s in sentences]

    # Remove all sentences not containing prefix, and remove prefix from
    # remaining sentences
    if prefix is not None:
        sentences = [s[len(prefix):] for s in sentences if s.startswith(prefix)]

    # Create sentence hierarchy and pie chart layers
    hierarchy = Hierarchy.from_sentences(sentences)
    hierarchy.apply_min_percentages(
        min_display_percentage, min_label_percentage)
    hierarchy.fill_depth(hierarchy.get_max_depth() - 1)
    hierarchy.apply_max_layers(max_layers)
    layers = hierarchy.get_layers()

    figure, axes = plt.subplots()
    figure.set_size_inches(20, 20)
    figure.set_dpi(DPI)
    axes.axis('equal')

    # Calculate where the layers will start and end
    radius_start, radius_end = 0.1, 1.5
    num_layers = len(layers)
    layer_width = (radius_end - radius_start) / num_layers
    layer_starting_points = np.arange(
        radius_start, radius_end + layer_width, layer_width)

    for layer, layer_start in zip(layers, layer_starting_points):
        widths = [cell.percentage for cell in layer.cells]
        labels = [
            cell.label if cell.label is not None else ""
            for cell in layer.cells]
        colors = [__get_color(cell, num_layers) for cell in layer.cells]

        # `matplotlib` is awful, so this is how you get the cell labels to be
        # centered...
        label_distance = layer_start / (layer_start + (layer_width * 0.8))

        # Create the pie chart layer
        pie, texts = axes.pie(
            widths,
            radius=layer_start + layer_width,
            labels=labels,
            colors=colors,
            labeldistance=label_distance,
            pctdistance=1,
            rotatelabels=True)
        plt.setp(pie, width=layer_width, edgecolor='white')

        # Set the font size for all labels
        for text in texts:
            text.set_fontsize(label_font_size * DPI / 100)

    axes.set_title(
        title if prefix is None
        else f"{title}, sentences prefixed by \"{prefix}\"",
        fontsize=DPI * 0.3,
        pad=DPI * 0.75)
    figure.savefig(output_file)


class Hierarchy:
    def __init__(
            self,
            root: Optional[str],
            children: List[Tuple["Hierarchy", float]]):
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
            child_cell = LayerCell(child.__root, percentage, [], i)
            layers[0].cells.append(child_cell)

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
                    cell.parents.insert(0, child_cell)

            # Add to our layers
            for j, layer in enumerate(child_layers):
                if len(layers) <= j + 1:
                    layers.append(Layer([]))
                layers[j + 1].cells.extend(layer.cells)

        return layers

    def apply_min_percentages(
            self,
            min_display_percentage: float,
            min_label_percentage: float) -> None:
        # Remove children below `min_percentage`
        total_removed_percentage = 0

        def filter_fn(child_tuple: Tuple["Hierarchy", float]) -> bool:
            nonlocal total_removed_percentage
            _, child_percentage = child_tuple
            keep = child_percentage > min_display_percentage
            if not keep:
                total_removed_percentage += child_percentage
            return keep

        self.__children = list(filter(filter_fn, self.__children))

        # Remove label if `label_min_percentage` is greater than 1
        if min_label_percentage >= 1:
            self.__root = ""

        if total_removed_percentage != 0:
            self.__children.append(
                (Hierarchy(None, []), total_removed_percentage))

        for child, percentage in self.__children:
            child.apply_min_percentages(
                min_display_percentage / percentage,
                min_label_percentage / percentage)

    def get_max_depth(self) -> int:
        if not self.__children:
            return 1

        return 1 + max(c.get_max_depth() for c, _ in self.__children)

    def fill_depth(self, depth: int) -> None:
        if depth <= 0:
            return

        if not self.__children:
            self.__children.append((Hierarchy(None, []), 1))

        for child, _ in self.__children:
            child.fill_depth(depth - 1)

    def apply_max_layers(self, max_layers: int) -> None:
        if max_layers == 0:
            self.__children = []
        for child, _ in self.__children:
            child.apply_max_layers(max_layers - 1)


def __get_color(cell: "LayerCell", max_layers: int) -> np.array:
    if cell.label is None:
        return np.array([1] * 3)

    # Get the highest parent cell
    root_cell = cell.parents[0] if cell.parents else cell

    # The first category determines the base color to use
    root_label_hash = int(hashlib.md5(root_cell.label.encode()).hexdigest(), 16)
    base_color = np.array(COLOR_MAP(root_label_hash % COLOR_MAP.N))

    # On a scale of 0 to 1, how "pastelly" should the color be?

    # Colors become more "pastelly" as they go to outer layers
    layer_pastel_scale = len(cell.parents) / max_layers

    # Colors become more "pastelly" as they go across a layer inside a segment
    segment_pastel_scale = min(cell.index / 3, 1)

    # The final pastel scale is the layer scale, with the segment scale added on
    # but never exceeding the next layer's scale
    pastel_scale = layer_pastel_scale + segment_pastel_scale * (1 / max_layers)

    # We make colors more "pastelly" by squashing the color space from `[0, 1]`
    # to `[new_zero, 1]`, where `new_zero` is between `[0, 0.5]`
    new_zero = pastel_scale

    # We apply this new color space to the color
    return (base_color * (1 - new_zero)) + new_zero


class LayerCell:
    def __init__(
            self,
            label: str,
            percentage: float,
            parents: List["LayerCell"],
            index: int):
        self.label = label
        self.percentage = percentage
        self.parents = parents
        self.index = index


class Layer(NamedTuple):
    cells: List[LayerCell]


class VocabPieError(BaseException):
    def __init__(self, message: str):
        self.__message = message

    def display(self):
        print("Error: " + self.__message, file=sys.stderr)
