# Data provenance

Source: [Kaggle Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist), the Zalando Research mirror of [the original Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist). The original dataset contains 60,000 training and 10,000 test grayscale apparel images, with 10 categories.

Kaggle lists the mirror license as “Other (specified in description).” The upstream project specifies **MIT**; the complete original license is retained in `FASHION_MNIST_LICENSE.txt`, including Zalando's copyright notice. This distinction is preserved in `source_metadata.json` rather than relabeling the mirror metadata.

`shelfsight download` retrieves the Kaggle archive and stores CSV files under ignored `data/raw/`. The benchmark uses a stratified 12,000-image training sample and 3,000 validation images from the original training pool, with the full 10,000-image official test partition. Exact duplicate pixels are removed within development data; any development image matching the official test set is excluded before sampling. Near duplicates are not removed.

`example_apparel.png` is one training image from Fashion-MNIST, provided under the retained MIT license to demonstrate inference. It is not an additional test observation.
