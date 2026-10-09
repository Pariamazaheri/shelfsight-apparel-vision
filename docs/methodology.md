# Modeling methodology

## Decision context

An apparel catalog team needs reliable category suggestions and a way to inspect ambiguous products. Fashion-MNIST provides an established ten-category benchmark for measuring these capabilities. Its 28-by-28 grayscale thumbnails are much simpler than real product photography; the project is a reproducible benchmark and inference prototype, not a validated live-catalog service.

## Evaluation contract

Keep the official 10,000-image test set intact. Hash the original pixels, remove within-development duplicates and exclude development images that exactly match the test set. This audit uses test pixels only to prevent contamination; no test labels influence training, sampling, normalization, configuration selection or stopping. Stratify a seeded 15,000-image development sample, split into 12,000 training and 3,000 validation images. Remaining development images are unused, explicitly limiting the experiment budget.

Pixel mean and standard deviation come exclusively from training images. A nine-run CNN comparison varies kernel size, initial convolution stride, filter count, max/average pooling, pooling window size, depth and augmentation. The base network has three convolutions, two pooling stages, adaptive 3-by-3 pooling and a dense classification head. Parameter counts and measured CPU training seconds accompany each configuration. All variants use Adam, learning rate 0.001, weight decay 0.0001, batch size 128, clipping at 5 and a maximum of eight epochs. Restore the best validation cross-entropy checkpoint; stop after three unimproved epochs.

Choose the custom CNN by validation macro F1 before examining test scores. Evaluate that winner plus the predeclared base/augmentation references, a class-prior reference, a linear-pixel SGD classifier, and frozen MobileNet transfer features. Accuracy, macro F1, log loss and ten-bin expected calibration error characterize different failure modes. Confidence is not guaranteed to be calibrated. Pixel-hash cluster bootstraps (500 draws) provide macro-F1 intervals and account for exact duplicates inside the official test set; near-duplicate dependence and seed variation remain unmeasured.

## Augmentation and transfer

Training augmentation combines horizontal flips, small ±8-degree rotations and padded random crops. Validation/test transforms never augment. These transformations may obscure fine garment distinctions at this resolution. Training curves compare a predeclared augmented run with the unaugmented base; both have the same epoch budget. Any performance drop is evidence about this recipe and budget, not proof that augmentation is generally harmful.

Frozen transfer uses torchvision's MobileNetV3-Small, ImageNet1K V1 weights, and the official resize-to-256/center-crop-224/ImageNet-normalization transform. Replicate grayscale into three channels without inventing color. Freeze all pretrained feature layers, discard the original 1,000-class classifier, and extract 576 pooled features per image. Fit a training-only StandardScaler and multinomial logistic head (`C=0.1`). No feature extraction updates the backbone or batch-normalization statistics. Validation results and weight URL are retained in `transfer_metadata.json`.

MobileNet is small and provides a practical fixed-feature baseline, but natural-image pretraining does not guarantee superiority on low-resolution grayscale apparel. The custom CNN's runtime measures training and validation; the transfer runtime also includes preprocessing and extraction for all three partitions plus head fitting. These scopes differ and must not be interpreted as an apples-to-apples training-speed comparison. The linear reference is bounded to 100 SGD iterations and reached that limit in this run; it is not a fully optimized linear benchmark.

Convolution shares filters across spatial positions, reducing the number of separately learned pixel interactions. Pooling adds local tolerance while potentially erasing important garment details. More depth/filters increases capacity and cost without guaranteeing better validation quality. The committed experiment commentary connects these tradeoffs to measured outcomes, rather than treating a larger architecture as an automatic improvement.
