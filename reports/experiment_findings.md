# Experiment findings

## CNN design under a fixed budget

All nine comparisons use the same 12,000 training images, 3,000 validation images, optimizer and eight-epoch budget. Values below are validation results; the test set does not choose the design.

| Configuration | Macro F1 | Parameters | CPU training seconds | Interpretation |
|---|---:|---:|---:|---|
| Base: three convolutions, max pool 2 | 0.8046 | 60,874 | 21.4 | Reference capacity and spatial resolution |
| Kernel 5 | 0.8347 | 102,090 | 35.8 | Larger receptive fields improve quality here at greater cost |
| Initial stride 2 | 0.8512 | 60,874 | 11.6 | Earlier spatial reduction is effective within this budget |
| More filters | 0.8418 | 167,114 | 34.9 | Added capacity helps over base but does not win |
| Average pooling | 0.7734 | 60,874 | 18.4 | Smoothing loses useful local category evidence in this run |
| Pooling window 3 | **0.8534** | 60,874 | 12.8 | Validation winner; favorable bounded capacity/compute tradeoff |
| Two convolutions | 0.7825 | 23,946 | 13.9 | Smaller representation underperforms the base |
| Four convolutions | 0.8181 | 97,802 | 26.7 | More depth costs more without matching the best pooling design |
| Base plus augmentation | 0.7578 | 60,874 | 21.4 | This recipe/budget underperforms, not evidence against augmentation generally |

Changes in depth/stride/pooling alter effective capacity and spatial information; single-seed observations cannot establish the mechanism causally. Parameters describe model capacity, while online training loss and validation loss reflect different dropout/augmentation conditions. Most runs reached the budget limit, so the table does not establish converged performance.

## Transfer and held-out performance

Frozen MobileNetV3-Small features plus a trained linear head achieve validation macro F1 **0.8938** and test macro F1 **0.8983**, above the selected custom CNN's **0.8541**. The custom CNN has **85.62%** held-out accuracy and a pixel-cluster bootstrap macro-F1 interval of approximately **[0.848, 0.860]**. Transfer achieves **89.83%** accuracy and interval **[0.893, 0.904]**. Intervals reflect test sampling, not model-selection uncertainty or seed variability.

MobileNet uses natural-image pretraining, official ImageNet preprocessing and replicated grayscale inputs. Its measured 402-second transfer runtime includes feature extraction for all 25,000 images plus head fitting. The custom winner's 12.8 seconds includes eight training/validation epochs only. This difference is not a direct training-speed comparison.

## Operational failure modes

The selected CNN's Shirt recall is **54.2%**, while Trouser recall is **98.0%**. Overall accuracy conceals difficulty distinguishing shirts, pullovers, coats and tops. The confidence gallery shows wrong classifications even with near-one softmax confidence. The selected CNN's aggregate ten-bin ECE is 0.0074; this aggregate calibration statistic does not make every individual prediction trustworthy.

The bounded linear-pixel reference achieves 82.59% accuracy but substantially worse probability log loss (3.052) and ECE (0.129). It reached its iteration limit, so it should not be described as an optimized or calibrated baseline. Accuracy alone is insufficient when model confidence determines automation.

The fixed augmentation recipe reduces test macro F1 from 0.8044 to 0.7623 at the same epoch budget. Small rotations/crops can remove fine shape evidence from these thumbnails; augmented training may also need more epochs. Further experiments should distinguish these explanations rather than claim a universal effect.

The packaged inference command deliberately exposes the selected custom CNN as a local benchmark prototype. Transfer is an evaluated reference, not a separately served model. Real catalog deployment needs representative product photography and taxonomy validation; thumbnail benchmark scores do not establish commercial performance.
