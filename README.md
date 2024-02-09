# XAI-Model

- Model with dichotomized features: finding the cut-off thresholds and weights
- Augmentation of model features: adding combinations of features which are binarized
- Refitting the intercept when weights are fixed
- Model with piecewise weights
- Mixed model of a logistic regression and a generalized additive model

## Algorithms

### Constructing baseline model

The baseline model includes binarized individual features.
The weights and cut-off thresholds are found by the following way.

The first step is to construct univariate models for each feature.
To this end, we find a weight and a cut-off threshold.
The weight is equal to width of border area, where suspected points are situated.
At the given weight, we have a border area Π. Let define a property Φ
of points at the border area. This property is selected in such a way that
allows to predict the "1" class at the border area.
More precise, the relation TPV/FPV (at the area Π∩Φ) is to be as high as possible.

### Improving the model

At the second step, we have the multivariate model, and feature-wise model enhancement
can be applied. Let remove an individual feature which is selected randomly and add it again to the model
with adjusted weight and cut-off threshold. This procedure is repeated until the stabilization
of coefficients.

### Augmentation of model features

Border area method is proposed here to create new binary features which are determined by a few features.
