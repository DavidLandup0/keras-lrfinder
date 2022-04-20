# keras-lrfinder
Keras Implementation of Leslie Smith's method for finding an optimal starting learning rate.

Combined implementation of [Pavel Surmenok](https://github.com/surmenok/keras_lr_finder), [Adrian Rosebrock](https://pyimagesearch.com/2019/08/05/keras-learning-rate-finder/) and [ktrain](https://github.com/amaiya/ktrain) into a single file.

### Usage


```python
from keraslrfinder import LearningRateFinder

# Compiled model
model = build_model()

# Optional `beta` argument applies smoothing, default is 0.7, which is fairly smooth
lr_finder = LearningRateFinder(model)
# Supports both generators and separate NumPy arrays
# find(X_train, y_train)
# find(generator)
lr_finder.find(train_generator, startLR=1e-10, endLR=10, epochs=3, stepsPerEpoch=np.ceil((len(train_generator.classes) / float(32))))
```

```python
lr_finder.plot_loss()
```

![](https://i.imgur.com/L8H4Ryz.png)


```python
# sma applies smoothing
lr_finder.plot_loss_change(sma=80, y_lim=(-0.1, 0.1))
```

![](https://i.imgur.com/FQTbjW1.png)

```python
# Get best LR (steepest loss)
lr_finder.get_best_lr(sma=80) # 0.0022606123
```

