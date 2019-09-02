# Challenger - Featurization

This submodule is focused on signal featurization, which is greatly appreciated for fast-prototyping with interpretable models. The idea is to automate feature that do usually make sense for time-series, 3D inertial sensors and so on.

## Run it for time-series

```python
# Import the featurizer
from featurizers import Featurize_1D
# Build a mock signal
sig = np.random.uniform(0, 1, 1250)
# Load the dataset
fea = Featurize_1D(sig, sampling_frequency=125).getFeatures()
```

Currently, there are **750 features** outputed by the exhaustive featurization. You can check out the (TDA toolbox)[https://github.com/Coricos/TdaToolbox] if you want to add more exotic features. Among the possible computations, here is the list of functions associated with the Featurize_1D object. This dictionnary configuration is translated to computation via the object directly for easier modularity. 

```bash
{
      'computeFFT': True,
      'computePeriodogram': True,
      'computeSpectrogram': True,
      'frequencyBands': True,
      'coefficientsAR': True,
      'crossingOver': True,
      'computeWavelet': True,
      'computePolarity': True,
      'computeChaos': True,
      'computeFractals': True,
      'signalDecomposition': True,
      'computeStatistics': True,
      'mainFrequency': True
}
```

One would then find the real FFT transformation of the signal, its periodogram and spectogram, from which features such as max frequencies or spectrum bands are computed. You will also find the coefficients of a fitted auto-regressive model, a specific wavelet decomposition, the signal polarity, the signal chaotic features (Hurst, DFA, Lyapunov, ...) and fractal features (fractal dimension and Hjorth dimension). All of this is added to mainstream and usual features, which finally account for those 750 data-points.

The output being a pandas DataFrame, it becomes straight-forward to manipulate the output, to both filter or engineer it.
