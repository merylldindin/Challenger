# Author:  DINDIN Meryll
# Date:    15/03/2019
# Project: featurizers

try: from featurizers.utils import *
except: from utils import *

class Featurize_1D:

    def __init__(self, signal, sampling_frequency=1):

        # Signal properties
        self.signal = np.asarray(signal)
        self.frequency = sampling_frequency
        self.period = 1 / self.frequency
        # Signal featurization
        self.feature = []
        self.columns = []

        # Initialize the configuration
        self._instantiate()

    def _instantiate(self):

        self.config = {

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

    def _update(self, config):

        self.config.update(config)

    def computeFFT(self):

        fft = np.abs(np.fft.rfft(self.signal))
        # Add to featurization
        self.feature += list(fft)
        self.columns += ['FFT_{}'.format(i) for i in range(len(fft))]
        # Memory efficiency
        del fft

    def computePeriodogram(self):

        f,s = sg.periodogram(self.signal, fs=self.frequency)
        # Add to featurization
        self.feature.append(f[s.argmax()])
        self.columns.append('MAX_FREQUENCY')
        self.feature.append(np.max(s))
        self.columns.append('MAX_POWER_SPECTRUM')
        self.feature.append(np.sum(s))
        self.columns.append('SUM_POWER_SPECTRUM')
        self.feature.append(np.mean(s))
        self.columns.append('MEAN_POWER_SPECTRUM')
        self.feature.append(np.std(s))
        self.columns.append('STD_POWER_SPECTRUM')
        self.feature.append(entropy(s))
        self.columns.append('ENTROPY_PERIODOGRAM')
        # Memory efficiency
        del f, s

    def computeSpectrogram(self):

        f,_,s = sg.spectrogram(self.signal, fs=self.frequency, return_onesided=True)
        # Add to featurization
        frq = f[s.argmax(axis=0)]
        self.feature += list(frq)
        self.columns += ['MAX_SLIDING_FREQUENCY_{}'.format(i) for i in range(len(frq))]
        psd = np.sum(s, axis=0)
        self.feature.append(np.max(psd))
        self.columns.append('MAX_SLIDING_POWER_SPECTRUM')
        self.feature.append(np.sum(psd))
        self.columns.append('SUM_SLIDING_POWER_SPECTRUM')
        self.feature.append(np.mean(psd))
        self.columns.append('MEAN_POWER_SPECTRUM')
        self.feature.append(np.std(psd))
        self.columns.append('STD_POWER_SPECTRUM')
        self.feature.append(entropy(psd))
        self.columns.append('ENTROPY_SPECTROGRAM')
        # Memory efficiency
        del f, s, frq, psd

    def frequencyBands(self, bands=[(0.5, 3), (3.5, 7.5), (7.5, 13), (14, 50)]):

        f,s = sg.periodogram(self.signal, fs=self.frequency)
        # Add to featurization
        for i, band in enumerate(bands):
            msk = np.where((f > band[0]) & (f < band[1]))[0]
            self.feature.append(np.sum(s[msk]))
            self.columns.append('SUM_SPECTRUM_BAND_{}'.format(i))
            self.feature.append(np.mean(s[msk]))
            self.columns.append('MEAN_SPECTRUM_BAND_{}'.format(i))
            self.feature.append(np.std(s[msk]))
            self.columns.append('STD_SPECTRUM_BAND_{}'.format(i))
        # Memory efficiency
        del f, s, msk

    def coefficientsAR(self):

        prm = AR(self.signal)
        prm = prm.fit()
        prm = prm.params
        # Add to featurization
        self.feature += list(prm)
        self.columns += ['PARAMS_AR_{}'.format(i) for i in range(len(prm))]
        # Memory efficiency
        del prm

    def crossingOver(self):
    
        sgn = np.sign(self.signal)
        sgn[sgn == 0] == -1
        # Add to featurization
        self.feature.append(len(np.where(np.diff(sgn))[0]))
        self.columns.append(['CROSSING_OVERS'])
        # Memory efficieny
        del sgn

    def computeWavelet(self, waveform='db4', level=5):

        coefficients = pywt.wavedec(self.signal, 'db4', level=5)
        # Add to featurization
        for i, signal in enumerate(coefficients):
            self.feature.append(np.min(signal))
            self.columns.append('MIN_WAVELET_{}'.format(i))
            self.feature.append(np.max(signal))
            self.columns.append('MAX_WAVELET_{}'.format(i))
            self.feature.append(np.sum(np.square(signal)))
            self.columns.append('POWER_WAVELET_{}'.format(i))
            self.feature.append(np.mean(signal))
            self.columns.append('MEAN_WAVELET_{}'.format(i))
            self.feature.append(np.std(signal))
            self.columns.append('STD_WAVELET_{}'.format(i))
        # Memory efficency
        del coefficients

    def computePolarity(self):

        sgn = np.sign(self.signal)
        sgn = np.split(sgn, np.where(np.diff(sgn) != 0)[0]+1)
        sgn = np.asarray([len(ele) for ele in sgn])
        # Add to featurization
        self.feature.append(np.mean(sgn))
        self.columns.append('MEAN_POLARITY')
        self.feature.append(np.std(sgn))
        self.columns.append('STD_POLARITY')
        # Memory efficiency
        del sgn

    def computeChaos(self):

        warnings.simplefilter('ignore')
        coe = nolds.lyap_e(self.signal)
        # Add to featurization
        self.feature.append(nolds.sampen(self.signal))
        self.columns.append('SAMPEN')
        self.feature.append(nolds.lyap_r(self.signal))
        self.columns.append('LYAP_R')
        self.feature += list(coe)
        self.columns += ['LYAP_E_{}'.format(i) for i in range(len(coe))]
        self.feature.append(nolds.hurst_rs(self.signal))
        self.columns.append('HURST_RS')
        self.feature.append(nolds.dfa(self.signal))
        self.columns.append('DFA')
        # Memory efficiency
        del coe

    def computeFractals(self):

        dif = np.diff(self.signal)
        dif = np.asarray([self.signal[0]] + list(dif))
        m_2 = float(np.sum(dif ** 2)) / self.signal.shape[0]
        t_p = np.sum(np.square(self.signal))
        m_4 = np.sum(np.square(np.diff(dif)))
        m_4 = m_4 / self.signal.shape[0]
        # Add to featurization
        self.feature.append(np.sqrt(m_2 / t_p))
        self.columns.append('HJORTH_COEFFICIENT')
        self.feature.append(np.sqrt(m_4 * t_p / m_2 / m_2))
        self.columns.append('FRACTAL_DIMENSION')
        # Memory efficiency
        del dif, m_2, t_p, m_4

    def signalDecomposition(self):

        arg =  {'model': 'additive', 'extrapolate_trend': 1, 'two_sided': True}
        dec = seasonal_decompose(self.signal, freq=self.frequency, **arg)
        # Add to featurization
        for name, signal in zip(['TREND', 'RESIDUAL'], [dec.trend, dec.resid]):
            self.feature.append(np.mean(signal))
            self.columns.append('MEAN_{}'.format(name))
            self.feature.append(np.std(signal))
            self.columns.append('STD_{}'.format(name))
            self.feature.append(entropy(signal))
            self.columns.append('ENTROPY_{}'.format(name))
            self.feature.append(kurtosis(signal))
            self.columns.append('KURTOSIS_{}'.format(name))
            self.feature.append(skew(signal))
            self.columns.append('SKEW_{}'.format(name))
        # Memory efficiency
        del dec

    def computeStatistics(self):

        # Add to featurization
        self.feature.append(np.min(self.signal))
        self.columns.append('MIN')
        self.feature.append(np.max(self.signal))
        self.columns.append('MAX')
        self.feature.append(np.mean(self.signal))
        self.columns.append('MEAN')
        self.feature.append(np.std(self.signal))
        self.columns.append('STD')
        self.feature.append(entropy(self.signal))
        self.columns.append('ENTROPY')
        self.feature.append(kurtosis(self.signal))
        self.columns.append('KURTOSIS')
        self.feature.append(skew(self.signal))
        self.columns.append('SKEW')
        self.feature.append(np.mean(self.signal) / np.std(self.signal))
        self.columns.append('ASYMMETRY')
        self.feature.append(np.sqrt(np.mean(np.square(self.signal))))
        self.columns.append('RMS')

        # Compute the percentiles
        for percentile in [1, 10, 25, 50, 75, 90, 95, 99]:
            self.feature.append(np.percentile(self.signal, percentile))
            self.columns.append('PERCENTILE_{}'.format(percentile))

    def mainFrequency(self, n_harm=1) :

        f,s = sg.welch(self.signal, self.frequency, scaling='spectrum')
        f,s = f[np.argsort(s)][::-1], sorted(s)[::-1]
        # Add to featurization
        self.feature.append(f[0])
        self.columns.append('MAIN_FREQUENCY')
        self.feature.append(s[0])
        self.columns.append('MAIN_FREQUENCY_SPECTRUM')
        # Memory efficiency
        del f, s

    def getFeatures(self, config=None):

        if not config is None: self._update(config)

        # Determine the functions that have to be launched
        lst = [key for key, value in self.config.items() if value]
        for function in lst: getattr(Featurize_1D, function)(self)

        dtf = pd.DataFrame([self.feature], columns=self.columns)
        dtf = dtf.astype('float32')

        return dtf

class Featurize_3D:

    def __init__(self, signals, sampling_frequency=1):

        assert signals.shape[0] == 3

        self.signals = signals
        self.frequency = sampling_frequency
        self.period = 1 / self.frequency

    def computeQuaternions(self):
        
        c = self.period * np.pi / 360
        quaternion = np.zeros((4, self.signals.shape[1]))
        quaternion[0,0] = 1

        for j in range(1, self.signals.shape[1]):

            r = quaternion[:,j-1]
            q = np.array([1, self.signals[0,j]*c, self.signals[1,j]*c, self.signals[2,j]*c])
            quaternion[0,j] = (r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3])
            quaternion[1,j] = (r[0]*q[1] + r[1]*q[0] - r[2]*q[3] + r[3]*q[2])
            quaternion[2,j] = (r[0]*q[2] + r[1]*q[3] + r[2]*q[0] - r[3]*q[1])
            quaternion[3,j] = (r[0]*q[3] - r[1]*q[2] + r[2]*q[1] + r[3]*q[0])
                    
        return quaternion

if __name__ == '__main__':

    signal = np.random.uniform(-1, 1, 360)
    dtf = Featurize_1D(signal, sampling_frequency=10).getFeatures()
    print(dtf)