# This script computes the MFCC features for automatic speech recognition
#
# You need to complete the part indicated by #### so that the code can produce
# sensible results.
#
# Ning Ma (n.ma@sheffield.ac.uk)
#

import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import numpy as np
from scipy.fftpack import dct
import numpy


def freq2mel(freq):
	"""Convert Frequency in Hertz to Mels

	Args:
		freq: A value in Hertz. This can also be a numpy array.

	Returns
		A value in Mels.
	"""
	return 2595 * np.log10(1 + freq / 700.0)

def mel2freq(mel):
	"""Convert a value in Mels to Hertz

	Args:
		mel: A value in Mels. This can also be a numpy array.

	Returns
		A value in Hertz.
	"""
	return 700 * (10 ** (mel / 2595.0) - 1)

# The function computes the Mel filterbank coefficients of the power spectrum using the formula
# for converting frequencies from Hertz to Mel scale and the triangular filterbank approach.
def log_mel_spec(fs_hz, num_filters, NFFT, pow_frames):
	'''
	This function takes as input some parameters related to the audio signal,
	such as the sampling frequency fs_hz, the number of filters num_filters, the FFT size NFFT,
	and a matrix pow_frames representing the power spectrum of the audio signal.
	'''
	low_freq_mel = 0
	high_freq_mel = (2595 * np.log10(1 + (fs_hz / 2) / 700))  # Convert Hz to Mel
	mel_points = np.linspace(low_freq_mel, high_freq_mel, num_filters + 2)  # Equally spaced in Mel scale
	hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
	bin = np.floor((NFFT + 1) * hz_points / fs_hz)
	fbank = np.zeros ((num_filters, int (np.floor(NFFT / 2 + 1))))
	for m in range (1, num_filters + 1):
		f_m_minus = int(bin[m - 1])  # left
		f_m = int (bin[m])  # center
		f_m_plus = int(bin[m + 1])  # right

		for k in range(f_m_minus, f_m):
			fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
		for k in range(f_m, f_m_plus):
			fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
	filter_banks = np.dot(pow_frames, fbank.T)
	filter_banks = np.where(filter_banks == 0, np.finfo (float).eps, filter_banks)  # Numerical Stability
	filter_banks = 20 * np.log10(filter_banks)  # dB
	# The Mel filterbank coefficients are returned in the variable filter_banks as a matrix,
	# and the array bin contains the indices of the corresponding FFT bins.
	return filter_banks,bin

def save_mfcc(mfccs, num_ceps, filename):
	# Save MFCC to text file
	text_file_path = f"C:/Users/dell/Desktop/assignments/ST/{filename}.txt"
	with open (text_file_path, "w") as f:
		for i in range (mfccs.shape[0]):
			mfcc_row = mfccs[i]
			mfcc_str = " ".join (str (x) for x in mfcc_row)
			f.write (mfcc_str + "\n")

def cmvn(mfcc_feat):
    """
    Apply cepstral mean and variance normalization (CMVN) to a given speech signal using MFCC.

    Args:
        signal (numpy.ndarray): Input speech signal.
        win_size (int): Window size in frames for mean and variance computation.

    Returns:
        numpy.ndarray: Normalized feature matrix.
    """

    # Compute mean and variance across frames
    mean = np.mean(mfcc_feat, axis=1)
    variance = np.var(mfcc_feat, axis=1)

    # Subtract mean from each frame and divide by standard deviation
    mfcc_feat = mfcc_feat - mean[:, np.newaxis]
    std = np.sqrt(variance)  # <-- make sure var is defined
    mfcc_feat = mfcc_feat / std[:, np.newaxis]

    return mfcc_feat

#Function to calculate mfcc
def compute_mfcc(fs_hz, signal, signal_length, ref, configs, cmnor = True):
	'''
	Args:
		fs_hz: the sampling freauency
		signal: adudio signal as a 1-dim array
		signal_length: length of signal
		ref: reference mfcc
		cmnor: whether to use Cepstral mean and variance normalisation
	'''
	# Define parameters
	preemph = 0.97  # pre-emphasis coeefficient
	frame_length_ms = 25  # frame length in ms
	frame_step_ms = 10  # frame shift in ms
	low_freq_hz = 0  # filterbank low frequency in Hz
	high_freq_hz = 8000  # filterbank high frequency in Hz
	nyquist = fs_hz / 2.0  # Check the Nyquist frequency
	if high_freq_hz > nyquist:
		high_freq_hz = nyquist
	num_filters = 41  # number of mel-filters
	num_ceps = 40  # number of cepstral coefficients (excluding C0)
	cep_lifter = 22  # Cepstral liftering order
	eps = 0.001  # Floor to avoid log(0)

	# Pre-emphasis question1
	emphasised = np.append (signal[0], signal[1:] - preemph * signal[:-1])

	# Compute number of frames and padding
	frame_length = int (round (frame_length_ms / 1000.0 * fs_hz));
	frame_step = int (round (frame_step_ms / 1000.0 * fs_hz));
	# return the ceil of the number of frames
	num_frames = int (np.ceil (float (signal_length - frame_length) / frame_step))
	print ("number of frames is {}".format (num_frames))
	pad_signal_length = num_frames * frame_step + frame_length
	pad_zeros = np.zeros ((pad_signal_length - signal_length))
	pad_signal = np.append (emphasised, pad_zeros)

	# Find the smallest power of 2 greater than frame_length
	NFFT = 1 << (frame_length - 1).bit_length ();  # bit shift, produces the binary number 0b1000000000

	# Compute mel-filters(uncompleted question)
	# NFFT // 2 + 1 computes the number of frequency bins that will be used to create the Mel filters.
	mel_filters = np.zeros ((NFFT // 2 + 1, num_filters))
	low_freq_mel = 0
	high_freq_mel = freq2mel (high_freq_hz)
	mel_points = np.linspace (low_freq_mel, high_freq_mel, num_filters + 2)
	# convert Mel-scale frequencies back to Hz
	hz_points = mel2freq (mel_points)

	# create the filterbank
	fbank = np.zeros ((num_filters, int (np.floor (frame_length / 2 + 1))))
	for m in range (1, num_filters + 1):
		f_m_minus = int (np.floor (hz_points[m - 1] / fs_hz * (frame_length + 1)))
		f_m = int (np.floor (hz_points[m] / fs_hz * (frame_length + 1)))
		f_m_plus = int (np.floor (hz_points[m + 1] / fs_hz * (frame_length + 1)))

		for k in range (f_m_minus, f_m):
			fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
		for k in range (f_m, f_m_plus):
			fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
	mel_filters = fbank
	print (mel_filters)

	# Compute MFCCs
	# Here you can choose either the off-line mode, i.e. save all the frames in a
	# matrix and process them in one go, or the online mode, i.e. compute MFCCs
	# frame by frame.

	# Hamming window
	win = np.hamming (frame_length)

	# Compute lifter
	lift = 1 + (cep_lifter / 2.0) * np.sin (np.pi * np.arange (num_ceps) / cep_lifter)

	# Pre-allocation
	feat_powspec = np.zeros ((num_frames, NFFT // 2 + 1))
	feat_fbank = np.zeros ((num_frames, num_filters))
	feat_mfcc = np.zeros ((num_frames, num_ceps))

	# Here we compute MFCCs frame by frame
	for t in range (0, num_frames):
		# Framing(question)

		indices = np.tile (np.arange (0, frame_length), (num_frames, 1)) + np.tile (
			np.arange (0, num_frames * frame_step, frame_step), (frame_length, 1)).T
		frame = pad_signal[indices.astype (np.int32, copy=False)]
		# Apply the Hamming window
		frame = frame * win

		# Compute magnitude spectrum (question)
		magspec = np.absolute (np.fft.rfft (frame, NFFT))  # Magnitude of the FFT

		# Compute power spectrum(question)
		powspec = ((1.0 / NFFT) * ((magspec) ** 2))

		# Save power spectrum features
		feat_powspec[:, :] = powspec;

		# Compute log mel spectrum(question)
		fbank, freq_bins = log_mel_spec (fs_hz, num_filters, NFFT, powspec)

		# Save fbank features
		feat_fbank[:, :] = fbank

		# Apply DCT to get num_ceps MFCCs, omit C0(question)
		mfcc = dct (fbank, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13

		# Liftering
		mfcc *= lift

		# Save mfcc features
		feat_mfcc[:, :] = mfcc

	# Log-compress power spectrogram
	feat_powspec[feat_powspec < eps] = eps
	feat_powspec = np.log (feat_powspec)

	if cmnor == False:
		print ("=== Before Cepstral mean and variance normalisation")
		print ("mfcc mean = {}".format (np.mean(feat_mfcc, axis=0)))
		print ("mfcc std = {}".format (np.std(feat_mfcc, axis=0)))
		# Save MFCC to text file
		filename = "feat_mfcc"
		save_mfcc(feat_mfcc, num_ceps, filename)
	else:
		# Cepstral mean and variance normalisation
		feat_mfcc_z = cmvn(mfcc)
		print ("=== After Cepstral mean and variance normalisation")
		print ("mfcc mean = {}".format(np.mean (feat_mfcc_z, axis=0)))
		print ("mfcc std = {}".format(np.std (feat_mfcc_z, axis=0)))
		# Save MFCC to text file
		filename = "feat_mfcc_z"
		save_mfcc(feat_mfcc_z, num_ceps, filename)

	# Plotting power spectrogram vs mel-spectrogram
	plt.figure (1)
	siglen = len (signal) / float (fs_hz);
	plt.subplot (211)
	plt.imshow (feat_powspec.T, origin='lower', aspect='auto', extent=(0, siglen, 0, fs_hz / 2000), cmap='gray_r')
	plt.title ('Power Spectrogram')
	plt.ylabel ('Frequency (kHz)')
	plt.subplots_adjust (hspace=0.8)
	# Set the tick positions
	tick_positions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
	plt.gca ().set_yticks (tick_positions)
	# Set the tick labels
	tick_labels = ['', 1, 2, 3, 4, 5, 6, 7, 8]
	plt.gca ().set_yticklabels (tick_labels)
	plt.subplot (212)
	freq_bins = freq_bins.astype (int)
	plt.imshow (feat_fbank.T, origin='lower', aspect='auto', extent=(0, siglen, 0, num_filters), cmap='gray_r')
	plt.yticks ([0, 5, 10, 15, 20, 26])
	plt.gca ().set_yticklabels (['', freq_bins[5], freq_bins[10], freq_bins[16], freq_bins[21], freq_bins[27]])
	plt.title ('Mel-filter Spectrogram')
	plt.xlabel ('Time (s)')
	plt.ylabel ('Frequency (Hz)')
	plt.savefig(f'C:/Users/dell/Desktop/assignments/ST/configs/{configs}_melspec_and_power_spectrogram.png')
	# choose whether to use Cepstral mean and variance normalisation
	if cmnor == True:
		# Plotting MFCCs with CMN
		plt.figure (2)
		plt.subplot (211)
		plt.imshow (feat_mfcc.T, origin='lower', aspect='auto', extent=(0, siglen, 1, num_ceps), cmap='jet')
		plt.title ('MFCC without mean and variance normalisation')
		plt.subplots_adjust (hspace=0.8)
		plt.colorbar ()

		plt.subplot (212)
		plt.imshow (feat_mfcc_z.T, origin='lower', aspect='auto', extent=(0, siglen, 1, num_ceps), cmap='jet')
		plt.title ('MFCC with mean and variance normalisation')
		plt.ylabel ('MFCC coeffcients')
		plt.xlabel ('Time')
		plt.colorbar ()
		plt.savefig (f'C:/Users/dell/Desktop/assignments/ST/configs/{configs}_feat_mfcc_z.png')
	else:
		plt.figure (2)
		plt.subplot (211)
		plt.imshow (feat_mfcc.T, origin='lower', aspect='auto', extent=(0, siglen, 1, num_ceps), cmap='jet')
		plt.title ('MFCC without mean and variance normalisation')
		plt.ylabel ('MFCC coeffcients')
		plt.xlabel ('Time')
		plt.colorbar ()
		plt.savefig (f'C:/Users/dell/Desktop/assignments/ST/feat_mfcc.png')


# Main Program
ref = numpy.loadtxt('SA1.mfcc')
# Read waveform
fs_hz, signal = wav.read('SA1.wav')
signal_length = len(signal)
configs = '40MFCCs'
compute_mfcc(fs_hz, signal, signal_length, ref, configs, cmnor=True)

