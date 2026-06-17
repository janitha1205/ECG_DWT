import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from scipy.signal import savgol_filter

def mother_mexican_hat(t):
    return (1 - t**2) * np.exp(-t**2 / 2)

def coefficients_from_scratch(signal, scales, span=10.0):

    signal = np.array(signal, dtype=float)
    n_samples = len(signal)

    coeffs = np.zeros((len(scales), n_samples))
    
    for i, scale in enumerate(scales):

        n_points = int(span * scale) * 2 + 1
        t = np.linspace(-span * scale, span * scale, n_points)
        scaled = mother_mexican_hat(t / scale)
        scaled = scaled / np.sqrt(scale)
        coeffs[i, :] = np.convolve(signal, scaled[::-1], mode='same')
        
    return coeffs




def inverse_scratch(coeffs, scales, dx=1.0):

    n_scales, n_samples = coeffs.shape
    reconstructed = np.zeros(n_samples)
    

    for idx, scale in enumerate(scales):

        reconstructed += coeffs[idx, :] / (scale ** 1.5)

    factor = 1.45 
    reconstructed *= (dx / factor)
    
    return reconstructed


sampling_rate = 1000
dt=1/sampling_rate
duration = 5
ecg_signal = nk.ecg_simulate(
    duration=duration, sampling_rate=sampling_rate, heart_rate=70
)
derivative = np.gradient(ecg_signal, dt)
derivative_2=np.gradient(derivative, dt)
y=np.zeros([sampling_rate*duration-1])
x=np.zeros([sampling_rate*duration-1])
y1=np.zeros([sampling_rate*duration-1])
x1=np.zeros([sampling_rate*duration-1])
y2=np.zeros([sampling_rate*duration-1])
x2=np.zeros([sampling_rate*duration-1])
t=np.zeros([len(ecg_signal)])

for k in range(sampling_rate*duration):
    t[k]=k*dt
for k in range(sampling_rate*duration-1):

    x[k]=ecg_signal[k]
    y[k]=ecg_signal[k+1]
   
for k in range(sampling_rate*duration-2):
    x2[k]=derivative[k]
    y2[k]=derivative[k+1]
clean_derivative = savgol_filter(ecg_signal, window_length=15, polyorder=3, deriv=1, delta=dt)
dim=100

scales = np.arange(1, dim/20, dtype=float)
coefficients=coefficients_from_scratch(ecg_signal, scales, span=10.0)



reconstructed_signal = inverse_scratch(coefficients, scales)
fig, axes = plt.subplots(3, 1, figsize=(10, 10))


axes[0].plot(t, ecg_signal, color='black', lw=2, label='ECG before')
axes[0].set_title("raw ECG")
axes[0].legend()

im = axes[1].imshow(coefficients, aspect='auto', cmap='jet', 
                     extent=[t[0], t[-1], scales[-1], scales[0]])
axes[1].set_title(" spectrum (Scalogram View)")
axes[1].set_ylabel("Scales (Width)")
fig.colorbar(im, ax=axes[1], label='Intensity')


axes[2].plot(t, reconstructed_signal, color='blue', lw=2, label='denoize or approximated ECG')
axes[2].set_title("approximation")
axes[2].legend()

for ax in [axes[0], axes[2]]:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
