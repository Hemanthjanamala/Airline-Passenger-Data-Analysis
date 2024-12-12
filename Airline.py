import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("airline8.csv")

print("First 5 rows of the dataset:")
print(df.head())

print("\nColumns in the dataset:")
print(df.columns)

print("\nMissing values in each column:")
print(df.isnull().sum())

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

print("\nRows with invalid dates (if any):")
print(df[df['Date'].isnull()])

print("\nData types of columns:")
print(df.dtypes)

df.dropna(subset=['Number', 'Price'], inplace=True)

print("\nCleaned dataframe (after dropping missing values):")
print(df.head())

print("\nFinal missing values check (should be 0):")
print(df.isnull().sum())

passenger_numbers = df['Number'].values

fourier_transform = np.fft.fft(passenger_numbers)

frequencies = np.fft.fftfreq(len(passenger_numbers))

plt.figure(figsize=(10, 6))
plt.plot(frequencies, np.abs(fourier_transform))

plt.xlabel('Frequency (1/days)', fontsize=12)
plt.ylabel('Magnitude', fontsize=12)
plt.title('Fourier Transform of Passenger Numbers', fontsize=14)

plt.grid(True)
plt.show()
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
monthly_passengers = df.groupby('Month')['Number'].mean()

passenger_numbers = monthly_passengers.values
n = len(passenger_numbers)
fourier_transform = np.fft.fft(passenger_numbers)
frequencies = np.fft.fftfreq(n)

terms_to_use = 8
approximation = np.zeros(n, dtype=np.float64)

for i in range(terms_to_use):
    approximation += (
        np.real(fourier_transform[i]) * np.cos(2 * np.pi * frequencies[i] * np.arange(n)) -
        np.imag(fourier_transform[i]) * np.sin(2 * np.pi * frequencies[i] * np.arange(n))
    ) / n

plt.figure(figsize=(12, 6))
plt.bar(monthly_passengers.index, monthly_passengers.values, color='lightgray', alpha=0.7, label='Average Monthly Passengers')
plt.plot(monthly_passengers.index, approximation, color='red', label='Fourier Series Approximation (8 terms)', linewidth=2)
plt.xticks(monthly_passengers.index, ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], rotation=45)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Number of Passengers', fontsize=12)
plt.title('Average Daily Passengers and Fourier Series Approximation', fontsize=14)
plt.legend(fontsize=12)
plt.text(12, max(monthly_passengers) * 0.9, 'Student ID: 22070987', fontsize=12, color='purple', ha='right')
plt.tight_layout()
plt.show()

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
passenger_numbers = df['Number'].values
fourier_transform = np.fft.fft(passenger_numbers)
frequencies = np.fft.fftfreq(len(passenger_numbers))
power_spectrum = np.abs(fourier_transform) ** 2

nonzero_indices = frequencies != 0
frequencies_nonzero = frequencies[nonzero_indices]
power_spectrum_nonzero = power_spectrum[nonzero_indices]

autumn_months = [9, 10, 11]
autumn_df = df[df['Month'].isin(autumn_months)]

autumn_revenue = (autumn_df['Price'] * autumn_df['Number']).sum()
total_revenue = (df['Price'] * df['Number']).sum()
X = (autumn_revenue / total_revenue) * 100

autumn_passengers = autumn_df['Number'].sum()
total_passengers = df['Number'].sum()
Y = (autumn_passengers / total_passengers) * 100

df['Revenue'] = df['Price'] * df['Number']
average_price = df['Price'].mean()

sorted_indices = np.argsort(power_spectrum_nonzero)[::-1]
peak_indices = sorted_indices[:2]
peak_periods = 1 / np.abs(frequencies_nonzero[peak_indices])  # Convert frequency to period

plt.figure(figsize=(12, 6))

plt.plot(1 / np.abs(frequencies_nonzero), power_spectrum_nonzero, color='blue', label='Power Spectrum')

for idx, period in zip(peak_indices, peak_periods):
    plt.scatter(1 / np.abs(frequencies_nonzero[idx]), power_spectrum_nonzero[idx], color='red', label='Peaks' if idx == peak_indices[0] else "")
    plt.annotate(f"X={period:.2f} days",
                 (1 / np.abs(frequencies_nonzero[idx]), power_spectrum_nonzero[idx]),
                 textcoords="offset points", xytext=(10, 10), ha='center', fontsize=10, color='green')

plt.title("Figure 2. Fourier Power Spectrum for Two Highest Peaks", fontsize=14)
plt.xlabel("Period (Days)", fontsize=12)
plt.ylabel("Power", fontsize=12)
plt.legend(fontsize=12)

plt.text(1.05 * max(1 / np.abs(frequencies_nonzero)), max(power_spectrum_nonzero) * 0.95,
         f"Student ID: 22070987\nPeriods of highest peaks: {peak_periods[0]:.2f}, {peak_periods[1]:.2f} days",
         fontsize=12, color='purple', ha='right', va='top')

plt.text(max(1 / np.abs(frequencies_nonzero)) * 0.5, max(power_spectrum_nonzero) * 0.5,
         f"X (Autumn Revenue %): {X:.2f}%\nY (Autumn Passengers %): {Y:.2f}%",
         fontsize=12, color='green', ha='center', va='top')

plt.tight_layout()
plt.show()


