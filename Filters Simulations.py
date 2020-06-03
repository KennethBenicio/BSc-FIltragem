#LMS Simulation

lenght, number_of_users, order = 10000, 1, 1
step = 0.05
runs = 100

MSE = np.zeros(lenght - order + 1)
for i in tqdm(range(runs)):
    signal = np.random.randint(2,size = [lenght,number_of_users]) + 1j*np.random.randint(2,size =[lenght,number_of_users])
    signal.real[signal.real == 0] = -1
    signal.imag[signal.imag == 0] = -1
    signal_corrupted = np.random.randn(1)*signal
    signal_corrupted = signal_corrupted[:,0]
    mse = lms(signal_corrupted, signal, step, order)

    MSE = MSE + mse
    
MSE = MSE/runs

plt.figure()
plt.plot(10*np.log10(np.abs(MSE)), label='step = 0.05')
plt.title('LMS Filter')
plt.ylabel('MSE(dB)')
plt.xlabel('Iterations')
plt.legend()
plt.savefig('LMS Filter - MSE x Iteration')
plt.show()

#NLMS Simulation

lenght, number_of_users, order = 10000, 1, 1
step = 0.05
runs = 100

MSE = np.zeros(lenght - order + 1)
for i in tqdm(range(runs)):
    signal = np.random.randint(2,size = [lenght,number_of_users]) + 1j*np.random.randint(2,size =[lenght,number_of_users])
    signal.real[signal.real == 0] = -1
    signal.imag[signal.imag == 0] = -1
    signal_corrupted = np.random.randn(1)*signal
    signal_corrupted = signal_corrupted[:,0]
    mse = nlms(signal_corrupted, signal, step, order)

    MSE = MSE + mse
    
MSE = MSE/runs

plt.figure()
plt.plot(10*np.log10(np.abs(MSE)), label='step = 0.05')
plt.title('LMS Filter')
plt.ylabel('MSE(dB)')
plt.xlabel('Iterations')
plt.legend()
plt.savefig('NLMS Filter - MSE x Iteration')
plt.show()

#TLMS Simulation
lenght, number_of_antenas, number_of_paths, buffer_size = 10000, 100, 4, 16

snr_dB = 30
step = 1.00
runs = 100

MSE = np.zeros(lenght)
for i in tqdm(range(runs)):
    signal = np.random.randint(2,size = [lenght,number_of_users]) + 1j*np.random.randint(2,size = [lenght,number_of_users])
    signal.real[signal.real == 0] = -1
    signal.imag[signal.imag == 0] = -1
    [channel_matrix, _] = channel(number_of_antenas, number_of_users, number_of_paths, buffer_size)
    [matrix_of_sampling, _] = sampled_matrix(signal, number_of_users, number_of_antenas, buffer_size, lenght)
    [_, signal_corrupted, _] = received_signal(matrix_of_sampling, channel_matrix, snr_dB)
    [w, mse] = tlms(signal_corrupted, signal, step)
    
    MSE = MSE + mse
    
MSE = MSE/runs

plt.figure()
plt.plot(10*np.log10(np.abs(MSE)), label='step = 1.00')
plt.title('TLMS Filter for N = 100, users = 1, paths = 4, buffer = 16 at snr = 30')
plt.ylabel('MSE(dB)')
plt.xlabel('Iterations')
plt.legend()
plt.savefig('TLMS Filter - MSE x Iteration')
plt.show()

#TLMS Simulation
lenght, number_of_antenas, number_of_paths, buffer_size = 10000, 100, 4, 16

snr_dB = 30
step = 1.00
runs = 100

MSE = np.zeros(lenght)
for i in tqdm(range(runs)):
    signal = np.random.randint(2,size = [lenght,number_of_users]) + 1j*np.random.randint(2,size = [lenght,number_of_users])
    signal.real[signal.real == 0] = -1
    signal.imag[signal.imag == 0] = -1
    [channel_matrix, _] = channel(number_of_antenas, number_of_users, number_of_paths, buffer_size)
    [matrix_of_sampling, _] = sampled_matrix(signal, number_of_users, number_of_antenas, buffer_size, lenght)
    [_, signal_corrupted, _] = received_signal(matrix_of_sampling, channel_matrix, snr_dB)
    [w, mse] = atlms(signal_corrupted, signal, step)
    
    MSE = MSE + mse
    
MSE = MSE/runs

plt.figure()
plt.plot(10*np.log10(np.abs(MSE)), label='step = 1.00')
plt.title('TLMS Filter for N = 100, users = 1, paths = 4, buffer = 16 at snr = 30')
plt.ylabel('MSE(dB)')
plt.xlabel('Iterations')
plt.legend()
plt.savefig('ATLMS Filter - MSE x Iteration')
plt.show()