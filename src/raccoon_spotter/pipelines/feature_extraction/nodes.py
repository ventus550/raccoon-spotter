import numpy as np

def train_test_split(raccoon_features_data_array: np.ndarray, test_size: float):
	X, Y = raccoon_features_data_array.values()
	samples = len(X)
	test_samples = int(test_size * samples)
	I = np.arange(samples)
	np.random.shuffle(I)
	Itest = I[:test_samples]
	Itrain = I[test_samples:]
	return (
		dict(x = X[Itrain], y = Y[Itrain]),
		dict(x = X[Itest], y = Y[Itest])
	)
	
