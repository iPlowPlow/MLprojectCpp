#include <stdlib.h>
#include <iostream>
#include <time.h>

extern "C"
{
	double randDouble(double low, double high)
	{
		double temp;

		/* swap low & high around if the user makes no sense */
		if (low > high)
		{
			temp = low;
			low = high;
			high = temp;
		}

		/* calculate the random number & return it */
		temp = (rand() / (static_cast<double>(RAND_MAX) + 1.0))
			* (high - low) + low;
		return temp;
	}

	int calculateSign(double* W, double x, double y) {
		double sign = W[1] * x + W[2] * y + W[0];
		if (sign < 0) return -1;
		return 1;
	}

	_declspec(dllexport) double* linear_create() {
		srand(time(NULL));
		double *W = (double*) malloc(sizeof(double) * 3);
		W[0] = randDouble(0.0, 2.0) -1;
		W[1] = randDouble(0.0, 2.0) - 1;
		W[2] = randDouble(0.0, 2.0) - 1;

		return W;
	}

	_declspec(dllexport) void linear_delete(double* W) {
		free(W);
	}

	//elem = nombre elem ; elem size = taille elem
	_declspec(dllexport) void linear_train_classification(double* W, int elem, int elemsize, double* tabSphere) {
		double alpha = 0.1;
		int iter = 10000;
		for (int j = 0; j < iter; j++) {
			for (int i = 0; i < elem*elemsize; i += elemsize) {				
				double x = tabSphere[i];
				double y = tabSphere[i + 1];
				double sign = calculateSign(W, x, y);
					
				W[0] = W[0] + alpha * (tabSphere[i + 2] - sign);
				W[1] = W[1] + alpha * (tabSphere[i + 2] - sign)*x;
				W[2] = W[2] + alpha * (tabSphere[i + 2] - sign)*y;

			}
		}	
	}

	_declspec(dllexport) double linear_classify(double* W, double x, double y) {
		return calculateSign(W, x, y);
	}
	
}