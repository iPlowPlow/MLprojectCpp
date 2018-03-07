#include <stdlib.h>

extern "C"
{
	_declspec(dllexport) int add_to_42(int value_to_add)
	{
		return value_to_add + 42;
	}

	_declspec(dllexport) double* linear_create() {
		double W[3];
		//W = new double[2 + 1];
		W[0] = rand() % 1 - 1;
		W[1] = rand() % 1 - 1;
		W[2] = rand() % 1 - 1;
		return W;
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

	int calculateSign(double* W , double x, double y) {
		double sign = W[1] * x + W[2] * y + W[0];
		if (sign < 0) return -1;
		return 1;
	}


}