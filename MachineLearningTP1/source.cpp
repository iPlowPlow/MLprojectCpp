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
	_declspec(dllexport) void linear_train_classification(double* W, int elem, int elemsize, double* tab) {
		double alpha = 0.1;
		int iter = 100;
		for (int i = 0; i < elem*elemsize; i+=2) {
			int x = tab[i];
			int z = tab[i + 1];

		}
	}


}