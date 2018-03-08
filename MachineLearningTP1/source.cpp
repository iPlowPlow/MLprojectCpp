#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <Eigen/Dense>
using Eigen::MatrixXd;

extern "C"
{
	void linear_predict(double* W, MatrixXd mT, MatrixXd mI, MatrixXd y) {

		MatrixXd mResult = (mI*mT)*y;
		W[0] = mResult(0);
		W[1] = mResult(1);
		W[2] = mResult(2);

	}

	_declspec(dllexport) void linear_train_regression(double* W, int elem, int elemsize, double* tabSphere) {
		MatrixXd m(elem, 3);
		MatrixXd y(elem, 1);
		int i = 0;

		int index = 0;
		for (i = 0; i < elem; i++) {
			y(i, 0) = tabSphere[3 * i + 2];
			m(i, 0) = 1;
			m(i, 1) = tabSphere[3 * i];
			m(i, 2) = tabSphere[3 * i + 1];
		}

	

		MatrixXd mT = m.transpose();
		MatrixXd mTmp = (mT * m);
		MatrixXd mI = mTmp.inverse();

		linear_predict(W, mT, mI, y);
	}

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

	double calculateSign(double* W, double x, double z) {
		double sign = W[1] * x + W[2] * z + W[0];
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
	_declspec(dllexport) void linear_train_classification(double* W, int elem, int elemsize, double* tabSphere, double alpha, int iter) {
		//double alpha = 0.1;
		//int iter = 10000;
		for (int j = 0; j < iter; j++) {
			for (int i = 0; i < elem*elemsize; i += elemsize) {				
				double x = tabSphere[i];
				double z = tabSphere[i + 1];
				double sign = calculateSign(W, x, z);
					
				W[0] = W[0] + alpha * (tabSphere[i + 2] - sign);
				W[1] = W[1] + alpha * (tabSphere[i + 2] - sign)*x;
				W[2] = W[2] + alpha * (tabSphere[i + 2] - sign)*z;

			}
		}	
	}

	_declspec(dllexport) double linear_classify(double* W, double x, double z) {
		return calculateSign(W, x, z);
	}
	
}