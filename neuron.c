#include "head.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

extern W w;
extern B b;
extern V v;
extern W wg;
extern B bg;
extern W ew;
extern B eb;
extern V ev;

//Loss function
//MSE & MSE gradient
//MSE(y, y_pred) = (1/n) * ∑(y - y_pred)^2
//e: expected; a: actual; 
double MSE(double e[], double a[], int n)
{
	double l = 0.0;
	for (int i = 0; i < LEN_O; i++)
		l += pow(e[i] - a[i], 2);
	l /= n;
	return l;
}
double MSE_grad(double e, double a)
{
	return -2.0*(e - a);
}

//Activation function
//None
double None(double x)
{
	return x;
}
double grad(double x)
{
	return 1.0;
}
//ReLU & RelU gradient
//ReLU(x) = max(x, 0)
double ReLU(double x)
{
	if (x >= 0)
		return x;
	else
		return 0.0;
}
double ReLU_grad(double x)
{
	if (x >= 0)
		return 1.0;
	else
		return 0.0;
}
//LeakyReLU & LeakyReLU gradient
//LeakyReLU(x) = max(x, alpha * x)
double alpha = 0.01;
double LReLU(double x)
{
	if (x >= alpha * x)
		return x;
	else
		return alpha * x;
}
double LReLU_grad(double x)
{
	if (x >= alpha * x)
		return 1;
	else
		return alpha;
}
//Sigmoid & Sigmoid gradient
double Sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}
double Sigmoid_grad(double x)
{
	return Sigmoid(x)*(1 - Sigmoid(x));
}
//Tanh & Tanh gradient
double Tanh(double x)
{
	return tanh(x);
}
double Tanh_grad(double x)
{
	return 1 - pow(Tanh(x), 2);
}

//Gradients

//seed random using sec * nsec
int strand(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	int seed = ts.tv_nsec * time(NULL);
	srand(seed);
	return seed;
}

//Normal distro random
double rand_uniform()
{
	return (double) rand() / RAND_MAX;
}
double box_muller()
{
	static double r[2];
	double u1, u2, r1, r2;
	do {
		u1 = rand_uniform();
		u2 = rand_uniform();
		r1 = 2.0 * u1 - 1.0;
		r2 = 2.0 * u2 - 1.0;
	} while (r1 * r1 + r2 * r2 > 1.0);
	double f = sqrt(-2.0 * log(u1));
	r[0] = r1 * f;
	r[1] = r2 * f;
	return r[0];
}
double rand_nml(double mean, double stddev)
{
	double r;
	r = box_muller();
	return r * stddev + mean;
}
