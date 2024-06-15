#include "head.h"
#include <math.h>
#include <stdlib.h>
#include <time.h>

//get weights & biases
extern W w;
extern B b;
extern V v;

//Loss function
//MSE & MSE gradient
//MSE(y, y_pred) = (1/n) * ∑(y - y_pred)^2
//e: expected; a: actual; 
double MSE(double e, double a)
{
	double l;
	l = pow(e - a, 2);
	return l;
}
double MSE_grad_w(double e, double a, double x)
{
	double grad_w;
	grad_w = 2.0*x*(nf.w*x + nf.b - e);
	return grad_w;
}
double MSE_grad_b(double e, double a, double x)
{
	double grad_b;
	grad_b = 2.0*(nf.w*x + nf.b - e);
	return grad_b;
}

//Activation function
//None
double None(double x)
{
	return x;
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
double Sigmoid(double x)
{
	return 1 / (1 + exp(-x));
}
double Tanh(double x)
{
	return tanh(x);
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

//Gradients
double MSE_grad_w_ReLU(double e, double a, double x)
{
	if (f(x) > 0)
		return 2.0*x*(a - e);
	else
		return 0.0;
}
double MSE_grad_b_ReLU(double e, double a, double x)
{
	if (f(x > 0))
		return 2.0*(a - e);
	else
		return 0.0;
}
double MSE_grad_w_LReLU(double e, double a, double x)
{
	if (f(x) > 0)
		return 2.0*x*(a - e);
	else
		return 2.0*alpha*x*(a - e);
}
double MSE_grad_b_LReLU(double e, double a, double x)
{
	if (f(x > 0))
		return 2.0*(a - e);
	else
		return 2.0*alpha*(a - e);
}
double MSE_grad_w_Sigmoid(double e, double a, double x)
{
	return (a - e)*e*(1 - e)*x;
}
double MSE_grad_b_Sigmoid(double e, double a, double x)
{
	return (a - e)*e*(1 - e);
}
double MSE_grad_w_Tanh(double e, double a, double x)
{
	return (a - e)*(1 - pow(e, 2))*x;
}
double MSE_grad_b_Tanh(double e, double a, double x)
{
	return (a - e)*(1 - pow(e, 2));
}

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
