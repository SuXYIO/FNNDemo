// FNNDemo (Feedforward Neural Network Demo)
// By SuXYIO
// Created on 2024.04.15

#pragma once
#ifndef HEAD_H
#define HEAD_H
//defines
	//Version stuff
		#define VER "1.0.0"
	//Network size
		#define DEPTH 2
		#define LEN_I 2
		#define LEN_H0 3
		#define LEN_O 1
	//Max number of threads to run
		#define MAX_THREADS 1024
	//Buffer size for strings
		#define STR_BUFSIZE 128
	//Float Precision for Print: how many digits to print after float decimal point
		#define FPP 4
	//ANSI color
		#define COLOR_NORM "\033[0m"
		#define COLOR_SUCC "\033[32m"
		#define COLOR_WARN "\033[33m"
		#define COLOR_ERROR "\033[31m"
		#define COLOR_END "\033[0m"
//typedefs
	//weights & biases
		typedef struct {
			double ih0[LEN_I][LEN_H0];
			double h0o[LEN_H0][LEN_O];
		} W;
		typedef struct {
			double h0[LEN_H0];
			double o[LEN_O];
		} B;
	//values
		typedef struct {
			double i[LEN_I];
			double h0[LEN_H0];
			double o[LEN_O];
			double l[LEN_O];
			double lall;
		} V;
//functions
	//neuron.c
		double f(double x);
		double g(double x);
		double MSE(double e, double a);
		double MSE_grad(double e, double a);
		double None(double x);
		double grad(double x);
		double ReLU(double x);
		double ReLU_grad(double x);
		double LReLU(double x);
		double LReLU_grad(double x);
		double Sigmoid(double x);
		double Sigmoid_grad(double x);
		double Tanh(double x);
		double Tanh_grad(double x);
		int strand(void);
		double rand_nml(double mean, double stddev);
		double rand_nmlstd(void);
	//func.c
		void* calc_batch(void* args);
		int init_wbv(void);
		int init_vl(void);
		int tfdp(void);
		int fdp(void);
		int bdp(void);
		int avg_wbvl(void);
		int avg_grad(void);
		int gd(void);
		int getfuncs(void);
		int isusablefile(char* filename);
		int printversion(void);
		int manualpage(void);
#endif
