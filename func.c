#include "head.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

//get weights & biases
extern W w;
extern B b;
extern V v;
//get extern options
extern int a_func_num;

//init w & b & v
//w, b = rand_nml(0.0, 1.0); v = 0.0;
int init_wbv(void)
{
	for (int i = 0; i < LEN_I; i++) {
		for (int j = 0; j < LEN_H0; j++)
			w.ih0[i][j] = rand_nml(0.0, 1.0);
		v.i[i] = 0.0;
	}
	for (int i = 0; i < LEN_H0; i++) {
		for (int j = 0; j < LEN_O; j++)
			w.h0o[i][j] = rand_nml(0.0, 1.0);
		b.h0[i] = rand_nml(0.0, 1.0);
		v.h0[i] = 0.0;
	}
	for (int i = 0; i < LEN_O; i++) {
		b.o[i] = rand_nml(0.0, 1.0);
		v.o[i] = 0.0;
	}
	return 0;
}

//forward propagation
int fdp(void)
{
	double tmp_wsum = 0.0;
	//i -> h
	for (int i = 0; i < LEN_H0; i++) {
		for (int j = 0; j < LEN_I; j++) {
			tmp_wsum += w.ih0[j][i] * v.i[j];
		}
	}
	//h -> o
	for (int i = 0; i < LEN_O; i++) {
		for (int j = 0; j < LEN_H0; j++) {
		}
	}
	return 0;
}

//calculate batch thread function
void* calc_batch(void* args)
{
	//get args
	void** arg = (void**) args;
	double* pl = arg[0];
	double* pgrad_w = arg[1];
	double* pgrad_b = arg[2];
	//function pointers
	double (*a_func)(double);
	double (*l_func)(double, double);
	double (*grad_w_func)(double, double, double);
	double (*grad_b_func)(double, double, double);
	//get functions
		/*
		functionnum:
			0: None
			1: ReLU
			2: LeakyReLU
			3: Sigmoid
			4: Tanh
		*/
	if (a_func_num == 0) {
		a_func = None;
		grad_w_func = MSE_grad_w;
		grad_b_func = MSE_grad_b;
	} else if (a_func_num == 1) {
		a_func = ReLU;
		grad_w_func = MSE_grad_w_ReLU;
		grad_b_func = MSE_grad_b_ReLU;
	} else if (a_func_num == 2) {
		a_func = LReLU;
		grad_w_func = MSE_grad_w_LReLU;
		grad_b_func = MSE_grad_b_LReLU;
	} else if (a_func_num == 3) {
		a_func = Sigmoid;
		grad_w_func = MSE_grad_w_Sigmoid;
		grad_b_func = MSE_grad_b_Sigmoid;
	} else if (a_func_num == 4) {
		a_func = Tanh;
		grad_w_func = MSE_grad_w_Tanh;
		grad_b_func = MSE_grad_b_Tanh;
	} else {
		return (void*)&a_func_num;
	}
	l_func = MSE;
	//calc
	double x = rand_nml(1.0, 1.0);
	double y_f = a_func(f(x));
	double y_g = a_func(g(x));
	//pass results
	*pl += l_func(y_g, y_f);
	*pgrad_w += grad_w_func(y_g, y_f, x);
	*pgrad_b += grad_b_func(y_g, y_f, x);
	return NULL;
}

//other function stuff
//test file for writing
int isusablefile(char* filename)
{
	FILE* filep = fopen(filename, "r");
	if (filep == NULL) {
		//file doesn't exist, probably
		return 0;
	} else {
		//file exists
		fclose(filep);
		printf("%sWARN: file already exists. \n%sFile: %s, in option '-f'. %s", COLOR_WARN, COLOR_END, filename, COLOR_END);
		bool input_valid = false;
		char usrinput = '\0';
		while (input_valid == false) {
			printf("\t\ny: yes; \t\nn: no, forget option '-f'; \t\nq: quit; \nOverwrite file? [y/n/q] > ");
			scanf("%c", &usrinput);
			if (usrinput == 'y') {
				//enable overwrite
				return 0;
				input_valid = true;
			} else if (usrinput == 'n') {
				//forget writetofile option
				return 1;
				input_valid = true;
			} else if (usrinput == 'q') {
				//quit
				return 2;
				input_valid = true;
			} else {
				//invalid option
				printf("Invalid option: \"%c\"", usrinput);
				input_valid = false;
			}
		}
	}
	return -1;
}
//print version
int printversion(void)
{
	printf("Name: LR\nVersion: %s\n", VER);
	return 0;
}
//show help (manual) page
int manualpage(void)
{
	char const manual_path[] = "./manual.txt";
	char const less_cmd[] = "less -R ";
	char cmd[sizeof(less_cmd) + sizeof(manual_path)] = "";
	sprintf(cmd, "%s%s", less_cmd, manual_path);
	int less_err = system(cmd);
	if (less_err != 0) {
		// less error
		FILE *file;
		char ch;
		file = fopen("example.txt", "r");
		if (file == NULL) {
			printf("%sError opening ./manual.txt. %sCheck if manual.txt exists. \n%s", COLOR_ERROR, COLOR_END, COLOR_END);
			return -1;
		}
		while ((ch = fgetc(file)) != EOF) {
			printf("%c", ch);
		}
		fclose(file);
	}
	return 0;
}
