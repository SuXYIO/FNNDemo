#include "head.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

extern W w;
extern B b;
extern V v;
extern W wg;
extern B bg;
extern W ew;
extern B eb;
extern V ev;
//get extern options
extern int a_func_num;
extern int l_func_num;
extern double eta;
extern int batch_size;

//linear transformation (wx+b) output
//(Alse called weighted sum)
V u;

//declare getfunc functions
double (*get_a_func(int funcnum))(double);
double (*get_l_func(int funcnum))(double, double);
double (*get_agrad_func(int funcnum))(double);
double (*get_lgrad_func(int funcnum))(double, double);

//function pointers
double (*a_func)(double) = NULL;
double (*l_func)(double, double) = NULL;
double (*agrad_func)(double) = NULL;
double (*lgrad_func)(double, double) = NULL;
double (*randfunc)(void) = rand_nmlstd;

//calculate batch thread function
void* calc_batch(void* args)
{
	init_vl();
	tfdp();
	fdp();
	return NULL;
}

//init w & b & v & l
//w, b = rand_nml(0.0, 1.0); v = 0.0;
int init_wbv(void)
{
	//TODO: Init b
	//network
	for (int i = 0; i < LEN_I; i++) {
		for (int j = 0; j < LEN_H0; j++) {
			w.ih0[i][j] = randfunc();
			wg.ih0[i][j] = randfunc();
		}
		v.i[i] = 0.0;
		u.i[i] = 0.0;
	}
	for (int i = 0; i < LEN_H0; i++) {
		for (int j = 0; j < LEN_O; j++) {
			w.h0o[i][j] = randfunc();
			wg.h0o[i][j] = randfunc();
		}
		b.h0[i] = randfunc();
		v.h0[i] = 0.0;
		u.h0[i] = 0.0;
	}
	for (int i = 0; i < LEN_O; i++) {
		b.o[i] = randfunc();
		v.o[i] = 0.0;
		u.o[i] = 0.0;
	}
	//expected network
	for (int i = 0; i < LEN_I; i++) {
		for (int j = 0; j < LEN_H0; j++)
			ew.ih0[i][j] = randfunc();
		ev.i[i] = 0.0;
	}
	for (int i = 0; i < LEN_H0; i++) {
		for (int j = 0; j < LEN_O; j++)
			ew.h0o[i][j] = randfunc();
		eb.h0[i] = randfunc();
		ev.h0[i] = 0.0;
	}
	for (int i = 0; i < LEN_O; i++) {
		eb.o[i] = randfunc();
		ev.o[i] = 0.0;
	}
	return 0;
}
//init values and losses
int init_vl(void)
{
	for (int i = 0; i < LEN_I; i++)
		v.i[i] = 0.0;
	for (int i = 0; i < LEN_H0; i++)
		v.h0[i] = 0.0;
	for (int i = 0; i < LEN_O; i++) {
		v.o[i] = 0.0;
		v.l[i] = 0.0;
	}
	v.lall = 0.0;
	return 0;
}

//training network forward propagation (get traning data)
int tfdp(void)
{
	for (int i = 0; i < LEN_I; i++)
		ev.i[i] = randfunc();
	//i -> h0
	for (int i = 0; i < LEN_H0; i++) {
		for (int j = 0; j < LEN_I; j++) {
			u.h0[i] += ew.ih0[j][i] * ev.i[j] + eb.h0[i];
			ev.h0[i] += a_func(u.h0[i]);
		}
		u.h0[i] /= LEN_I;
		ev.h0[i] /= LEN_I;
	}
	//h0 -> o
	for (int i = 0; i < LEN_O; i++) {
		for (int j = 0; j < LEN_H0; j++) {
			u.o[i] += ew.h0o[j][i] * ev.h0[j] + eb.o[i];
			ev.o[i] += a_func(u.o[i]);
		}
		u.o[i] /= LEN_H0;
		ev.o[i] /= LEN_H0;
	}
	return 0;
}

//forward propagation
int fdp(void)
{
	//ev.i >> v.i
	for (int i = 0; i < LEN_O; i++)
		v.i[i] = ev.i[i];
	//i >> u.i
	for (int i = 0; i < LEN_I; i++)
		u.i[i] = v.i[i];
	//i -> h0
	for (int i = 0; i < LEN_H0; i++) {
		for (int j = 0; j < LEN_I; j++) {
			u.h0[i] += w.ih0[j][i] * v.i[j] + b.h0[i];
			v.h0[i] += a_func(u.h0[i]);
		}
		u.h0[i] /= LEN_I;
		v.h0[i] /= LEN_I;
	}
	//h0 -> o
	for (int i = 0; i < LEN_O; i++) {
		for (int j = 0; j < LEN_H0; j++) {
			u.o[i] += w.h0o[j][i] * v.h0[j] + b.o[i];
			v.o[i] += a_func(u.o[i]);
		}
		u.o[i] /= LEN_H0;
		v.o[i] /= LEN_H0;
	}
	return 0;
}
//backward propagation
/*
	For backpropagating layer x <- layer y, 
		h0<-o (2):
			dE/dw2jk = dE/dyk * dyk/du2k * du2k/dw2jk
		i<-h0 (1):
			dE/dw1ij = sigma(k=1, q) [dE/dyk * dyk/du2k * du2k/dw1ij]
			du2k/dw1ij = du2k/dzj * dzj/dw1ij
			dzj/dw1ij = dzj/du1j * du1j/dw1ij
*/
int bdp(void)
{
	//TODO: Write for b
	//loss
	for (int i = 0; i < LEN_O; i++)
		v.l[i] += l_func(ev.o[i], v.o[i]);
	//gradients
	//h0 <- o
	for (int i = 0; i < LEN_O; i++)
		for (int j = 0; j < LEN_H0; j++)
			wg.h0o[j][i] += lgrad_func(ev.o[i], v.o[i]) * agrad_func(v.o[i]) * v.h0[j];
	//i <- h0
	for (int i = 0; i < LEN_H0; i++)
		for (int j = 0; j < LEN_I; j++)
			for (int k = 0; k < LEN_O; k++)
				wg.ih0[j][i] += lgrad_func(ev.o[k], v.o[k]) * agrad_func(v.o[k]) * w.h0o[i][k] * agrad_func(v.h0[i]) * v.i[j];
	return 0;
}
//average batch
int avg_wbv(void)
{
	//network
	for (int i = 0; i < LEN_I; i++)
		for (int j = 0; j < LEN_H0; j++) {
			w.ih0[i][j] /= batch_size;
			wg.ih0[i][j] /= batch_size;
		}
	for (int i = 0; i < LEN_H0; i++) {
		for (int j = 0; j < LEN_O; j++) {
			w.h0o[i][j] /= batch_size;
			wg.h0o[i][j] /= batch_size;
		}
		b.h0[i] /= batch_size;
	}
	for (int i = 0; i < LEN_O; i++)
		b.o[i] /= batch_size;
	return 0;
}
int avg_gl(void)
{
	//gradients
	for (int i = 0; i < LEN_O; i++)
		for (int j = 0; j < LEN_H0; j++)
			wg.h0o[j][i] /= batch_size;
	for (int i = 0; i < LEN_H0; i++)
		for (int j = 0; j < LEN_I; j++)
			wg.ih0[j][i] /= batch_size;
	//loss
	for (int i = 0; i < LEN_O; i++) {
		v.l[i] /= batch_size;
		v.lall += v.l[i];
	}
	v.lall /= LEN_O;
	return 0;
}
//gradient descent
int gd(void)
{
	//TODO: Write for b
	//w
	//ih0
	for (int i = 0; i < LEN_H0; i++)
		for (int j = 0; j < LEN_I; j++)
			w.ih0[j][i] -= eta * wg.ih0[j][i];
	//h0o
	for (int i = 0; i < LEN_O; i++)
		for (int j = 0; j < LEN_H0; j++)
			w.h0o[j][i] -= eta * wg.h0o[j][i];
	return 0;
}

//get function pointers
/*
	functionnum:
		0: None
		1: ReLU
		2: LeakyReLU
		3: Sigmoid
		4: Tanh
*/
int getfuncs(void)
{
	a_func = get_a_func(a_func_num);
	agrad_func = get_agrad_func(a_func_num);
	l_func = get_l_func(l_func_num);
	lgrad_func = get_lgrad_func(l_func_num);
	if (a_func == NULL || agrad_func == NULL || lgrad_func == NULL || l_func == NULL)
		return -1;
	return 0;
}
double (*get_a_func(int funcnum))(double)
{
	double (*func)(double);
	if (funcnum == 0)
		func = None;
	else if (funcnum == 1)
		func = ReLU;
	else if (funcnum == 2)
		func = LReLU;
	else if (funcnum == 3)
		func = Sigmoid;
	else if (funcnum == 4)
		func = Tanh;
	else
		func = NULL;
	return func;
}
double (*get_l_func(int funcnum))(double, double)
{
	double (*func)(double, double);
	if (funcnum == 0)
		func = MSE;
	else
		func = NULL;
	return func;
}
double (*get_agrad_func(int funcnum))(double)
{
	double (*func)(double);
	if (funcnum == 0)
		func = grad;
	else if (funcnum == 1)
		func = ReLU_grad;
	else if (funcnum == 2)
		func = LReLU_grad;
	else if (funcnum == 3)
		func = Sigmoid_grad;
	else if (funcnum == 4)
		func = Tanh_grad;
	else
		func = NULL;
	return func;
}
double (*get_lgrad_func(int funcnum))(double, double)
{
	double (*func)(double, double);
	if (funcnum == 0) {
		func = MSE_grad;
	} else {
		func = NULL;
	}
	return func;
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
