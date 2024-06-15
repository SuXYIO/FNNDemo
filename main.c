#include "head.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

//network
W w;
B b;
V v;
//extern options
int a_func_num = 0;

//TODO: integrate the whole system with mutilayer
int main(int const argc, char* const argv[])
{
	//init
	int seed = 0;
	//seed rand
	seed = strand();
	double l = 0.0;
	double grad_w = 0.0;
	double grad_b = 0.0;
	double eta = 0.001;
	double l_exp = 0.0001;
	int batch_size = 256;
	int thread_size = 0;
	//check command args
	//store opts
	bool verbose = false;
	bool use_thread = false;
	bool writetofile = false;
	char csvfilename[STR_BUFSIZE];
	FILE* csvfilep = NULL;
	//tmp var
	int o;
	const char* optstring = "Vts:f:A:b:e:l:vh";
	while ((o = getopt(argc, argv, optstring)) != -1) {
		switch (o) {
			case 'V':
				verbose = true;
				break;
			case 't':
				use_thread = true;
				break;
			case 's':
				seed = atoi(optarg);
				srand(atoi(optarg));
				break;
			case 'f':
				writetofile = true;
				strcpy(csvfilename, optarg);
				break;
			case 'A':
				a_func_num = atoi(optarg);
				break;
			case 'b':
				batch_size = atoi(optarg);
				break;
			case 'e':
				eta = atof(optarg);
				break;
			case 'l':
				l_exp = atof(optarg);
				break;
			case 'v':
				printversion();
				return 0;
				break;
			case 'h':
				manualpage();
				return 0;
				break;
			case '?':
				printf("%sError: invalid option: '%c'. %s\nUse \"./LR.out -h\" for help. \n%s", COLOR_ERROR, optopt, COLOR_END, COLOR_END);
				return -1;
				break;
		}
	}
	//init weights and biases with nml distro
	init_wbv();
	//init thread_size
	//prevent too many threads
	if (use_thread == true) {
		if (batch_size <= MAX_THREADS)
			thread_size = batch_size;
		else
			thread_size = MAX_THREADS;
	}
	//open file if -f is on
	//check if file is usable
	if (writetofile == true) {
		int usable = -1;
		usable = isusablefile(csvfilename);
		if (usable == 0) {
			//file usable
			writetofile = true;
		} else if (usable == 1) {
			//disable writetofile
			writetofile = false;
		} else if (usable == 2) {
			//quit
			return 0;
		} else {
			//error
			printf("%sERROR: return of isusablefile() is invalid. \n%sReturn of isusablefile() = %d\n%s", COLOR_ERROR, COLOR_END, usable, COLOR_END);
			return -1;
		}
	}
	//open file and write head
	if (writetofile == true) {
		csvfilep = fopen(csvfilename, "w");
		if (csvfilep == NULL) {
			printf("%sERROR: error opening file. \n%sFile: %s, from option '-f'\n%s", COLOR_ERROR, COLOR_END, csvfilename, COLOR_END);
			return -1;
		}
		//TODO: change whole write to file system. 
		fprintf(csvfilep, "f_w,f_b,g_w,g_b,l,grad_w,grad_b\n");
	}
	//count iteration
	int iter = 0;
	do {
		//calc batch
		if (use_thread == true) {
			//use muti-thread to calculate batch
			pthread_t tid[thread_size];
			void* ret = NULL;
			void* args[3] = {&l, &grad_w, &grad_b};
			for (int i = 0; i < thread_size; i++)
				pthread_create(&tid[i], NULL, calc_batch, args);
			for (int i = 0; i < thread_size; i++) {
				pthread_join(tid[i], &ret);
				if (ret != NULL) {
					printf("%sERROR: undefined activation function number. %sFrom option '-A', a_func_num = %d. \n%s", COLOR_ERROR, COLOR_END, a_func_num, COLOR_END);
					return -1;
				}
			}
		} else if (use_thread == false) {
			//normal calculate batch
			void* args[3] = {&l, &grad_w, &grad_b};
			void* ret;
			for (int i = 0; i < batch_size; i++) {
				ret = calc_batch(args);
				if (ret != NULL) {
					printf("%sERROR: undefined activation function number. %sFrom option '-A', a_func_num = %d. \n%s", COLOR_ERROR, COLOR_END, a_func_num, COLOR_END);
					return -1;
				}
			}
		}
		//calc loss & gradients (average of batch)
		l /= batch_size;
		grad_w /= batch_size;
		grad_b /= batch_size;
		//update weights & biases
		nf.w -= eta * grad_w;
		nf.b -= eta * grad_b;
		//print results
		//TODO: change whole write to file system. 
		if (writetofile == true)
			fprintf(csvfilep, "%.*f,%.*f,%.*f,%.*f,%.*f,%.*f,%.*f\n", FPP, nf.w, FPP, nf.b, FPP, ng.w, FPP, ng.b, FPP, l, FPP, grad_w, FPP, grad_b);
		if (verbose == true)
			printf("%siter = %d, nf.w = %.*f, nf.b = %.*f, ng.w = %.*f, ng.b = %.*f, l = %.*f, grad_w = %.*f, grad_b = %.*f; \n%s", COLOR_END, iter, FPP, nf.w, FPP, nf.b, FPP, ng.w, FPP, ng.b, FPP, l, FPP, grad_w, FPP, grad_b, COLOR_END);
		//check if gradient explosion
		if (isfinite(l) != true || isfinite(grad_w) != true || isfinite(grad_b) != true) {
			printf("%sERROR: l or grad_w or grad_b not finite, probably gradient explosion. \n%siter = %d, \nnf.w = %.*f, nf.b = %.*f, \nng.w = %.*f, ng.b = %.*f, \neta = %.*f, batch_size = %d, \nl = %.*f, l_exp = %.*f\n%s", COLOR_ERROR, COLOR_END, iter, FPP, nf.w, FPP, nf.b, FPP, ng.w, FPP, ng.b, FPP, eta, batch_size, FPP, l, FPP, l_exp, COLOR_END);
			if (writetofile == true)
				fclose(csvfilep);
			return -1;
		}
		iter++;
	} while (l >= l_exp);
	printf("%sSUCC: l >= l_exp. \n%sseed = %d, \niter = %d, \nnf.w_init = %.*f, nf.b_init = %.*f, \nnf.w = %.*f, nf.b = %.*f, \nng.w = %.*f, ng.b = %.*f, \neta = %.*f, batch_size = %d, \nl = %.*f, l_exp = %.*f\n%s", COLOR_SUCC, COLOR_END, seed, iter, FPP, nf_w_init, FPP, nf_b_init, FPP, nf.w, FPP, nf.b, FPP, ng.w, FPP, ng.b, FPP, eta, batch_size, FPP, l, FPP, l_exp, COLOR_END);
	if (writetofile == true)
		fclose(csvfilep);
	return 0;
}
