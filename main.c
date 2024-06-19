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
//network gradients
W wg;
B bg;
//expected network
W ew;
B eb;
V ev;
//extern options
int a_func_num = 0;
int l_func_num = 0;
double eta = 0.001;
int batch_size = 256;

int main(int const argc, char* const argv[])
{
	//init
	int seed = 0;
	//seed rand
	seed = strand();
	double l_exp = 0.0001;
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
			printf("%sERROR: error opening file. \n%s", COLOR_ERROR, COLOR_END);
			return -1;
		}
		//TODO: change whole write to file system. 
		fprintf(csvfilep, "\n");
	}
	//get functions
	if (getfuncs() == -1) {
		printf("%sERROR: error activation or loss function. \n%s", COLOR_ERROR, COLOR_END);
		return -1;
	}
	//count iteration
	int iter = 0;
	do {
		//calc batch
		if (use_thread == true) {
			//times of threads to run
			double tt = (float)batch_size / (float)thread_size;
			//executed thread number
			int exedt = 0;
			//use muti-thread to calculate batch
			pthread_t tid[thread_size];
			void* ret = NULL;
			for (int i = 0; i < tt; i++)
				pthread_create(&tid[i], NULL, calc_batch, NULL);
			for (int i = 0; i < tt; i++) {
				pthread_join(tid[i], &ret);
				if (ret != NULL) {
					printf("%sERROR: undefined activation function number. \n%s", COLOR_ERROR, COLOR_END);
					return -1;
				}
			}
			if (batch_size > exedt) {
				//remaining threads to be executed
				int rmnt = batch_size - exedt;
				for (int i = 0; i < rmnt; i++)
					pthread_create(&tid[i], NULL, calc_batch, NULL);
				for (int i = 0; i < rmnt; i++)
					pthread_join(tid[i], &ret);
				if (ret != NULL) {
					printf("%sERROR: undefined activation function number. \n%s", COLOR_ERROR, COLOR_END);
					return -1;
				}
			}
		} else if (use_thread == false) {
			//normal calculate batch
			void* ret;
			for (int i = 0; i < batch_size; i++) {
				ret = calc_batch(NULL);
				if (ret != NULL) {
					printf("%sERROR: undefined activation function number. \n%s", COLOR_ERROR, COLOR_END);
					return -1;
				}
			}
		}
		//calc average
		avg_wbv();
		//backward propagation
		bdp();
		avg_gl();
		//update weights & biases
		gd();
		//print results
		//TODO: change whole write to file system. 
		if (writetofile == true)
			fprintf(csvfilep, "\n");
		if (verbose == true)
			printf("%siter = %d, lall = %.*f; \n%s", COLOR_NORM, iter, FPP, v.lall, COLOR_END);
		//check if gradient explosion
		if (isfinite(v.lall) != true) {
			printf("%sERROR: loss not finite, probably gradient explosion. \n%sseed = %d, \niter = %d, \neta = %.*f, batch_size = %d, \nl = %.*f, l_exp = %.*f\n%s", COLOR_ERROR, COLOR_NORM, seed, iter, FPP, eta, batch_size, FPP, v.lall, FPP, l_exp, COLOR_END);
			if (writetofile == true)
				fclose(csvfilep);
			return -1;
		}
		iter++;
	} while (v.lall >= l_exp);
	printf("%sSUCC: l >= l_exp. \n%sseed = %d, \niter = %d, \neta = %.*f, batch_size = %d, \nlall = %.*f, l_exp = %.*f\n%s", COLOR_SUCC, COLOR_NORM, seed, iter, FPP, eta, batch_size, FPP, v.lall, FPP, l_exp, COLOR_END);
	if (writetofile == true)
		fclose(csvfilep);
	return 0;
}
