/* Exponential Integral Calculation Functions adapted from
	The book Numerical Recipes: The Art of Scientific Computing, Third Edition (2007) is published in hardcover by Cambridge University Press (ISBN-10: 0521880688, or ISBN-13: 978-0521880688)
	See: http://apps.nrbook.com/empanel/index.html?pg=268#
*/




#include <time.h>
#include <iostream>
#include <limits>       // std::numeric_limits
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include "exp_gpu.h"
#include "exp_gpu_double.h"

using namespace std;

float	exponentialIntegralFloat		(const int n,const float x);
double	exponentialIntegralDouble		(const int n,const double x);
void	outputResultsCpu			(const std::vector< std::vector< float  > > &resultsFloatCpu,const std::vector< std::vector< double > > &resultsDoubleCpu);
int	parseArguments				(int argc, char **argv);
void	printUsage				(void);


bool verbose,timing,cpu, gpu, simple, streams, mpi, dynamic;
int maxIterations, jump, nostreams;
unsigned int n,numberOfSamples;
double a,b;	// The interval that we are going to use

int main(int argc, char *argv[]) {
	unsigned int ui,uj;
	cpu=true;
	gpu=true;
	verbose=false;
	timing=false;
	simple=true;
	streams=false;
	dynamic=false;
	// n is the maximum order of the exponential integral that we are going to test
	// numberOfSamples is the number of samples in the interval [0,10] that we are going to calculate
	n=10;
	numberOfSamples=10;
	a=0.0;
	b=10.0;
	maxIterations=2000000000;
	jump=100;
	nostreams=2;
	int rank=0;

	struct timeval expoStart, expoEnd;

	parseArguments(argc, argv);

	if (verbose) {
		cout << "n=" << n << endl;
		cout << "numberOfSamples=" << numberOfSamples << endl;
		cout << "a=" << a << endl;
		cout << "b=" << b << endl;
		cout << "cpu="<<cpu<<endl;
		cout << "dynamic="<<dynamic<<endl;
		cout<<"gpu="<<gpu<<endl;
		cout<<"mpi="<<mpi<<endl;
		cout<<"streams="<<streams<<endl;
		cout<<"shared="<<!simple<<endl;
		cout << "timing=" << timing << endl;
		cout << "verbose=" << verbose << endl;
	}

	// Sanity checks
	if (a>=b) {
		cout << "Incorrect interval ("<<a<<","<<b<<") has been stated!" << endl;
		return 0;
	}
	if (n<=0) {
		cout << "Incorrect orders ("<<n<<") have been stated!" << endl;
		return 0;
	}
	if (numberOfSamples<=0) {
		cout << "Incorrect number of samples ("<<numberOfSamples<<") have been stated!" << endl;
		return 0;
	}

	std::vector< std::vector< float  > > resultsFloatCpu;
	std::vector< std::vector< double > > resultsDoubleCpu;

	double floatTimeCPU=0.0, doubleTimeCPU=0.0;


	double x,division=(b-a)/((double)(numberOfSamples));
	double floatTimeGPU=0.0, doubleTimeGPU=0.0;
	float *CPUresultsFromGPU;
	double *CPUresultsFromGPU_double;


	// *************************************
	// *************************************
	// **********   FLOATS   ***************
	// *************************************
	// *************************************
	
	// Runs on the GPU first
	if(gpu){
		// sets up CPU array to store GPU results
		CPUresultsFromGPU=(float *)malloc(n*numberOfSamples*sizeof(float));

		// Streams Dynamic and MPI
		if(streams && dynamic && mpi){
			floatTimeGPU=gpu_dynamic_streams_mpi(&CPUresultsFromGPU, n, numberOfSamples, a, b, maxIterations, &rank, jump, simple, nostreams);
			if(simple && rank==0){printf("Using Simple MPI with %d streams AND Dynamic\n", nostreams);}
			else if(rank==0){printf("Using Shared MPI with %d streams AND Dynamic\n", nostreams);}
		}

		// Streams and Dynamic
		else if(streams && dynamic){
			floatTimeGPU=gpu_streams_dynamic(&CPUresultsFromGPU, n, numberOfSamples, a, b, maxIterations, simple, nostreams);
			if(simple){printf("Using %d Simple Streams AND Dynamic\n", nostreams);}
			else{printf("Using %d Shared Streams AND Dynamic\n", nostreams);}
			
		}
		
		// MPI and Dynamic
		else if(mpi && dynamic){
			floatTimeGPU=gpu_dynamic_mpi(&CPUresultsFromGPU, n, numberOfSamples, a, b, maxIterations, &rank, jump, simple);
			if(simple && rank==0){printf("Using Simple MPI with Dynamic\n");}
			else if(rank==0){printf("Using Shared MPI with Dynamic\n");}
		}

		// MPI and Streams
		else if(mpi && streams){
			floatTimeGPU=gpu_streams_mpi(&CPUresultsFromGPU, n, numberOfSamples, a, b, maxIterations, &rank, jump, simple, nostreams);
			if(simple && rank==0){printf("Using Simple MPI with %d streams\n", nostreams);}
			else if(rank==0){printf("Using Shared MPI with %d streams\n", nostreams);}
		}

		// Streams Method
		else if(streams){
			floatTimeGPU=gpu_streams(&CPUresultsFromGPU, n, numberOfSamples, a, b, maxIterations, simple, nostreams);
			if(simple){printf("Using %d Simple Streams\n", nostreams);}
			else{printf("Using %d Shared Streams\n", nostreams);}
		}

		// Multiple Cards Method
		else if(mpi){
			floatTimeGPU=gpu_mpi(&CPUresultsFromGPU, n, numberOfSamples, a, b, maxIterations, &rank, jump, simple);
			if(simple && rank==0){printf("Using Simple MPI\n");}
			else if(rank==0){printf("Using Shared MPI\n");}
		}

		// else we first set up the GPU memory
		else{
			float *resultsGPU;
			floatTimeGPU=allocGPU(&resultsGPU, n, numberOfSamples);

			// Dynamic Parallelism Method
			if(dynamic){
				floatTimeGPU+=gpu_dynamic(resultsGPU, n, numberOfSamples, a, b, maxIterations, simple);
				if(simple){printf("Using Simple Dynamic\n");}
				else{printf("Using Shared Dynamic\n");	}
			}

			// Standard Method
			else{
				floatTimeGPU+=gpu_standard(resultsGPU, n, numberOfSamples, a, b, maxIterations, simple);
				if(simple){printf("Using Simple\n");}
				else{printf("Using Shared\n");}
			}

			// Copies back and times
			floatTimeGPU+=copyfromGPU(&CPUresultsFromGPU, &resultsGPU, n, numberOfSamples);
			freeGPU(&resultsGPU);
		}
	}

	// Then runs on cpu (rank==0 check make it only run once if using mpi)
	if (cpu && rank==0) {
		try {
			resultsFloatCpu.resize(n,vector< float >(numberOfSamples));
		} catch (std::bad_alloc const&) {
			cout << "resultsFloatCpu memory allocation fail!" << endl;	exit(1);
		}
		gettimeofday(&expoStart, NULL);
		for (ui=1;ui<=n;ui++) {
			for (uj=1;uj<=numberOfSamples;uj++) {
				x=a+uj*division;
				resultsFloatCpu[ui-1][uj-1]=exponentialIntegralFloat (ui,x);
			}
		}
		gettimeofday(&expoEnd, NULL);
		floatTimeCPU=((expoEnd.tv_sec + expoEnd.tv_usec*0.000001) - (expoStart.tv_sec + expoStart.tv_usec*0.000001));
	}

	// Error checks the two results if run on both
	// Prints number of errors (locations too if verbose turned on)
	if(gpu && cpu && rank==0){
		int errors=0;
		for(unsigned int i=0;i<n;i++){	
			for(unsigned int j=0;j<numberOfSamples;j++){
				if(fabsf(resultsFloatCpu[i][j]-CPUresultsFromGPU[i*numberOfSamples+j])>1.E-5){
					if(verbose)
					std::cout<<i<<" and "<<j<<"\t"<<resultsFloatCpu[i][j]<<"   "<<CPUresultsFromGPU[i*numberOfSamples+j]<<"\n";
					errors++;
				}
			}
		}
		if(errors>0){printf("\n\n");}
		printf("%d errors detected between float cpu and gpu results\n", errors);
		if(errors>0){printf("\n\n");}
	}
	if(gpu){free(CPUresultsFromGPU);}
	




	// *************************************
	// *************************************
	// **********   DOUBLES   ***************
	// *************************************
	// *************************************
	
	// Runs on the GPU first
	if(gpu){
		// sets up CPU array to store GPU results
		CPUresultsFromGPU_double=(double *)malloc(n*numberOfSamples*sizeof(double));

		// Streams, Dynamic and MPI
		if(streams && dynamic && mpi){
			doubleTimeGPU=double_gpu_dynamic_streams_mpi(&CPUresultsFromGPU_double, n, numberOfSamples, a, b, maxIterations, &rank, jump, simple, nostreams);
			//if(simple && rank==0){printf("Using Simple MPI with %d streams AND Dynamic\n", nostreams);}
			//else if(rank==0){printf("Using Shared MPI with %d streams AND Dynamir\n", nostreams);}
		}

		// Streams and Dynamic
		else if(streams && dynamic){
			doubleTimeGPU=double_gpu_streams_dynamic(&CPUresultsFromGPU_double, n, numberOfSamples, a, b, maxIterations, simple, nostreams);
			//if(simple){printf("Using %d Simple Streams AND Dynamic\n", nostreams);}
			//else{printf("Using %d Shared Streams AND Dynamic\n", nostreams);}
			
		}

		// MPI and Dynamic
		else if(mpi && dynamic){
			doubleTimeGPU=double_gpu_dynamic_mpi(&CPUresultsFromGPU_double, n, numberOfSamples, a, b, maxIterations, &rank, jump, simple);
			//if(simple && rank==0){printf("Using Simple MPI with Dynamic\n");}
			//else if(rank==0){printf("Using Shared MPI with Dynamic\n");}
		}

		// MPI and Streams
		else if(mpi && streams){
			doubleTimeGPU=double_gpu_streams_mpi(&CPUresultsFromGPU_double, n, numberOfSamples, a, b, maxIterations, &rank, jump, simple, nostreams);
			//if(simple && rank==0){printf("Using Simple MPI with %d streams\n", nostreams);}
			//else if(rank==0){printf("Using Shared MPI with %d streams\n", nostreams);}
		}

		// Streams Method
		else if(streams){
			doubleTimeGPU=double_gpu_streams(&CPUresultsFromGPU_double, n, numberOfSamples, a, b, maxIterations, simple, nostreams);
			//if(simple){printf("Using %d Simple Streams\n", nostreams);}
			//else{printf("Using %d Shared Streams\n", nostreams);}
		}

		// Multiple Cards Method
		else if(mpi){
			doubleTimeGPU=double_gpu_mpi(&CPUresultsFromGPU_double, n, numberOfSamples, a, b, maxIterations, &rank, jump, simple);
			//if(simple && rank==0){printf("Using Simple MPI\n");}
			//else if(rank==0){printf("Using Shared MPI\n");}
		}

		// else we first set up the GPU memory
		else{
			double *resultsGPU;
			doubleTimeGPU=double_allocGPU(&resultsGPU, n, numberOfSamples);

			// Dynamic Parallelism Method
			if(dynamic){
				doubleTimeGPU+=double_gpu_dynamic(resultsGPU, n, numberOfSamples, a, b, maxIterations, simple);
				//if(simple){printf("Using Simple Dynamic\n");}
				//else{printf("Using Shared Dynamic\n");	}
			}

			// Standard Method
			else{
				doubleTimeGPU+=double_gpu_standard(resultsGPU, n, numberOfSamples, a, b, maxIterations, simple);
				//if(simple){printf("Using Simple\n");}
				//else{printf("Using Shared\n");}
			}

			// Copies back and times
			doubleTimeGPU+=double_copyfromGPU(&CPUresultsFromGPU_double, &resultsGPU, n, numberOfSamples);
			double_freeGPU(&resultsGPU);
		}
	}

	// Then runs on cpu (rank==0 check make it only run once if using mpi)
	if (cpu && rank==0) {
		try {
			resultsDoubleCpu.resize(n,vector< double >(numberOfSamples));
		} catch (std::bad_alloc const&) {
			cout << "resultsDoubleCpu memory allocation fail!" << endl;	exit(1);
		}
		gettimeofday(&expoStart, NULL);
		for (ui=1;ui<=n;ui++) {
			for (uj=1;uj<=numberOfSamples;uj++) {
				x=a+uj*division;
				resultsDoubleCpu[ui-1][uj-1]=exponentialIntegralDouble (ui,x);
			}
		}
		gettimeofday(&expoEnd, NULL);
		doubleTimeCPU=((expoEnd.tv_sec + expoEnd.tv_usec*0.000001) - (expoStart.tv_sec + expoStart.tv_usec*0.000001));
	}

	// Error checks the two results if run on both
	// Prints number of errors (locations too if verbose turned on)
	if(gpu && cpu && rank==0){
		int errors=0;
		for(unsigned int i=0;i<n;i++){	
			for(unsigned int j=0;j<numberOfSamples;j++){
				if(fabsf(resultsDoubleCpu[i][j]-CPUresultsFromGPU_double[i*numberOfSamples+j])>1.E-5){
					if(verbose)
					std::cout<<i<<" and "<<j<<"\t"<<resultsDoubleCpu[i][j]<<"   "<<CPUresultsFromGPU_double[i*numberOfSamples+j]<<"\n";
					errors++;
				}
			}
		}
		if(errors>0){printf("\n\n");}
		printf("%d errors detected between double cpu and gpu results\n", errors);
		if(errors>0){printf("\n\n");}
	}
	if(gpu){free(CPUresultsFromGPU_double);}








	// Finally prints the timings on cpu and/or gpu with speed-ups if both
	if (timing && rank==0) {
		if (cpu) {
			printf ("Calculating the exponentials on the CPU took:\n\t%f seconds for floats\n\t%f seconds for doubles\n",floatTimeCPU, doubleTimeCPU);
		}
		if(gpu){
			printf("Calculating the exponentials on the GPU took:\n\t%f seconds for floats\n\t%f seconds for doubles\n", floatTimeGPU, doubleTimeGPU);
		}
		if(cpu && gpu){
			printf("Speed up is:\n\t%fx for floats\n\t%fx for doubles\n", floatTimeCPU/floatTimeGPU, doubleTimeCPU/doubleTimeGPU);
		}
	}

	if (verbose){
		if (cpu) {
			outputResultsCpu (resultsFloatCpu,resultsDoubleCpu);
		}
	}
	return 0;
}

void	outputResultsCpu(const std::vector< std::vector< float  > > &resultsFloatCpu, const std::vector< std::vector< double > > &resultsDoubleCpu) {
	unsigned int ui,uj;
	double x,division=(b-a)/((double)(numberOfSamples));

	for (ui=1;ui<=n;ui++) {
		for (uj=1;uj<=numberOfSamples;uj++) {
			x=a+uj*division;
			std::cout << "CPU==> exponentialIntegralDouble (" << ui << "," << x <<")=" << resultsDoubleCpu[ui-1][uj-1] << " ,";
			std::cout << "exponentialIntegralFloat  (" << ui << "," << x <<")=" << resultsFloatCpu[ui-1][uj-1] << endl;
		}
	}
}
double exponentialIntegralDouble (const int n,const double x) {
	static const double eulerConstant=0.5772156649015329;
	double epsilon=1.E-30;
	double bigDouble=std::numeric_limits<double>::max();
	int i,ii,nm1=n-1;
	double a,b,c,d,del,fact,h,psi,ans=0.0;


	if (n<0.0 || x<0.0 || (x==0.0&&( (n==0) || (n==1) ) ) ) {
		cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
		exit(1);
	}
	if (n==0) {
		ans=exp(-x)/x;
	} else {
		if (x>1.0) {
			b=x+n;
			c=bigDouble;
			d=1.0/b;
			h=d;
			for (i=1;i<=maxIterations;i++) {
				a=-i*(nm1+i);
				b+=2.0;
				d=1.0/(a*d+b);
				c=b+a/c;
				del=c*d;
				h*=del;
				if (fabs(del-1.0)<=epsilon) {
					ans=h*exp(-x);
					return ans;
				}
			}
			//cout << "Continued fraction failed in exponentialIntegral" << endl;
			return ans;
		} else { // Evaluate series
			ans=(nm1!=0 ? 1.0/nm1 : -log(x)-eulerConstant);	// First term
			fact=1.0;
			for (i=1;i<=maxIterations;i++) {
				fact*=-x/i;
				if (i != nm1) {
					del = -fact/(i-nm1);
				} else {
					psi = -eulerConstant;
					for (ii=1;ii<=nm1;ii++) {
						psi += 1.0/ii;
					}
					del=fact*(-log(x)+psi);
				}
				ans+=del;
				if (fabs(del)<fabs(ans)*epsilon) return ans;
			}
			//cout << "Series failed in exponentialIntegral" << endl;
			return ans;
		}
	}
	return ans;
}

float exponentialIntegralFloat (const int n,const float x) {
	static const float eulerConstant=0.5772156649015329;
	float epsilon=1.E-30;
	float bigfloat=std::numeric_limits<float>::max();
	int i,ii,nm1=n-1;
	float a,b,c,d,del,fact,h,psi,ans=0.0;

	if (n<0.0 || x<0.0 || (x==0.0&&( (n==0) || (n==1) ) ) ) {
		cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
		exit(1);
	}
	if (n==0) {
		ans=exp(-x)/x;
	} else {
		if (x>1.0) {
			b=x+n;
			c=bigfloat;
			d=1.0/b;
			h=d;
			for (i=1;i<=maxIterations;i++) {
				a=-i*(nm1+i);
				b+=2.0;
				d=1.0/(a*d+b);
				c=b+a/c;
				del=c*d;
				h*=del;
				if (fabs(del-1.0)<=epsilon) {
					ans=h*exp(-x);
					return ans;
				}
			}
			ans=h*exp(-x);
			return ans;
		} else { // Evaluate series
			ans=(nm1!=0 ? 1.0/nm1 : -log(x)-eulerConstant);	// First term
			fact=1.0;
			for (i=1;i<=maxIterations;i++) {
				fact*=-x/i;
				if (i != nm1) {
					del = -fact/(i-nm1);
				} else {
					psi = -eulerConstant;
					for (ii=1;ii<=nm1;ii++) {
						psi += 1.0/ii;
					}
					del=fact*(-log(x)+psi);
				}
				ans+=del;
				if (fabs(del)<fabs(ans)*epsilon) return ans;
			}
			return ans;
		}
	}
	return ans;
}


int parseArguments (int argc, char *argv[]) {
	int c;

	while ((c = getopt (argc, argv, "cghi:n:m:a:b:tvsr:pj:d")) != -1) {
		switch(c) {
			case 'c':
				cpu=false; break;	 //Skip the CPU test
			case 'g':
				gpu=false; break;
			case 'h':
				printUsage(); exit(0); break;
			case 'i':
				maxIterations = atoi(optarg); break;
			case 'n':
				n = atoi(optarg); break;
			case 'm':
				numberOfSamples = atoi(optarg); break;
			case 'a':
				a = atof(optarg); break;
			case 'b':
				b = atof(optarg); break;
			case 't':
				timing = true; break;
			case 'v':
				verbose = true; break;
			case 's':
				simple=false; break;
			case 'r':
				streams=true; 
				nostreams=atoi(optarg);
				if(nostreams==0){printf("ERROR: Please specify number of streams\n");exit(1);}
				break;
			case 'p':
				mpi=true; break;
			case 'j':
				jump=atoi(optarg);break;
			case 'd':
				dynamic=true;break;
			default:
				fprintf(stderr, "Invalid option given\n");
				printUsage();
				return -1;
		}
	}
	return 0;
}
void printUsage () {
	printf("exponentialIntegral program\n");
	printf("This program will calculate a number of exponential integrals\n");
	printf("usage:\n");
	printf("exponentialIntegral.out [options]\n");
	printf("      -a   value   : will set the a value of the (a,b) interval in which the samples are taken to value (default: 0.0)\n");
	printf("      -b   value   : will set the b value of the (a,b) interval in which the samples are taken to value (default: 10.0)\n");
	printf("      -c           : will skip the CPU test\n");
	printf("      -d           : will use dynamic parallelism\n");
	printf("      -g           : will skip the GPU test\n");
	printf("      -h           : will show this usage\n");
	printf("      -i   size    : will set the number of iterations to size (default: 2000000000)\n");
	printf("      -j   size    : will set the number of n's each slave runs at a time for multiple cards (default: 100)\n");
	printf("      -m   size    : will set the number of samples taken in the (a,b) interval to size (default: 10)\n");
	printf("      -n   size    : will set the n (the order up to which we are calculating the exponential integrals) to size (default: 10)\n");
	printf("      -p           : will use multiple cards (requires mpirun)\n");
	printf("      -r   size    : turns on streams sets how many to use\n");
	printf("      -s           : uses shared memory\n");
	printf("      -t           : will output the amount of time that it took to generate each norm (default: no)\n");
	printf("      -v           : will activate the verbose mode  (default: no)\n");
	printf("     \n");
}
