#include "exp_gpu_double.h"
#include <mpi.h>

typedef double gputype;

#define N 128	// uses fixed value for block size for use with shared memory

/*
// returns number of available cards
extern int getnocards(){
	int num;
	cudaGetDeviceCount(&num);
	return num;
}
*/	


// times and allocates memory to GPU
extern float double_allocGPU(gputype **results, int n, int m){
	cudaEvent_t start, finish;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start, 0);
	cudaMalloc( (void **) results, sizeof(gputype)*n*m);
	cudaEventRecord(finish, 0);

	cudaEventSynchronize(start);
	cudaEventSynchronize(finish);
	
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, finish);
	return elapsedTime/1000.0;
}


// times and copies result back from GPU
extern float double_copyfromGPU(gputype **CPUresults, gputype **resultsGPU, int n, int m){
	cudaEvent_t start, finish;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start, 0);
	cudaMemcpy(*CPUresults, *resultsGPU, n*m*sizeof(gputype), cudaMemcpyDeviceToHost);
	cudaEventRecord(finish, 0);

	cudaEventSynchronize(start);
	cudaEventSynchronize(finish);
	
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, finish);
	return elapsedTime/1000.0;
}

// frees up memory on GPU
extern void double_freeGPU(gputype **results){
	cudaFree(*results);
}





// *************************************
// *************************************
// ********** DEVICE FUNCTIONS *********
// *************************************
// *************************************


// 2 device functions that are used for the update step depending on whether shared memory is used or not


// *************************************
// ********* SIMPLE VERSION ************
// *************************************

// device function to run a single update step for a given n and x
__device__ gputype simple_update(int n, gputype x, int maxIterations){
	static const gputype eulerConstant=0.5772156649015329;
        gputype epsilon=1.E-30;
        gputype biggputype=1.E30;
        int i,ii,nm1=n-1;
     	gputype a,b,c,d,del,fact,h,psi,ans=0.0;
        if (n<0.0 || x<0.0 || (x==0.0&&( (n==0) || (n==1) ) ) ) {
                //cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
                //exit(1);
        }
        if (n==0) {
                ans=exp(-x)/x;
        } else {
                if (x>1.0) {
                        b=x+n;
                        c=biggputype;
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
                        ans=(nm1!=0 ? 1.0/nm1 : -log(x)-eulerConstant); // First term
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



// *************************************
// ********* SHARED VERSION ************
// *************************************
// device function to run a single update step for a given x and n using shared memory
// An array of shared memory is used for each variable to avoid having to use registers
__device__ gputype shared_update(int n, int tid, int maxIterations, gputype mem[N][10], gputype *constants){
        int i,ii,nm1=n-1;
        if (n<0.0 || mem[tid][9]<0.0 || (mem[tid][9]==0.0&&( (n==0) || (n==1) ) ) ) {
                //cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
                //exit(1);
        }
        if (n==0) {
                mem[tid][8]=exp(-mem[tid][9])/mem[tid][9];
        } else {
                if (mem[tid][9]>1.0) {
                        mem[tid][1]=mem[tid][9]+n;
                        mem[tid][2]=constants[2];
                        mem[tid][3]=1.0/mem[tid][1];
                        mem[tid][6]=mem[tid][3];
                        for (i=1;i<=maxIterations;i++) {
                                mem[tid][0]=-i*(nm1+i);
                                mem[tid][1]+=2.0;
                                mem[tid][3]=1.0/(mem[tid][0]*mem[tid][3]+mem[tid][1]);
                                mem[tid][2]=mem[tid][1]+mem[tid][0]/mem[tid][2];
                                mem[tid][4]=mem[tid][2]*mem[tid][3];
                                mem[tid][6]*=mem[tid][4];
                                if (fabs(mem[tid][4]-1.0)<=constants[1]) {
                                        mem[tid][8]=mem[tid][6]*exp(-mem[tid][9]);
                                        return mem[tid][8];
                                }
                        }
                        mem[tid][8]=mem[tid][6]*exp(-mem[tid][9]);
                        return mem[tid][8];
                } else { // Evaluate series
                        mem[tid][8]=(nm1!=0 ? 1.0/nm1 : -log(mem[tid][9])-constants[0]); // First term
                        mem[tid][5]=1.0;
                        for (i=1;i<=maxIterations;i++) {
                                mem[tid][5]*=-mem[tid][9]/i;
                                if (i != nm1) {
                                        mem[tid][4] = -mem[tid][5]/(i-nm1);
                                } else {
                                        mem[tid][7] = -constants[0];
                                        for (ii=1;ii<=nm1;ii++) {
                                                mem[tid][7] += 1.0/ii;
                                        }
                                        mem[tid][4]=mem[tid][5]*(-log(mem[tid][9])+mem[tid][7]);
                                }
                                mem[tid][8]+=mem[tid][4];
                                if (fabs(mem[tid][4])<fabs(mem[tid][8])*constants[1]) return mem[tid][8];
                        }
                        return mem[tid][8];
                }
        }
        return mem[tid][8];

}




















// *************************************
// *************************************
// ******** GLOBAL FUNCTIONS ***********
// *************************************
// *************************************

// different global functions for each combination of streams, multiple cards and dynamic parallelism
// for each combo there is a shared and non-shared version

// *************************************
// ********* SIMPLE n*m VERSION ********
// *************************************
// runs nxm threads to calculate each value
__global__ void simplenxmGPU(gputype *results, int n, int m, gputype a, gputype division, int maxIterations){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int nval=idx/m+1;
	int mval=idx%m+1;


	// runs for each valid entry
	if(idx<n*m){
		gputype x=a+mval*division;
		results[idx]=simple_update(nval, x, maxIterations);
	}
}


// *************************************
// ********* SHARED n*m VERSION ********
// *************************************
// sets up the shared memory and runs nxm threads
__global__ void sharednxmGPU(gputype *results, int n, int m, gputype a, gputype division, int maxIterations){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int tid=threadIdx.x;
	int nval=idx/m+1;
	int mval=idx%m+1;

	// 10 values are needed for each thread
	__shared__ gputype mem[N][10];
	// and every thread needs access to these three constants
	__shared__ gputype constants[3];
	constants[0]=0.5772156649015329;
	constants[1]=1.E-30;
	constants[2]=1.E30;

	if(idx<n*m){
		mem[tid][9]=a+mval*division;	// sets the x value
		results[idx]=shared_update(nval, tid, maxIterations, mem, constants);
	}
}







// *************************************
// **** SIMPLE Stream n*m VERSION ******
// *************************************
// similar to standard version but specifies a start position (which changes for each stream)
__global__ void stream_kernel(gputype *results, int n, int m, gputype a, gputype division, int maxIterations, int start){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int nval=idx/m+1+start;
	int mval=idx%m+1;


	// runs for each valid value
	if(idx<n*m){
		gputype x=a+mval*division;
		results[idx+start*m]=simple_update(nval, x, maxIterations);
	}
}


// *************************************
// **** SHARED Stream n*m VERSION ******
// *************************************
// similar to standard version but specifies a start position (which changes for each stream)
__global__ void stream_shared_kernel(gputype *results, int n, int m, gputype a, gputype division, int maxIterations, int start){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int tid=threadIdx.x;
	int nval=idx/m+1+start;
	int mval=idx%m+1;

	__shared__ gputype mem[N][10];
	__shared__ gputype constants[3];
	constants[0]=0.5772156649015329;
	constants[1]=1.E-30;
	constants[2]=1.E30;


	// runs for each valid value
	if(idx<n*m){
		mem[tid][9]=a+mval*division;
		results[idx+start*m]=shared_update(nval, tid, maxIterations, mem, constants);
	}
}







// *************************************
// **** SIMPLE MPI n*m VERSION ******
// *************************************
// runs the calculations for 'jump' amount of values starting from 'nval'
__global__ void simple_mpi(gputype *results, int nval, int m, gputype a, gputype division, int maxIterations, int jump){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	nval+=idx/m;
	int mval=idx%m+1;

	// runs for each valid value
	if(idx<m*jump){
		gputype x=a+mval*division;
		results[idx]=simple_update(nval, x, maxIterations);
	}
}


// *************************************
// **** SHARED MPI n*m VERSION ******
// *************************************
// runs the calculations for 'jump' amount of values starting from 'nval'
__global__ void shared_mpi(gputype *results, int nval, int m, gputype a, gputype division, int maxIterations, int jump){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int tid=threadIdx.x;
	nval+=idx/m;
	int mval=idx%m+1;

	__shared__ gputype mem[N][10];
	__shared__ gputype constants[3];
	constants[0]=0.5772156649015329;
	constants[1]=1.E-30;
	constants[2]=1.E30;

	// runs for each valid value
	if(idx<m*jump){
		mem[tid][9]=a+mval*division;
		results[idx]=shared_update(nval, tid, maxIterations, mem, constants);
	}
}







// *************************************
// **** SIMPLE DYNAMIC n*m VERSION *****
// *************************************
// the child kernel is spawned for each n and then runs m threads for its given n
__global__ void child(gputype *results, int nval, int m, gputype a, gputype division, int maxIterations){
	int idx=blockIdx.x*blockDim.x+threadIdx.x+1;
	if(idx<=m){
		gputype x=a+idx*division;
		results[(nval-1)*m+idx-1]=simple_update(nval, x, maxIterations);
	}
}

// spawns the child kernel for each n
__global__ void simpledynamicGPU(gputype *results, int n, int m, gputype a, gputype division, int maxIterations){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;


	// runs for each valid value
	if(idx<n){
		dim3 dimBlock(N);
		dim3 dimGrid((m/dimBlock.x)+(!(m%dimBlock.x)?0:1));
		child<<<dimGrid, dimBlock>>>(results, idx+1, m, a, division, maxIterations);
	}
}


// *************************************
// **** SHARED DYNAMIC n*m VERSION *****
// *************************************
// the child kernel is spawned for each n and then runs m threads for its given n
__global__ void sharedchild(gputype *results, int nval, int m, gputype a, gputype division, int maxIterations){
	int idx=blockIdx.x*blockDim.x+threadIdx.x+1;
	int tid=threadIdx.x;

	__shared__ gputype mem[N][10];
	__shared__ gputype constants[3];
	constants[0]=0.5772156649015329;
	constants[1]=1.E-30;
	constants[2]=1.E30;

	if(idx<=m){
		mem[tid][9]=a+idx*division;
		results[(nval-1)*m+idx-1]=shared_update(nval, tid, maxIterations, mem, constants);
	}
}

// spawns the child kernel for each n
__global__ void shareddynamicGPU(gputype *results, int n, int m, gputype a, gputype division, int maxIterations){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;


	// runs for each valid value
	if(idx<n){
		dim3 dimBlock(N);
		dim3 dimGrid((m/dimBlock.x)+(!(m%dimBlock.x)?0:1));
		sharedchild<<<dimGrid, dimBlock>>>(results, idx+1, m, a, division, maxIterations);
	}
}




////////////////////////////////////////////
//////////   EXTRA COMBOS //////////////////
////////////////////////////////////////////





// *************************************
// **** SIMPLE STREAM DYNAMIC n*m VERSION *****
// *************************************
// each stream runs its given n values, each of which spawns the child kernel
__global__ void dynamic_stream_kernel(gputype *results, int n, int m, gputype a, gputype division, int maxIterations, int start){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;


	// runs for each valid value
	if(idx<n){
		dim3 dimBlock(N);
		dim3 dimGrid((m/dimBlock.x)+(!(m%dimBlock.x)?0:1));
		child<<<dimGrid, dimBlock>>>(results, idx+1+start, m, a, division, maxIterations);
	}
}


// *************************************
// **** SHARED STREAM DYNAMIC n*m VERSION *****
// *************************************
// each stream runs its given n values, each of which spawns the child kernel
__global__ void dynamic_stream_shared_kernel(gputype *results, int n, int m, gputype a, gputype division, int maxIterations, int start){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;


	// runs for each valid value
	if(idx<n){
		dim3 dimBlock(N);
		dim3 dimGrid((m/dimBlock.x)+(!(m%dimBlock.x)?0:1));
		sharedchild<<<dimGrid, dimBlock>>>(results, idx+1+start, m, a, division, maxIterations);
	}
}







// *************************************
// **** SIMPLE MPI DYNAMIC n*m VERSION *****
// *************************************
// mpi version of the child kernel (changes where the result is stored as each MPI process has its own results array
__global__ void mpichild(gputype *results, int nval, int m, gputype a, gputype division, int maxIterations, int storepoint){
	int idx=blockIdx.x*blockDim.x+threadIdx.x+1;
	if(idx<=m){
		gputype x=a+idx*division;
		results[storepoint*m+idx-1]=simple_update(nval, x, maxIterations);
	}
}

// runs for 'jump' values of n starting from 'nval' and spawns the child each time
__global__ void simple_dynamic_mpi(gputype *results, int nval, int m, gputype a, gputype division, int maxIterations, int jump){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	nval+=idx;

	// runs for each valid value
	if(idx<jump){
		dim3 dimBlock(N);
		dim3 dimGrid((m/dimBlock.x)+(!(m%dimBlock.x)?0:1));
		mpichild<<<dimGrid, dimBlock>>>(results, nval, m, a, division, maxIterations, idx);
	}
}


// *************************************
// **** SHARED MPI DYNAMIC n*m VERSION *****
// *************************************
// mpi version of the child kernel (changes where the result is stored as each MPI process has its own results array
__global__ void sharedmpichild(gputype *results, int nval, int m, gputype a, gputype division, int maxIterations, int storepoint){
	int idx=blockIdx.x*blockDim.x+threadIdx.x+1;
	int tid=threadIdx.x;

	__shared__ gputype mem[N][10];
	__shared__ gputype constants[3];
	constants[0]=0.5772156649015329;
	constants[1]=1.E-30;
	constants[2]=1.E30;

	if(idx<=m){
		mem[tid][9]=a+idx*division;
		results[storepoint*m+idx-1]=shared_update(nval, tid, maxIterations, mem, constants);
	}
}

// runs for 'jump' values of n starting from 'nval' and spawns the child each time
__global__ void shared_dynamic_mpi(gputype *results, int nval, int m, gputype a, gputype division, int maxIterations, int jump){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	nval+=idx;

	// runs for each valid value
	if(idx<jump){
		dim3 dimBlock(N);
		dim3 dimGrid((m/dimBlock.x)+(!(m%dimBlock.x)?0:1));
		sharedmpichild<<<dimGrid, dimBlock>>>(results, nval, m, a, division, maxIterations, idx);
	}
}







// *************************************
// **** SIMPLE MPI STREAMS VERSION *****
// *************************************
// runs for 'jump' number of  values, where the process starts from 'nval' and the stream starts 'start' values after that
__global__ void simple_streams_mpi(gputype *results, int nval, int m, gputype a, gputype division, int maxIterations, int jump, int start){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	nval+=idx/m+start;
	int mval=idx%m+1;

	// runs for each valid value
	if(idx<m*jump){
		gputype x=a+mval*division;
		results[idx+start*m]=simple_update(nval, x, maxIterations);
	}
}


// *************************************
// **** SHARED MPI STREAMS VERSION *****
// *************************************
// runs for 'jump' number of  values, where the process starts from 'nval' and the stream starts 'start' values after that
__global__ void shared_streams_mpi(gputype *results, int nval, int m, gputype a, gputype division, int maxIterations, int jump, int start){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	nval+=idx/m+start;
	int mval=idx%m+1;
	int tid=threadIdx.x;

	__shared__ gputype mem[N][10];
	__shared__ gputype constants[3];
	constants[0]=0.5772156649015329;
	constants[1]=1.E-30;
	constants[2]=1.E30;


	// runs for each valid value
	if(idx<m*jump){
		mem[tid][9]=a+mval*division;
		results[idx+start*m]=shared_update(nval, tid, maxIterations, mem, constants);
	}
}







// *************************************
// **** SIMPLE MPI STREAMS & DYNAMIC VERSION *****
// *************************************
// runs for 'jump' number of  values, where the process starts from 'nval' and the stream starts 'start' values after that AND each time it spawns a child kernel
__global__ void simple_dynamic_streams_mpi(gputype *results, int nval, int m, gputype a, gputype division, int maxIterations, int jump, int start){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	nval+=idx+start;

	// runs for each valid value
	if(idx<jump){
		dim3 dimBlock(N);
		dim3 dimGrid((m/dimBlock.x)+(!(m%dimBlock.x)?0:1));
		mpichild<<<dimGrid, dimBlock>>>(results, nval, m, a, division, maxIterations, idx+start);

	}
}


// *************************************
// **** SHARED MPI STREAMS & DYNAMIC VERSION *****
// *************************************
// runs for 'jump' number of  values, where the process starts from 'nval' and the stream starts 'start' values after that AND each time it spawns a child kernel
__global__ void shared_dynamic_streams_mpi(gputype *results, int nval, int m, gputype a, gputype division, int maxIterations, int jump, int start){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	nval+=idx+start;

	// runs for each valid value
	if(idx<jump){
		dim3 dimBlock(N);
		dim3 dimGrid((m/dimBlock.x)+(!(m%dimBlock.x)?0:1));
		mpichild<<<dimGrid, dimBlock>>>(results, nval, m, a, division, maxIterations, idx+start);

	}
}































// *************************************
// *************************************
// ******** HOST FUNCTIONS *************
// *************************************
// *************************************



// *************************************
// ********* STANDARD VERSION ********
// *************************************
// times and runs the standard version depending on simple or shared memory
extern float double_gpu_standard(gputype *results, int n, int m, gputype a, gputype b, int maxIterations, bool simple){	
	dim3 dimBlock(N);	
	dim3 dimGrid(((n*m)/dimBlock.x)+(!((n*m)%dimBlock.x)?0:1));

	cudaEvent_t start, finish;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start, 0);
	
	gputype division=(b-a)/(gputype)m;

	if(simple){
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		simplenxmGPU<<<dimGrid, dimBlock>>>(results, n, m, a, division, maxIterations);
	}
	else{
		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		sharednxmGPU<<<dimGrid, dimBlock>>>(results, n, m, a, division, maxIterations);
	}
	
	cudaEventRecord(finish, 0);

	cudaEventSynchronize(start);
	cudaEventSynchronize(finish);
	
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, finish);
	return elapsedTime/1000.0;
}




// *************************************
// ********* STREAM VERSION ************
// *************************************
// runs and times using streams
extern float double_gpu_streams(gputype **CPUresults, int n, int m, gputype a, gputype b, int maxIterations, bool simple, int nostreams){	
	dim3 dimBlock(N);	

	cudaEvent_t start, finish;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start, 0);

	gputype *results;
	cudaStream_t streams[nostreams];	// streams
	int ns[nostreams];			// amount of n values for each stream
	int singlen=n/nostreams;		// all but last stream run n/nostream values
	int i;
	// creates streams and fills ns array
	for(i=0;i<nostreams;i++){
		cudaStreamCreate(&streams[i]);
		ns[i]=singlen;
	}
	ns[nostreams-1]=n-(nostreams-1)*singlen;	// changes last ns value to account for any leftovers
	
	cudaMalloc( (void **) &results, sizeof(gputype)*n*m);
	
	gputype division=(b-a)/(gputype)m;

	// sets grid sizes for each size of n
	dim3 dimGrid_main(((singlen*m)/dimBlock.x)+(!((singlen*m)%dimBlock.x)?0:1));
	dim3 dimGrid_last(((ns[nostreams-1]*m)/dimBlock.x)+(!((ns[nostreams-1]*m)%dimBlock.x)?0:1));

		
	if(simple){
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		// runs all but last stream using the 'singlen' setup
		for(i=0;i<nostreams-1;i++){
			stream_kernel<<<dimGrid_main, dimBlock, 0, streams[i]>>>(results, singlen, m, a, division, maxIterations, i*singlen);
		}
		// then runs the last stream in case the ns value is different
		stream_kernel<<<dimGrid_last, dimBlock, 0, streams[nostreams-1]>>>(results, ns[nostreams-1], m, a, division, maxIterations, (nostreams-1)*singlen);
	}
	// SIMILARLY for shared memory (calls a different kernel)
	else{
		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		for(i=0;i<nostreams-1;i++){
			stream_shared_kernel<<<dimGrid_main, dimBlock, 0, streams[i]>>>(results, singlen, m, a, division, maxIterations, i*singlen);
		}
		stream_shared_kernel<<<dimGrid_last, dimBlock, 0, streams[nostreams-1]>>>(results, ns[nostreams-1], m, a, division, maxIterations, (nostreams-1)*singlen);
	}

	//cudaDeviceSynchronize();
	// async copies the results back
	for(i=0;i<nostreams;i++){
		cudaMemcpyAsync(&((*CPUresults)[i*singlen*m]), &(results[i*singlen*m]), sizeof(gputype)*ns[i]*m, cudaMemcpyDeviceToHost, streams[i]);
	}
	//cudaDeviceSynchronize();
	
	// frees results and destroys streams
	cudaFree(results);
	for(i=0;i<nostreams;i++){
		cudaStreamDestroy(streams[i]);
	}
	cudaEventRecord(finish, 0);

	cudaEventSynchronize(start);
	cudaEventSynchronize(finish);
	
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, finish);
	return elapsedTime/1000.0;
}












// *************************************
// ************ MPI VERSION ************
// *************************************
// runs and times using MPI to use multiple cards
// runs with a process per card and one process as the master
// the master distributes values of n and then each slave calculates the next 'jump' values
// the slave then sends back their results and waits for a new n value
extern float double_gpu_mpi(gputype **CPUresults, int n, int m, gputype a, gputype b, int maxIterations, int *CPUrank, int jump, bool simple){	
	int i;

	dim3 dimBlock(N);	
	dim3 dimGrid(((m*jump)/dimBlock.x)+(!((m*jump)%dimBlock.x)?0:1));

	cudaEvent_t start, finish;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start, 0);

	// MPI admin
	int rank, size;
	MPI_Status stat;
	int no_cards=getnocards();
	//MPI_Init(NULL, NULL);			// commented out as already initialised in floats version
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	*CPUrank=rank;

	// Master Process
	if(rank==0){
		int nval=1, index;
		// if more cards than slaves then only use 'slave' cards
		if(size<no_cards+1){no_cards=size-1;}

		// sends out first job to each slave
		for(i=1;i<=no_cards && nval+jump-1<=n;i++){
			MPI_Send(&nval, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			nval+=jump;
		}

		// while more jobs available, receive results form MPI_ANY_SOURCE and then send back next nval
		while(nval+jump-1<=n){
			// receives the slaves nval first to use as an index to store results in correct place on CPU
			MPI_Recv(&index, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &stat);
			MPI_Recv(&((*CPUresults)[(index-1)*m]), m*jump, MPI_DOUBLE, stat.MPI_SOURCE, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(&nval, 1, MPI_INT, stat.MPI_SOURCE, 0, MPI_COMM_WORLD);
			nval+=jump;
		}

		// Receive results from every slave and send finish signal (via tag)
		for(i=1;i<=no_cards;i++){
			MPI_Recv(&index, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &stat);
			MPI_Recv(&((*CPUresults)[(index-1)*m]), m*jump, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(&i, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
		}

	}
	// Slaves
	else if(rank-1<no_cards){
		cudaSetDevice(rank-1);
		gputype *results;
		cudaMalloc( (void **) &results, sizeof(gputype)*m*jump);
		gputype division=(b-a)/(gputype)m;
		
		int nval;

		if(simple){
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			// loops until told to stop
			while(true){
				// receives value (checks tag to see if it should stop)
				MPI_Recv(&nval, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
				if(stat.MPI_TAG==1){break;}

				// adapts jump value if at near the end
				if(nval+jump-1>n){jump=n-nval+1;}
				// runs calculations
				simple_mpi<<<dimGrid, dimBlock>>>(results, nval, m, a, division, maxIterations, jump);
				// copies back to CPU
				cudaMemcpy(*CPUresults, results, m*jump*sizeof(gputype), cudaMemcpyDeviceToHost);

				// sends index and then results
				MPI_Send(&nval, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
				MPI_Send(*CPUresults, m*jump, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			}
		}
		// SIMILARLY for shared memory
		else{
			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
			while(true){
				MPI_Recv(&nval, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
				if(stat.MPI_TAG==1){break;}

				if(nval+jump-1>n){jump=n-nval+1;}
				shared_mpi<<<dimGrid, dimBlock>>>(results, nval, m, a, division, maxIterations, jump);
				cudaMemcpy(*CPUresults, results, m*jump*sizeof(gputype), cudaMemcpyDeviceToHost);


				MPI_Send(&nval, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
				MPI_Send(*CPUresults, m*jump, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			}
		}
		cudaFree(results);
		
	}
	MPI_Finalize();

	cudaEventRecord(finish, 0);

	cudaEventSynchronize(start);
	cudaEventSynchronize(finish);
	
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, finish);
	return elapsedTime/1000.0;
}









// *************************************
// ******** DYNAMIC VERSION ************
// *************************************
// runs and times dynamic version
extern float double_gpu_dynamic(gputype *results, int n, int m, gputype a, gputype b, int maxIterations, bool simple){	
	dim3 dimBlock(N);	
	dim3 dimGrid((n/dimBlock.x)+(!(n%dimBlock.x)?0:1));

	cudaEvent_t start, finish;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start, 0);
	
	gputype division=(b-a)/(gputype)m;

	// runs dynamic version
	if(simple){
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		simpledynamicGPU<<<dimGrid, dimBlock>>>(results, n, m, a, division, maxIterations);
	}
	else{
		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		shareddynamicGPU<<<dimGrid, dimBlock>>>(results, n, m, a, division, maxIterations);
	}
	
	cudaEventRecord(finish, 0);

	cudaEventSynchronize(start);
	cudaEventSynchronize(finish);
	
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, finish);
	return elapsedTime/1000.0;
}


////////////////////////////////////////////
//////////   EXTRA COMBOS //////////////////
////////////////////////////////////////////
// the rest of the functions use each combination of dynamic, streams and mpi
// they are just adapted versions of the above functions



// *************************************
// **** DYNAMIC & STREAMS VERSION ******
// *************************************
extern float double_gpu_streams_dynamic(gputype **CPUresults, int n, int m, gputype a, gputype b, int maxIterations, bool simple, int nostreams){	
	dim3 dimBlock(N);	

	cudaEvent_t start, finish;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start, 0);

	gputype *results;
	cudaStream_t streams[nostreams];
	int ns[nostreams];
	int singlen=n/nostreams;
	int i;
	for(i=0;i<nostreams;i++){
		cudaStreamCreate(&streams[i]);
		ns[i]=singlen;
	}
	ns[nostreams-1]=n-(nostreams-1)*singlen;
	
	cudaMalloc( (void **) &results, sizeof(gputype)*n*m);
	
	gputype division=(b-a)/(gputype)m;


	dim3 dimGrid_main(((singlen)/dimBlock.x)+(!((singlen)%dimBlock.x)?0:1));
	dim3 dimGrid_last(((ns[nostreams-1])/dimBlock.x)+(!((ns[nostreams-1])%dimBlock.x)?0:1));
	if(simple){
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
		for(i=0;i<nostreams-1;i++){
			dynamic_stream_kernel<<<dimGrid_main, dimBlock, 0, streams[i]>>>(results, singlen, m, a, division, maxIterations, i*singlen);
		}
		dynamic_stream_kernel<<<dimGrid_last, dimBlock, 0, streams[nostreams-1]>>>(results, ns[nostreams-1], m, a, division, maxIterations, (nostreams-1)*singlen);
	}
	else{
		cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
		for(i=0;i<nostreams-1;i++){
			dynamic_stream_shared_kernel<<<dimGrid_main, dimBlock, 0, streams[i]>>>(results, singlen, m, a, division, maxIterations, i*singlen);
		}
		dynamic_stream_shared_kernel<<<dimGrid_last, dimBlock, 0, streams[nostreams-1]>>>(results, ns[nostreams-1], m, a, division, maxIterations, (nostreams-1)*singlen);
	}
	//cudaDeviceSynchronize();

	for(i=0;i<nostreams;i++){
		cudaMemcpyAsync(&((*CPUresults)[i*singlen*m]), &(results[i*singlen*m]), sizeof(gputype)*ns[i]*m, cudaMemcpyDeviceToHost, streams[i]);
	}
	
	//cudaDeviceSynchronize();
	cudaFree(results);

	for(i=0;i<nostreams;i++){
		cudaStreamDestroy(streams[i]);
	}
	cudaEventRecord(finish, 0);

	cudaEventSynchronize(start);
	cudaEventSynchronize(finish);
	
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, finish);
	return elapsedTime/1000.0;
}







// *************************************
// **** DYNAMIC & MPI VERSION ******
// *************************************
extern float double_gpu_dynamic_mpi(gputype **CPUresults, int n, int m, gputype a, gputype b, int maxIterations, int *CPUrank, int jump, bool simple){	
	int i;

	dim3 dimBlock(N);	
	dim3 dimGrid(((jump)/dimBlock.x)+(!((jump)%dimBlock.x)?0:1));

	cudaEvent_t start, finish;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start, 0);


	int rank, size;
	MPI_Status stat;
	int no_cards=getnocards();
	//MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	*CPUrank=rank;

	// Master
	if(rank==0){
		int nval=1, index;
		if(size<no_cards+1){no_cards=size-1;}
		for(i=1;i<=no_cards && nval+jump-1<=n;i++){
			MPI_Send(&nval, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			nval+=jump;
		}
		while(nval+jump-1<=n){
			MPI_Recv(&index, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &stat);
			MPI_Recv(&((*CPUresults)[(index-1)*m]), m*jump, MPI_DOUBLE, stat.MPI_SOURCE, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(&nval, 1, MPI_INT, stat.MPI_SOURCE, 0, MPI_COMM_WORLD);
			nval+=jump;
		}
		for(i=1;i<=no_cards;i++){
			MPI_Recv(&index, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &stat);
			MPI_Recv(&((*CPUresults)[(index-1)*m]), m*jump, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(&i, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
		}

	}
	// Slaves
	else if(rank-1<no_cards){
		cudaSetDevice(rank-1);
		gputype *results;
		cudaMalloc( (void **) &results, sizeof(gputype)*m*jump);
		gputype division=(b-a)/(gputype)m;
		
		int nval;

		if(simple){
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			while(true){
				MPI_Recv(&nval, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
				if(stat.MPI_TAG==1){break;}

				if(nval+jump-1>n){jump=n-nval+1;}
				simple_dynamic_mpi<<<dimGrid, dimBlock>>>(results, nval, m, a, division, maxIterations, jump);
				cudaMemcpy(*CPUresults, results, m*jump*sizeof(gputype), cudaMemcpyDeviceToHost);


				MPI_Send(&nval, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
				MPI_Send(*CPUresults, m*jump, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			}
		}
		else{
			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
			while(true){
				MPI_Recv(&nval, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
				if(stat.MPI_TAG==1){break;}

				if(nval+jump-1>n){jump=n-nval+1;}
				shared_dynamic_mpi<<<dimGrid, dimBlock>>>(results, nval, m, a, division, maxIterations, jump);
				cudaMemcpy(*CPUresults, results, m*jump*sizeof(gputype), cudaMemcpyDeviceToHost);


				MPI_Send(&nval, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
				MPI_Send(*CPUresults, m*jump, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			}
		}
		cudaFree(results);
		
	}
	MPI_Finalize();

	cudaEventRecord(finish, 0);

	cudaEventSynchronize(start);
	cudaEventSynchronize(finish);
	
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, finish);
	return elapsedTime/1000.0;
}


// *************************************
// **** STREAMS & MPI VERSION ******
// *************************************
extern float double_gpu_streams_mpi(gputype **CPUresults, int n, int m, gputype a, gputype b, int maxIterations, int *CPUrank, int jump, bool simple, int nostreams){	
	int i;

	dim3 dimBlock(N);	

	cudaEvent_t start, finish;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start, 0);


	int rank, size;
	MPI_Status stat;
	int no_cards=getnocards();
	//MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	*CPUrank=rank;

	// Master
	if(rank==0){
		int nval=1, index;
		if(size<no_cards+1){no_cards=size-1;}
		for(i=1;i<=no_cards && nval+jump-1<=n;i++){
			MPI_Send(&nval, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			nval+=jump;
		}
		while(nval+jump-1<=n){
			MPI_Recv(&index, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &stat);
			MPI_Recv(&((*CPUresults)[(index-1)*m]), m*jump, MPI_DOUBLE, stat.MPI_SOURCE, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(&nval, 1, MPI_INT, stat.MPI_SOURCE, 0, MPI_COMM_WORLD);
			nval+=jump;
		}
		for(i=1;i<=no_cards;i++){
			MPI_Recv(&index, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &stat);
			MPI_Recv(&((*CPUresults)[(index-1)*m]), m*jump, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(&i, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
		}

	}
	// Slaves
	else if(rank-1<no_cards){
		cudaSetDevice(rank-1);
		gputype *results;
		cudaMalloc( (void **) &results, sizeof(gputype)*m*jump);

		cudaStream_t streams[nostreams];
		int ns[nostreams];
		int singlen;
		int i;
		for(i=0;i<nostreams;i++){
			cudaStreamCreate(&streams[i]);
			ns[i]=singlen;
		}

		gputype division=(b-a)/(gputype)m;

		
		int nval;

		if(simple){
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			while(true){
				MPI_Recv(&nval, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
				if(stat.MPI_TAG==1){break;}

				if(nval+jump-1>n){jump=n-nval+1;}
				singlen=jump/nostreams;
				ns[nostreams-1]=jump-(nostreams-1)*singlen;
				dim3 dimGrid_main(((singlen*m)/dimBlock.x)+(!((singlen*m)%dimBlock.x)?0:1));
				dim3 dimGrid_last(((ns[nostreams-1]*m)/dimBlock.x)+(!((ns[nostreams-1]*m)%dimBlock.x)?0:1));
				for(i=0;i<nostreams-1;i++){
					ns[i]=singlen;
					simple_streams_mpi<<<dimGrid_main, dimBlock, 0, streams[i]>>>(results, nval, m, a, division, maxIterations, singlen, i*singlen);
				}
				simple_streams_mpi<<<dimGrid_last, dimBlock, 0, streams[nostreams-1]>>>(results, nval, m, a, division, maxIterations, jump-(nostreams-1)*singlen, (nostreams-1)*singlen);

				//cudaDeviceSynchronize();
				for(i=0;i<nostreams;i++){
					cudaMemcpyAsync(&((*CPUresults)[(i*singlen)*m]), &(results[i*singlen*m]), sizeof(gputype)*ns[i]*m, cudaMemcpyDeviceToHost, streams[i]);
				}
	
				//cudaDeviceSynchronize();


				MPI_Send(&nval, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
				MPI_Send(*CPUresults, m*jump, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			}
		}
		else{
			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
			while(true){
				MPI_Recv(&nval, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
				if(stat.MPI_TAG==1){break;}

				if(nval+jump-1>n){jump=n-nval+1;}
				singlen=jump/nostreams;
				ns[nostreams-1]=jump-(nostreams-1)*singlen;
				dim3 dimGrid_main(((singlen*m)/dimBlock.x)+(!((singlen*m)%dimBlock.x)?0:1));
				dim3 dimGrid_last(((ns[nostreams-1]*m)/dimBlock.x)+(!((ns[nostreams-1]*m)%dimBlock.x)?0:1));
				for(i=0;i<nostreams-1;i++){
					ns[i]=singlen;
					shared_streams_mpi<<<dimGrid_main, dimBlock, 0, streams[i]>>>(results, nval, m, a, division, maxIterations, singlen, i*singlen);
				}
				shared_streams_mpi<<<dimGrid_last, dimBlock, 0, streams[nostreams-1]>>>(results, nval, m, a, division, maxIterations, jump-(nostreams-1)*singlen, (nostreams-1)*singlen);

				//cudaDeviceSynchronize();
				for(i=0;i<nostreams;i++){
					cudaMemcpyAsync(&((*CPUresults)[(i*singlen)*m]), &(results[i*singlen*m]), sizeof(gputype)*ns[i]*m, cudaMemcpyDeviceToHost, streams[i]);
				}
				//cudaDeviceSynchronize();


				MPI_Send(&nval, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
				MPI_Send(*CPUresults, m*jump, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			}
		}
		for(i=0;i<nostreams;i++){
			cudaStreamDestroy(streams[i]);
		}
		cudaFree(results);
		
	}
	MPI_Finalize();

	cudaEventRecord(finish, 0);

	cudaEventSynchronize(start);
	cudaEventSynchronize(finish);
	
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, finish);
	return elapsedTime/1000.0;
}


// *************************************
// **** STREAMS & MPI  & DYNAMIC VERSION ******
// *************************************
extern float double_gpu_dynamic_streams_mpi(gputype **CPUresults, int n, int m, gputype a, gputype b, int maxIterations, int *CPUrank, int jump, bool simple, int nostreams){	
	int i;

	dim3 dimBlock(N);	

	cudaEvent_t start, finish;
	cudaEventCreate(&start);
	cudaEventCreate(&finish);

	cudaEventRecord(start, 0);


	int rank, size;
	MPI_Status stat;
	int no_cards=getnocards();
	//MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	*CPUrank=rank;

	// Master
	if(rank==0){
		int nval=1, index;
		if(size<no_cards+1){no_cards=size-1;}
		for(i=1;i<=no_cards && nval+jump-1<=n;i++){
			MPI_Send(&nval, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			nval+=jump;
		}
		while(nval+jump-1<=n){
			MPI_Recv(&index, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &stat);
			MPI_Recv(&((*CPUresults)[(index-1)*m]), m*jump, MPI_DOUBLE, stat.MPI_SOURCE, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(&nval, 1, MPI_INT, stat.MPI_SOURCE, 0, MPI_COMM_WORLD);
			nval+=jump;
		}
		for(i=1;i<=no_cards;i++){
			MPI_Recv(&index, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &stat);
			MPI_Recv(&((*CPUresults)[(index-1)*m]), m*jump, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &stat);
			MPI_Send(&i, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
		}

	}
	// Slaves
	else if(rank-1<no_cards){
		cudaSetDevice(rank-1);
		gputype *results;
		cudaMalloc( (void **) &results, sizeof(gputype)*m*jump);

		cudaStream_t streams[nostreams];
		int ns[nostreams];
		int singlen;
		int i;
		for(i=0;i<nostreams;i++){
			cudaStreamCreate(&streams[i]);
			ns[i]=singlen;
		}

		gputype division=(b-a)/(gputype)m;

		
		int nval;

		if(simple){
			cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
			while(true){
				MPI_Recv(&nval, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
				if(stat.MPI_TAG==1){break;}

				if(nval+jump-1>n){jump=n-nval+1;}
				singlen=jump/nostreams;
				ns[nostreams-1]=jump-(nostreams-1)*singlen;
				dim3 dimGrid_main(((singlen)/dimBlock.x)+(!((singlen)%dimBlock.x)?0:1));
				dim3 dimGrid_last(((ns[nostreams-1])/dimBlock.x)+(!((ns[nostreams-1])%dimBlock.x)?0:1));
				for(i=0;i<nostreams-1;i++){
					ns[i]=singlen;
					simple_dynamic_streams_mpi<<<dimGrid_main, dimBlock, 0, streams[i]>>>(results, nval, m, a, division, maxIterations, singlen, i*singlen);
				}
				simple_dynamic_streams_mpi<<<dimGrid_last, dimBlock, 0, streams[nostreams-1]>>>(results, nval, m, a, division, maxIterations, jump-(nostreams-1)*singlen, (nostreams-1)*singlen);
				
				//cudaDeviceSynchronize();
				for(i=0;i<nostreams;i++){
					cudaMemcpyAsync(&((*CPUresults)[(i*singlen)*m]), &(results[i*singlen*m]), sizeof(gputype)*ns[i]*m, cudaMemcpyDeviceToHost, streams[i]);
				}
				//cudaDeviceSynchronize();


				MPI_Send(&nval, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
				MPI_Send(*CPUresults, m*jump, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			}
		}
		else{
			cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
			while(true){
				MPI_Recv(&nval, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
				if(stat.MPI_TAG==1){break;}

				if(nval+jump-1>n){jump=n-nval+1;}
				singlen=jump/nostreams;
				ns[nostreams-1]=jump-(nostreams-1)*singlen;
				dim3 dimGrid_main(((singlen)/dimBlock.x)+(!((singlen)%dimBlock.x)?0:1));
				dim3 dimGrid_last(((ns[nostreams-1])/dimBlock.x)+(!((ns[nostreams-1])%dimBlock.x)?0:1));
				for(i=0;i<nostreams-1;i++){
					ns[i]=singlen;
					shared_dynamic_streams_mpi<<<dimGrid_main, dimBlock, 0, streams[i]>>>(results, nval, m, a, division, maxIterations, singlen, i*singlen);
				}
				shared_dynamic_streams_mpi<<<dimGrid_last, dimBlock, 0, streams[nostreams-1]>>>(results, nval, m, a, division, maxIterations, jump-(nostreams-1)*singlen, (nostreams-1)*singlen);

				
				//cudaDeviceSynchronize();
				for(i=0;i<nostreams;i++){
					cudaMemcpyAsync(&((*CPUresults)[(i*singlen)*m]), &(results[i*singlen*m]), sizeof(gputype)*ns[i]*m, cudaMemcpyDeviceToHost, streams[i]);
				}
				//cudaDeviceSynchronize();


				MPI_Send(&nval, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
				MPI_Send(*CPUresults, m*jump, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			}
		}
		for(i=0;i<nostreams;i++){
			cudaStreamDestroy(streams[i]);
		}
		cudaFree(results);
		
	}
	MPI_Finalize();

	cudaEventRecord(finish, 0);

	cudaEventSynchronize(start);
	cudaEventSynchronize(finish);
	
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, finish);
	return elapsedTime/1000.0;
}
