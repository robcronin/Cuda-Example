#ifdef __cplusplus
extern "C" {
	int	getnocards();
	float	allocGPU(float **radGPU, int n, int m);
	float	copyfromGPU(float **CPUresults, float **resultsGPU, int n, int m);
	void	freeGPU(float **radGPU);


	float	gpu_standard(float *results, int n, int m, float a, float b, int maxIterations, bool simple);
	float	gpu_streams(float **CPUresults, int n, int m, float a, float b, int maxIterations, bool simple, int nostreams);
	float	gpu_mpi(float **CPUresults, int n, int m, float a, float b, int maxIterations, int *CPUrank, int jump, bool simple);
	float	gpu_dynamic_mpi(float **CPUresults, int n, int m, float a, float b, int maxIterations, int *CPUrank, int jump, bool simple);
	float	gpu_dynamic(float *results, int n, int m, float a, float b, int maxIterations, bool simple);
	float	gpu_streams_dynamic(float **CPUresults, int n, int m, float a, float b, int maxIterations, bool simple, int nostreams);
	float	gpu_streams_mpi(float **CPUresults, int n, int m, float a, float b, int maxIterations, int *CPUrank, int jump, bool simple, int nostreams);
	float	gpu_dynamic_streams_mpi(float **CPUresults, int n, int m, float a, float b, int maxIterations, int *CPUrank, int jump, bool simple, int nostreams);
}
#endif
