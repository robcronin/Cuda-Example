#ifdef __cplusplus
extern "C" {
	int	getnocards();
	float	double_allocGPU(double **radGPU, int n, int m);
	float	double_copyfromGPU(double **CPUresults, double **resultsGPU, int n, int m);
	void	double_freeGPU(double **radGPU);


	float	double_gpu_standard(double *results, int n, int m, double a, double b, int maxIterations, bool simple);
	float	double_gpu_streams(double **CPUresults, int n, int m, double a, double b, int maxIterations, bool simple, int nostreams);
	float	double_gpu_mpi(double **CPUresults, int n, int m, double a, double b, int maxIterations, int *CPUrank, int jump, bool simple);
	float	double_gpu_dynamic_mpi(double **CPUresults, int n, int m, double a, double b, int maxIterations, int *CPUrank, int jump, bool simple);
	float	double_gpu_dynamic(double *results, int n, int m, double a, double b, int maxIterations, bool simple);
	float	double_gpu_streams_dynamic(double **CPUresults, int n, int m, double a, double b, int maxIterations, bool simple, int nostreams);
	float	double_gpu_streams_mpi(double **CPUresults, int n, int m, double a, double b, int maxIterations, int *CPUrank, int jump, bool simple, int nostreams);
	float	double_gpu_dynamic_streams_mpi(double **CPUresults, int n, int m, double a, double b, int maxIterations, int *CPUrank, int jump, bool simple, int nostreams);
}
#endif
