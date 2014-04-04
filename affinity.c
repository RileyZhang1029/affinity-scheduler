#include <stdio.h>
#include <math.h>


#define N 729
#define REPS 100
#include <omp.h> 

#define MINIMUM_CHUNK_SIZE 16
#define true 1
#define false 0

double a[N][N], b[N][N], c[N];
int jmax[N];  


void init1(void);
void init2(void);
void runloop(int); 
void loop1chunk(int, int);
void loop2chunk(int, int);
void valid1(void);
void valid2(void);
int findThreadToStealFrom(int*, int*, int, int, float, int*, int*);
void printArray(int*, int);

int main(int argc, char *argv[]) { 

	double start1,start2,end1,end2;
	int r;

	init1(); 

	start1 = omp_get_wtime(); 

	for (r=0; r<REPS; r++){ 
		runloop(1);
	} 

	end1  = omp_get_wtime();  

	valid1(); 

	printf("Total time for %d REPS of loop 1 = %f\n",REPS, (float)(end1-start1)); 


	init2(); 

	start2 = omp_get_wtime(); 

	for (r=0; r<REPS; r++){ 
		runloop(2);
	} 

	end2  = omp_get_wtime(); 

	valid2(); 

	printf("Total time for %d REPS of loop 2 = %f\n",REPS, (float)(end2-start2));

} 

void init1(void){
	int i,j; 

	for (i=0; i<N; i++){ 
		for (j=0; j<N; j++){ 
			a[i][j] = 0.0; 
			b[i][j] = 3.142*(i+j); 
		}
	}

}

void init2(void){ 
	int i,j, expr; 

	for (i=0; i<N; i++){ 
		expr =  i%( 3*(i/30) + 1); 
		if ( expr == 0) { 
			jmax[i] = N;
		}
		else {
			jmax[i] = 1; 
		}
		c[i] = 0.0;
	}

	for (i=0; i<N; i++){ 
		for (j=0; j<N; j++){ 
			b[i][j] = (double) (i*j+1) / (double) (N*N); 
		}
	}
 
} 

/**
 * The idea is to implement a general work-stealing algorithm using critical sections (alternatives are discussed in the report)/
 * However, rather than computing iterations owned by the current thread, we're going to steal from ourself. Once own own iterations have
 * been completed, we will start stealing from other threads. The preference will be given to threads with higher IDs due to the way the
 * work is distributed (this, again, is explained in the report).
 */
void runloop(int loopid)
{
	int thread_count = omp_get_max_threads();							// the number of threads in the system.
																		// we don't know how many exist yet, so use this. alternatively, we
																		// could have used getenv() from <stdlib.h> to get the env variable, but
																		// this seems cleaner. it should always work within our setup as well.
	
	int n_over_p = (int) ceil((double) N / (double) thread_count);		// what it says on the tin
	
	float one_over_p = 1.0 / thread_count;								// one over p
	
	int lower_bounds[thread_count];										// stores the lower bound of the array not already computed.
	
	int upper_bounds[thread_count];										// stores the upper bound of the array not already computed.
																		// upper_bounds[i] - lower_bounds[i] = remaining iterations

	#pragma omp parallel default(none)  \
						 shared(thread_count, loopid, lower_bounds, upper_bounds, n_over_p, one_over_p)
	{
		int thread_id	= omp_get_thread_num(),
			thread_low	= thread_id * n_over_p,
			thread_high = ((thread_id + 1) * n_over_p) > N ? N : (thread_id + 1) * n_over_p; // in case n mod p != 0

		lower_bounds[thread_id] = thread_low;
		upper_bounds[thread_id] = thread_high;

		// We need to ensure that the last iteration does not compute twice. Although this could be done with an if statement below the
		// switch, I feel that it should be achievable in a more succict method. Thus, in the first iteration we will perform no work
		// which allows findThreadToSteaFrom() to perform it's computation and update current_low and current_high. Hence, the second
		// iteration is the first one that will perform any work.
		int current_low	  = 0,
			current_high  = 0,
			stealing_from = 0;
		
		while(stealing_from != -1)
		{
			switch(loopid)
			{
				case 1: loop1chunk(current_low, current_high); break;
				case 2: loop2chunk(current_low, current_high); break;
			}

			// Find the next current_low and current_high. Notice the use of pointers to these values as replacements for C#/C++-style out params.
			// This would go nicely in the while loop condition, but unfortunately we need the #pragma block.
			#pragma omp critical
			{
				stealing_from = findThreadToStealFrom(lower_bounds, upper_bounds, thread_count, thread_id, one_over_p, &current_low, &current_high);
			}
		}
	}
}

/**
 * this method computes where the thread should steal from (locally or remotely) as well as the number of iterations to process.
 * it updates the appropriate arrays too, and returns via pointers - which is a kludge to get around the lack of out params in C
 */
int findThreadToStealFrom(int* lower_bounds, int* upper_bounds, int size, int thread, float one_over_p, int* current_low, int *current_high)
{
	int position;

	// In the first instance, we want to 'steal' from our local set of iterations. This is the section of code that handles this.
	if (upper_bounds[thread] - lower_bounds[thread] > 0)
	{
		position = thread;
	}
	else
	{
		// No iterations left in local set. Find lowest ID with some remaining
		int index = -1,
			value = 0;

		for (int i = 0; i < size; i++)
		{
			int diff = upper_bounds[i] - lower_bounds[i];
			if (diff > value)
			{
				index = i;
				value = diff;
			}
		}

		position = index;
	}

	// No other threads have any left either.
	if (position == -1) return -1;

	// Perform the real work
	int remaining_iterations = upper_bounds[position] - lower_bounds[position];
	int chunk_size = (int) ceil(one_over_p * remaining_iterations);

	// Check for minimum chunk size.
	if (MINIMUM_CHUNK_SIZE)
	{
		if (chunk_size < MINIMUM_CHUNK_SIZE)
		{
			if (remaining_iterations >= MINIMUM_CHUNK_SIZE)
			{
				chunk_size = MINIMUM_CHUNK_SIZE;
			}
			else
			{
				// This seems like a reasonable thing to do if there are less than MIN_CHUNK_SIZE
				// iterations left
				chunk_size = remaining_iterations;
			}
		}
	}

	// Update the "out" params
	*current_low = lower_bounds[position];
	*current_high = lower_bounds[position] + chunk_size;
	lower_bounds[position] = lower_bounds[position] + chunk_size;

	return position;
}

void printArray(int* array, int size)
{
	for (int i = 0; i < size; i++)
	{
		printf("%d: %d, ", i, array[i]);
	}

	printf("\n");
}

void loop1chunk(int lo, int hi) { 
	int i,j; 
	
	for (i=lo; i<hi; i++){ 
		for (j=N-1; j>i; j--){
			a[i][j] += cos(b[i][j]);
		} 
	}

} 



void loop2chunk(int lo, int hi) {
	int i,j,k; 
	double rN2; 

	rN2 = 1.0 / (double) (N*N);  

	for (i=lo; i<hi; i++){ 
		for (j=0; j < jmax[i]; j++){
			for (k=0; k<j; k++){ 
	c[i] += (k+1) * log (b[i][j]) * rN2;
			} 
		}
	}

}

void valid1(void) { 
	int i,j; 
	double suma; 
	
	suma= 0.0; 
	for (i=0; i<N; i++){ 
		for (j=0; j<N; j++){ 
			suma += a[i][j];
		}
	}
	printf("Loop 1 check: Sum of a is %lf\n", suma);

} 


void valid2(void) { 
	int i; 
	double sumc; 
	
	sumc= 0.0; 
	for (i=0; i<N; i++){ 
		sumc += c[i];
	}
	printf("Loop 2 check: Sum of c is %f\n", sumc);
} 