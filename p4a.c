/*
 *  Copyright 2015 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
/*
 *
 * Modified 2015-11-23 Steve Novakov
 * To work with system environment & NVVP.
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <openacc.h>
#include "time.h"

int n = 256;
int batches = 100;

void CLArgs(int argc, char * argv[]);

int main(int argc, char * argv[])
{
    CLArgs(argc, argv);

    int iter_max = 1000000;

    const float pi  = 2.0f * asinf(1.0f);
    const float tol = 1.0e-5f;
    float error     = 1.0f;
    float expfpi = expf(-pi);

    float A[n][n];
    float Anew[n][n];
    float y0[n];

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, n);

    memset(A, 0, n * n * sizeof(float));

    // set boundary conditions
    for (int i = 0; i < n; i++)
    {
        A[0][i]   = 0.f;
        A[n-1][i] = 0.f;
    }

    for (int j = 0; j < n; j++)
    {
        y0[j] = sinf(pi * j / (n-1));
        A[j][0] = y0[j];
        A[j][n-1] = y0[j]*expfpi;
    }

#if _OPENACC
    acc_init(acc_device_nvidia);
#endif

    clock_t itertime;
    itertime = clock();

    int iter = 0;

    for (int i = 1; i < n; i++)
    {
       Anew[0][i]   = 0.f;
       Anew[n-1][i] = 0.f;
    }

    for (int j = 1; j < n; j++)
    {
        Anew[j][0]   = y0[j];
        Anew[j][n-1] = y0[j]*expfpi;
    }

#pragma acc data copy(A), create(Anew)
    while ( error > tol && iter < iter_max )
    {
        error = 0.f;

#pragma acc kernels loop gang(32), vector(16)
        for( int j = 1; j < n-1; j++)
        {
#pragma acc loop gang(16), vector(32)
            for( int i = 1; i < n-1; i++ )
            {
                Anew[j][i] = 0.25f * ( A[j][i+1] + A[j][i-1]
                                     + A[j-1][i] + A[j+1][i]);
                error = fmaxf( error, fabsf(Anew[j][i]-A[j][i]));
            }
        }

#pragma acc kernels loop
        for( int j = 1; j < n-1; j++)
        {
#pragma acc loop gang(16), vector(32)
            for( int i = 1; i < n-1; i++ )
            {
                A[j][i] = Anew[j][i];
            }
        }

        if(iter % batches == 0) printf("%5d, %0.6f\n", iter, error);

        iter++;
    }

    itertime = clock() - itertime;

    printf("Iteration Complete. Total Iterations: %d, Time Elapsed: %f (s)\n",
        iter -1, (double) itertime / CLOCKS_PER_SEC);

    FILE *pfile;
    int ind;

    pfile = fopen("a_oacc.csv", "w");

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
          ind = i + n * j;

          fprintf(pfile, "%f", A[ind]);

          if(j < n-1)
            fprintf(pfile, ",");
        }

        if (i < n-1)
           fprintf(pfile, "\n");
    }

    fclose(pfile);
}

void CLArgs(int argc, char * argv[])
{
    printf("argc: %i\n", argc);

  for (int i = 0; i < argc; i++)
  {
    printf("%d", atoi(argv[i]));
    if (i == 1)
        n = atoi(argv[i]);
    if (i == 2)
        batches = atoi(argv[i]);
  }
}
