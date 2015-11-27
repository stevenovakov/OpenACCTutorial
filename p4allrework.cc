/*
# p4allrework.cc
#     part of OpenACC tutorial to accelerate Jacobi Iteration
#     Copyright (C) 2015 Steve Novakov

#     This program is free software; you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation; either version 2 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License along
#     with this program; if not, write to the Free Software Foundation, Inc.,
#     51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/
// This particular file is a straight c++ port of
//
// http://devblogs.nvidia.com/parallelforall/openacc-example-part-1/ (and part 2)
//
// for testing and comparison purposes
//

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <cstring>
#include <stdio.h>

#ifdef OMP
  #include "omp.h"
#else
  #include <time.h>
#endif

#ifdef OACC
  #include "openacc.h"
#endif

uint32_t n = 512;
uint32_t batches = 100;

void CLArgs(int argc, char * argv[]);

int main(int argc, char** argv)
{
  CLArgs(argc, argv);

  uint32_t iter_max = 1000000;

  const float pi  = 2.0f * std::asin(1.0f);
  const float tol = 1.0e-6f;
  float error     = 1.0f;
  float expfpi = expf(-1*pi);

  float * A = new float[n*n];
  float * Anew = new float[n*n];
  float * y0 = new float[n];

  std::memset(A, 0, n * n * sizeof(float));

  // set boundary conditions
  for (uint32_t i = 0; i < n; i++)
  {
      A[0*n+i]   = 0.f;
      A[(n-1)*n+i] = 0.f;
  }

  for (uint32_t j = 0; j < n; j++)
  {
      y0[j] = sinf(pi * j / (n-1));
      A[j*n+0] = y0[j];
      A[j*n+n-1] = y0[j]*expfpi;
  }

#ifdef OACC
  acc_init(acc_device_nvidia);
#endif

  printf("Jacobi relaxation Calculation: %d x %d mesh\n", n, n);

#ifdef OMP
  double start, end;
  start = omp_get_wtime();
#else
  clock_t itertime;
  itertime = clock();
#endif

  // start iteration

  uint32_t iter = 0;

#ifdef OMP
  #pragma omp parallel for shared(Anew)
#endif
    for (uint32_t i = 1; i < n; i++)
    {
       Anew[0*n+i]   = 0.f;
       Anew[(n-1)*n+i] = 0.f;
    }

#ifdef OMP
  #pragma omp parallel for shared(Anew, expfpi)
#endif
    for (uint32_t j = 1; j < n; j++)
    {
        Anew[j*n+0]   = y0[j];
        Anew[j*n+n-1] = y0[j]*expfpi;
    }

#ifdef OACC
  #pragma acc data copy(A), create(Anew)
#endif
  while ( error > tol && iter < iter_max )
  {
    error = 0.f;

#ifdef OMP
  #pragma omp parallel for shared(n, Anew, A) reduction(max: error)
#elif OACC
  #pragma acc kernels
  {
    #pragma acc loop gang(32), vector(16)
#endif
    for( uint32_t j = 1; j < n-1; j++)
    {
#ifdef OACC
      #pragma acc loop gang(16), vector(32)
#endif
      for( uint32_t i = 1; i < n-1; i++ )
      {
        float diff;

        Anew[j*n+i] = 0.25f * ( A[j*n+i+1] + A[j*n+i-1]
                             + A[(j-1)*n+i] + A[(j+1)*n+i]);

        diff = std::abs(Anew[j*n+i]-A[j*n+i]);

        if (error < diff)
            error = diff;
      }
    }

#ifdef OMP
  #pragma omp parallel for shared(n, Anew, A)
#elif OACC
  }
  #pragma acc kernels
  {
    #pragma acc loop
#endif
    for( uint32_t j = 1; j < n-1; j++)
    {
#ifdef OACC
      #pragma acc loop gang(16), vector(32)
#endif
      for( uint32_t i = 1; i < n-1; i++ )
      {
        A[j*n+i] = Anew[j*n+i];
      }
    }

    if(iter % batches == 0)
      printf("%5d, %0.6f\n", iter, error);

    iter++;
#ifdef OACC
  }
#endif

  }

#ifdef OMP
  end = omp_get_wtime();

  printf("Iteration Complete. Total Iterations: %d, Time Elapsed: %f (s)\n",
    iter, end - start);
#else
  itertime = clock() - itertime;

  printf("Iteration Complete. Total Iterations: %d, Time Elapsed: %f (s)\n",
    iter -1, static_cast<float>(itertime) / CLOCKS_PER_SEC);
#endif

   // write potential to csv file
  FILE *pfile;
  uint32_t ind;

  puts("Writing to \"output csvs\" ...\n");

#ifdef OMP
  pfile = fopen("a_omp.csv", "w");
#elif OACC
  pfile = fopen("a_oacc.csv", "w");
#else
  pfile = fopen("a.csv", "w");
#endif

  for (uint32_t i = 0; i < n; i++)
  {
    for (uint32_t j = 0; j < n; j++)
    {
      ind = i + n * j;

      fprintf(pfile, "%f", Anew[ind]);

      if(j < n-1)
        fprintf(pfile, ",");
    }

    if (i < n-1)
       fprintf(pfile, "\n");
  }

  fclose(pfile);

  delete[] A;
  delete[] Anew;
  delete[] y0;
}

void CLArgs(int argc, char * argv[])
{
  std::vector<std::string> args(argv, argv+argc);

  for (uint32_t i = 0; i < args.size(); i++)
  {
    if (args.at(i).find("-n") == 0)
      n = std::stoul(args.at(i).substr(args.at(i).find('=')+1));
    if (args.at(i).find("-b") == 0)
      batches = std::stoul(args.at(i).substr(args.at(i).find('=')+1));
  }
}
