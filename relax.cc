/*
# relax.cc
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

// Influenced by
//
// http://devblogs.nvidia.com/parallelforall/openacc-example-part-1/


//
// TODO
//    install CUDA 7.5
//    install OpenACC toolkit
//    create makefile with cpp, openmp, openacc options
//    create installation instructions for openacc
//

#include <iostream>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <complex>

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

int main(int argc, char * argv[])
{
  CLArgs(argc, argv);

  float nlim = -1.2;
  float plim = 1.2;

  float h = (plim - nlim) / static_cast<float>(n-1);

  float *xx, *yy, *phi;

  xx = new float[n*n];
  yy = new float[n*n];
  phi = new float[2*n*n];

  // initialize domains
  for (uint32_t i = 0; i < n; i++)
  {
    for (uint32_t j = 0; j < n; j++)
    {
      uint32_t index = i + n * j;

      xx[index] = nlim + (i * h);
      yy[index] = nlim + (j * h);
    }
  }

  // initialize potentials
#ifdef OMP
  #pragma omp parallel for shared(n, xx, yy, phi)
#endif
  for (uint32_t i = 0; i < n; i++)
  {
    for (uint32_t j = 0; j < n; j++)
    {
      uint32_t xy = i + n * j;
      uint32_t xy2 = xy + n*n;

      float x = xx[xy];
      float y = yy[xy];

      if ( x*x + y*y >= 1.0)
      {
        if ( std::cos(std::atan2(y, x)) > 0)
        {
          phi[xy] = 1.0;
          phi[xy2] = 1.0;
        }
        else
        {
          phi[xy] = -1.0;
          phi[xy2] = -1.0;
        }
      }
      else
      {
        phi[xy] = 0.0;
        phi[xy2] = 0.0;
      }
    }
  }

  std::puts("Initialization Complete. Starting iteration...\n");
  //
  // Iteration Start
  //

  uint32_t iter = 0;
  uint32_t sel = 0;
  uint32_t max_iter = 1000000;
  float error = 1000.0f;
  float epsilon = 1.0e-6f;

#ifdef OMP
  double start, end;
  start = omp_get_wtime();
#else
  clock_t itertime;
  itertime = clock();
#endif

  //
  // TODO
  // pretty naive termination condition, not guaranteed to converge for all
  // grid sizes with fixed error, look at using the L-infinity norm of the
  // Laplacian as a termination condition
  //
  while (error > epsilon && iter < max_iter)
  {
    error = 0.0f;
    sel = iter % 2;

#ifdef OMP
    #pragma omp parallel for shared(sel, n, xx, yy, phi) reduction(max:error)
#elif OACC
    #pragma acc kernels
    {
      #pragma acc loop independent reduction(max:error)
#endif
      for (uint32_t i = 1; i < n-1; i++)
      {
#ifdef OACC
        #pragma acc loop independent
#endif
        for (uint32_t j = 1; j < n-1; j++)
        {
          uint32_t ibase = i + n * j;

          float x = xx[ibase];
          float y = yy[ibase];
          float diff;

          if ( x*x + y*y > 1.0)
            continue;

          uint32_t iread = sel * n * n;
          uint32_t iwrite = (1-sel) * n * n;

          uint32_t i1 = (i+1) + n * j;
          uint32_t i2 = (i-1) + n * j;
          uint32_t i3 = i + n * (j+1);
          uint32_t i4 = i + n * (j-1);
          uint32_t i5 = (i+1) + n * (j+1);
          uint32_t i6 = (i-1) + n * (j+1);
          uint32_t i7 = (i+1) + n * (j-1);
          uint32_t i8 = (i-1) + n * (j-1);

          float pc = 0.25 * (phi[iread + i1] + phi[iread + i2] +
            phi[iread + i3] + phi[iread + i4]);
          float ps = 0.25 * (phi[iread + i5] + phi[iread + i6] +
            phi[iread + i7] + phi[iread + i8]);

          phi[iwrite + ibase] = 0.8 * pc + 0.2 * ps;

          diff = std::abs(phi[iwrite + ibase] - phi[iread + ibase]);

          if (error < diff)
            error = diff;
        }
      }
#ifdef OACC
    }
#endif

    if (iter % batches == 0)
      printf("%d, %3.10f\n", iter, error);

    iter++;
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
  FILE *pfile, *xfile, *yfile;

  puts("Writing to \"output csvs\" ...\n");

  xfile = fopen("xx.csv", "w");
  yfile = fopen("yy.csv", "w");

#ifdef OMP
  pfile = fopen("phi_omp.csv", "w");
#elif OACC
  pfile = fopen("phi_oacc.csv", "w");
#else
  pfile = fopen("phi.csv", "w");
#endif

  uint32_t readfrom = (1-sel) * n * n;
  uint32_t ind, iread;

  for (uint32_t i = 0; i < n; i++)
  {
    for (uint32_t j = 0; j < n; j++)
    {
      ind = i + n * j;
      iread = readfrom + ind;

      fprintf(pfile, "%f", phi[iread]);
      fprintf(xfile, "%f", xx[ind]);
      fprintf(yfile, "%f", yy[ind]);

      if(j < n-1)
      {
        fprintf(pfile, ",");
        fprintf(xfile, ",");
        fprintf(yfile, ",");
      }
    }
    if (i < n-1)
    {
      fprintf(pfile, "\n");
      fprintf(xfile, "\n");
      fprintf(yfile, "\n");
    }
  }

  fclose(pfile);
  fclose(xfile);
  fclose(yfile);

  delete[] xx;
  delete[] yy;
  delete[] phi;

  return 0;
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
