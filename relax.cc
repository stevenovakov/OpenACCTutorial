/*
# jacobi.cc
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

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <complex>
#include <time.h>

uint32_t n = 1000;

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
        if ( std::cos(std::atan2(x, y)) > 0)
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
  uint32_t max_iter = 100000;
  float error = 1000.0f;
  float epsilon = 1.0e-5f;

  clock_t itertime;
  itertime = clock();

  while (error > epsilon && iter < max_iter)
  {
    error = 0.f;
    sel = iter % 2;

    for (uint32_t i = 1; i < n-1; i++)
    {
      for (uint32_t j = 1; j < n-1; j++)
      {
        uint32_t ibase = i + n * j;

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

        error = std::max(error, std::abs(
          phi[iwrite + ibase] - phi[iread + ibase]));
      }
    }

    if (iter % 100 == 0)
      printf("%d, %3.10f\n", iter, error);

    iter++;
  }

  itertime = clock() - itertime;

  printf("Iteration Complete. Time Elapsed: %f (s)\n",
    static_cast<float>(itertime) / CLOCKS_PER_SEC);

  // write potential to csv file
  // use imshow in python to display (square domain, who cares)

  FILE * file;

  puts("Writing to \"output.csv\" ...\n");

  file = fopen("output.csv", "w");

  uint32_t readfrom = (1-sel) * n * n;
  uint32_t iread;

  for (uint32_t i = 0; i < n; i++)
  {
    for (uint32_t j = 0; j < n; j++)
    {
      iread = readfrom + i + n * j;

      fprintf(file, "%f", phi[iread]);

      if(j < n-1)
        fprintf(file, ",");
    }
    if (i < n-1)
      fprintf(file, "\n");
  }

  fclose(file);


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
      n=std::stoul(args.at(i).substr(args.at(i).find('=')+1));
  }
}
