# Makefile
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

GPP = g++
PGI = pgc++

GPPFLAGS = -Wall -ansi -pedantic -fPIC -std=c++11
DGPPFLAGS = -g -Wall -ansi -pedantic -fPIC -std=c++11
PGIFLAGS = -acc -ta=tesla:managed -Minfo=accel -std=c++11 -D OACC
DPGIFLAGS = -g -acc -ta=tesla:managed -Minfo=accel -std=c++11 -D OACC

OMPFLAGS = -fopenmp -lpthread -D OPENMP

TARGETS = relax relax_omp relax_oacc

all: $(TARGETS)

debug: GPPFLAGS=$(DGPPFLAGS)
debug: DPGIFLAGS=$(PGIFLAGS)
debug: $(TARGETS)

relax: relax.cc
	$(GPP) $(GPPFLAGS) -o $@ $<

relax_omp: relax.cc
	$(GPP) $(GPPFLAGS) -o $@ $< $(OMPFLAGS)

relax_oacc: relax.cc
	$(PGI) $(PGIFLAGS) -o $@ $<

clean:
	$(RM) $(TARGETS)
