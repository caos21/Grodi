/*
 * Copyright 2019 <Benjamin Santos> <caos21@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#ifndef CHARGING_H
#define CHARGING_H

#include <math.h>
#include <omp.h>
#include <stdio.h>

#include "common.h"
#include "lsoda.h"

struct collision_data {
  short l;
  short nchrgs;
  double with_tunnel;
  double** efreq;
  double** ifreq;
  double** tfreq;
};

struct plasma_data {
  short nvols;
  short nchrgs;
  double* pdensity;
  double** pdens;
  double** ndens;
  double** qrate2d;
};

int fcharging(double t, double* n, double* dndt, void* data);

int charging_step(double time, double delta_t, void* pdata, void* cdata);

#endif
