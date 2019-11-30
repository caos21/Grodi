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
#include "charging.h"

int fcharging(double t, double *n, double *dndt, void *data) {
  struct collision_data *mcd;
  mcd = (struct collision_data *)data;

  short l = mcd->l;
  short nchrgs = mcd->nchrgs;

  short q = 0;
  for (q = 0; q < nchrgs - 1; ++q) {
    dndt[q] =
        ((mcd->ifreq[l][q - 1] + mcd->with_tunnel * mcd->tfreq[l][q - 1]) *
             n[q - 1] +
         mcd->efreq[l][q + 1] * n[q + 1] -
         n[q] * ((mcd->ifreq[l][q] + mcd->with_tunnel * mcd->tfreq[l][q]) +
                 mcd->efreq[l][q]));
  }
  q = 0;
  dndt[q] = (mcd->efreq[l][q + 1] * n[q + 1] -
             n[q] * ((mcd->ifreq[l][q] + mcd->with_tunnel * mcd->tfreq[l][q])));

  q = nchrgs - 1;
  dndt[q] = ((mcd->ifreq[l][q - 1] + mcd->with_tunnel * mcd->tfreq[l][q - 1]) *
                 n[q - 1] -
             n[q] * mcd->efreq[l][q]);

  return 0;
}

int charging_step(double time, double delta_t, void *pdata, void *cdata) {
  struct plasma_data *mpd = pdata;

  struct collision_data *mcd = cdata;

  short nchrgs = mcd->nchrgs;
  short nvols = mpd->nvols;

#pragma omp parallel for schedule(static) default(none) shared(nvols, nchrgs, mcd, mpd, time, delta_t, stderr)
  for (short l = 0; l < nvols; ++l) {
    struct collision_data local_mcd;

    local_mcd.l = l;
    local_mcd.nchrgs = nchrgs;
    local_mcd.with_tunnel = mcd->with_tunnel;
    local_mcd.efreq = mcd->efreq;
    local_mcd.ifreq = mcd->ifreq;
    local_mcd.tfreq = mcd->tfreq;

    double tin = time;
    double tout = time + delta_t;

    double atol[nchrgs], rtol[nchrgs], nqdens[nchrgs];

    for (short q = 0; q < nchrgs; ++q) {
      nqdens[q] = mpd->pdens[l][q];
      atol[q] = 1.0e-3;
      rtol[q] = 1.0e-3;
    }

    struct lsoda_opt_t opt = {0};
    opt.ixpr = 0;            // additional printing
    opt.rtol = rtol;         // relative tolerance
    opt.atol = atol;         // absolute tolerance
    opt.itask = 1;           // normal integration
    opt.mxstep = 100000000;  // max steps

    struct lsoda_context_t ctx = {
        .function = fcharging,
        .data = &local_mcd,
        .neq = (int)(nchrgs),
        .state = 1,
    };
    lsoda_prepare(&ctx, &opt);
    // integrate
    lsoda(&ctx, nqdens, &tin, tout);
    if (ctx.state <= 0) {
      fprintf(stderr, "\n------------------------------------");
      fprintf(stderr, "\n[ee] LSODA error in charging");
      fprintf(stderr, "\n[ee] error istate = %d", ctx.state);
      fprintf(stderr, "\n[ii] volume section = %d ", l);
      fprintf(stderr, "\n[ii] min dtq = %f ", opt.hmin);
      fprintf(stderr, "\n[ii] h0 = %f ", opt.h0);
      fprintf(stderr, "\n[ii] time = %f", time);
      fprintf(stderr, "\n[ii] delta_t = %f", delta_t);
      fprintf(stderr, "\n------------------------------------\n");
      exit(0);
    }

    lsoda_free(&ctx);

    for (short q = 0; q < nchrgs; ++q) {
      mpd->ndens[l][q] = (nqdens[q] > 0 ? nqdens[q] : 0.0);
      mpd->qrate2d[l][q] = (mpd->ndens[l][q] - mpd->pdens[l][q]) / delta_t;
    }
  }

  return 0;
}
