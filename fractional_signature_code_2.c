/*
fractional_signature_code_2.c
Copyright (C) 2025  Rubén

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/



/* The goal of this code is to show in a self-contained and isolated way how we have computed the discrete fractional signature of a given piecewise linear path*/

/* This program uses the GNU Scientific library, so the flags -lgsl -lgslcblas must be added when compiling the code. */

/* To make the code easier to understand, we will assume that all our allocations of memory work, so there is no need to check the result after each allocation. */

/* Finally, note that this program computes the corresponding discrete fractional signature up to order 7, which is the maximum we used in our machine learning application. */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<gsl/gsl_sf.h>
#include<time.h>

int main(void) {
   int i, j, k, h, a, a1, d, ii, jj, kk, uu, vv, hh, ll, m, n = 784, comp = 3, ind, cum = 0;
   int **mat;
   double alpha = 0.9;
   double **vec, **aux1, **aux2, **aux3, **aux4, **aux5, **aux6, **aux7;
   double ***mati, ****sig;

/* As an example, here we prepare a random vector that defines a piecewise linear path, whose discrete fractional signature we will compute. */
/*------------------------------------------------------------*/
   
   
   vec = (double **) malloc(comp*sizeof(double *));

   for (i = 0; i < comp; i++) {
      vec[i] = (double *) malloc(n*sizeof(double));
   }

   for (i = 0; i < n; i++) {
      vec[0][i] = 255.*rand()/RAND_MAX;
      vec[1][i] = 255.*rand()/RAND_MAX;
      vec[2][i] = 255.*rand()/RAND_MAX;
   }

/*-----------------------------------------------------------*/

/* From here, we start to do some preparatory work that will later be used to guide the computations that produce the desired discrete fractional signature.*/
   
   m = floor(log(n)/log(2)) + 2;

   mat = (int **) malloc(m*sizeof(int *));

   j = 2;
   for (i = 0; i < m; i++) {
      mat[i] = (int *) malloc (j*sizeof(int));

      if (i == 0) {
	 mat[i][0] = 0;
	 mat[i][1] = n-1;
      } else {
	 for (k = 0; k < j; k++) {
	    if (k%2 == 0) {
	      mat[i][k] = mat[i-1][k/2];
	    } else {
	       mat[i][k] = (mat[i-1][k/2] + mat[i-1][k/2 + 1])/2 + (mat[i-1][k/2] + mat[i-1][k/2 + 1])%2;
	    }
	 }
      }

      j = 2*j -1;
   }


   j = 2;
   mati = (double ***) malloc (m*sizeof(double **));
   sig = (double ****) malloc(m*sizeof(double ***));
   for (i = 0; i < m; i++) {
      mati[i] = (double **) malloc(j*sizeof(double *));
      sig[i] = (double ***) malloc(j*sizeof(double *));
      for (k = 1; k < j; k++) {
	 mati[i][k] = (double *) malloc((i+1)*sizeof(double));
	 sig[i][k] = (double **) malloc((i+1)*sizeof(double *));
	 for (h = 0; h < i; h++) {
	    mati[i][k][h] = mati[i-1][k/2 + k%2][h];
	 }
	 if (i == 0) {
	    mati[i][k][i] = n-1;
	 } else {
	    if (k%2 == 1) {
	      mati[i][k][i] = (mat[i][k] + mat[i][k+1])/2.;
	      if (mati[i][k][i] == (double) mat[i][k]) {
		 mati[i][k][i] = 0.;
	      }
	    } else {
	      mati[i][k][i] = 0.;
	    }
	 }
      }

      j = 2*j -1;
   }

   
   for (i = 1; i <= 7; i++) {
      ind = 1;
      for (j = 0; j < i; j++) {
	 ind *= comp;
      }
      cum += ind;
   }
  
   aux1 = (double **) malloc((n-1)*sizeof(double *));
   aux2 = (double **) malloc((n-1)*sizeof(double *));
   aux3 = (double **) malloc((n-1)*sizeof(double *));
   aux4 = (double **) malloc((n-1)*sizeof(double *));
   aux5 = (double **) malloc((n-1)*sizeof(double *));
   aux6 = (double **) malloc((n-1)*sizeof(double *));
   aux7 = (double **) malloc((n-1)*sizeof(double *));

   for (a = 0; a < n-1; a++) {
         aux1[a] = (double *) malloc ((2*n-1)*sizeof(double));
         aux2[a] = (double *) malloc ((2*n-1)*sizeof(double));
         aux3[a] = (double *) malloc ((2*n-1)*sizeof(double));
         aux4[a] = (double *) malloc ((2*n-1)*sizeof(double));
	 aux5[a] = (double *) malloc ((2*n-1)*sizeof(double));
	 aux6[a] = (double *) malloc ((2*n-1)*sizeof(double));
	 aux7[a] = (double *) malloc ((2*n-1)*sizeof(double));

      for (d = 2*a + 2; d < 2*n-1;d++) {
         aux1[a][d] = pow(d/2. - a, alpha)*gsl_sf_beta_inc(1, alpha, 1./(d/2. - a))*gsl_sf_beta(1, alpha)/gsl_sf_gamma(alpha);
         aux2[a][d] = (pow(d/2.-a, alpha*2)*gsl_sf_beta_inc(alpha + 1, alpha, 1./(d/2. - a))*gsl_sf_beta(alpha + 1, alpha))/(gsl_sf_gamma(alpha)*gsl_sf_gamma(alpha + 1));
         aux3[a][d] = (pow(d/2.-a, alpha*3)*gsl_sf_beta_inc(2*alpha + 1, alpha, 1./(d/2. - a))*gsl_sf_beta(2*alpha + 1, alpha))/(gsl_sf_gamma(alpha)*gsl_sf_gamma(2*alpha + 1));
         aux4[a][d] = (pow(d/2.-a, alpha*4)*gsl_sf_beta_inc(3*alpha + 1, alpha, 1./(d/2. - a))*gsl_sf_beta(3*alpha + 1, alpha))/(gsl_sf_gamma(alpha)*gsl_sf_gamma(3*alpha + 1));
	 aux5[a][d] = (pow(d/2.-a, alpha*5)*gsl_sf_beta_inc(4*alpha + 1, alpha, 1./(d/2. - a))*gsl_sf_beta(4*alpha + 1, alpha))/(gsl_sf_gamma(alpha)*gsl_sf_gamma(4*alpha + 1));
	 aux6[a][d] = (pow(d/2.-a, alpha*6)*gsl_sf_beta_inc(5*alpha + 1, alpha, 1./(d/2. - a))*gsl_sf_beta(5*alpha + 1, alpha))/(gsl_sf_gamma(alpha)*gsl_sf_gamma(5*alpha + 1));
	 aux7[a][d] = (pow(d/2.-a, alpha*7)*gsl_sf_beta_inc(6*alpha + 1, alpha, 1./(d/2. - a))*gsl_sf_beta(6*alpha + 1, alpha))/(gsl_sf_gamma(alpha)*gsl_sf_gamma(6*alpha + 1));
      }
   }

   j = 2;
   for (i = 0; i < m; i++) {
      for (k = 1; k < j; k++) {
	 for (h = 0; h <= i; h++) {
	    sig[i][k][h] = (double *) malloc(cum*sizeof(double));
	 }
      }

      j = 2*j - 1;
   } 


/* Now we compute the discrete fractional signature using a bottom-up and iterative approach, which is much faster than computing it using its recursive definition. */

   j = 2;
   for (i = 0; i < m-1; i++) {
      j = 2*j -1;
   }

   for (i = m-1; i >= 0; i--) {
      for (k = 1; k < j; k++) {
	 for (h = 0; h <= i; h++) {
	    if ((mati[i][k][h] != 0.) && (mat[i][k]-mat[i][k-1] != 0)) {
	       if (mat[i][k-1] + 1 ==  mat[i][k]) {
		  d = (int)(2*mati[i][k][h]);
		  a = mat[i][k-1];
		  a1 = mat[i][k];
		 
		  ind = 0;
                  for (ii = 0; ii < comp; ii++) {
                     sig[i][k][h][ind] = (vec[ii][a1] - vec[ii][a])*aux1[a][d];
                     ind += 1;
                  }
                  for (ii = 0; ii < comp; ii++) {
                     for (jj = 0; jj < comp; jj++) {
                        sig[i][k][h][ind] = (vec[ii][a1] - vec[ii][a])*(vec[jj][a1] - vec[jj][a])*aux2[a][d];
                        ind += 1;
		     }
                  }
                  for (ii = 0; ii < comp; ii++) {
                     for (jj = 0; jj < comp; jj++) {
                        for (kk = 0; kk < comp; kk++) {
                           sig[i][k][h][ind] = (vec[ii][a1] - vec[ii][a])*(vec[jj][a1] - vec[jj][a])*(vec[kk][a1] - vec[kk][a])*aux3[a][d];
                           ind += 1;
                        }
                     }
                  }
                  for (ii = 0; ii < comp; ii++) {
                     for (jj = 0; jj < comp; jj++) {
                        for (kk = 0; kk < comp; kk++) {
                           for (uu = 0; uu < comp; uu++) {
                              sig[i][k][h][ind] = (vec[ii][a1] - vec[ii][a])*(vec[jj][a1] - vec[jj][a])*(vec[kk][a1] - vec[kk][a])*(vec[uu][a1] - vec[uu][a])*aux4[a][d];
                              ind += 1;
                           }
                        }
                     }
                  }

		  for (ii = 0; ii < comp; ii++) {
                     for (jj = 0; jj < comp; jj++) {
                        for (kk = 0; kk < comp; kk++) {
                           for (uu = 0; uu < comp; uu++) {
			      for (vv = 0; vv < comp; vv++) {
                                 sig[i][k][h][ind] = (vec[ii][a1] - vec[ii][a])*(vec[jj][a1] - vec[jj][a])*(vec[kk][a1] - vec[kk][a])*(vec[uu][a1] - vec[uu][a])*(vec[vv][a1] - vec[vv][a])*aux5[a][d];
                                 ind += 1;
                              }
			   }
                        }
                     }
		  }

		  for (ii = 0; ii < comp; ii++) {
                     for (jj = 0; jj < comp; jj++) {
                        for (kk = 0; kk < comp; kk++) {
                           for (uu = 0; uu < comp; uu++) {
                              for (vv = 0; vv < comp; vv++) {
                                 for (hh = 0; hh < comp; hh++) {
				    sig[i][k][h][ind] = (vec[ii][a1] - vec[ii][a])*(vec[jj][a1] - vec[jj][a])*(vec[kk][a1] - vec[kk][a])*(vec[uu][a1] - vec[uu][a])*(vec[vv][a1] - vec[vv][a])*(vec[hh][a1] - vec[hh][a])*aux6[a][d];
                                    ind += 1;
				 }
                              }
                           }
                        }
                     }
                  }

		  for (ii = 0; ii < comp; ii++) {
                     for (jj = 0; jj < comp; jj++) {
                        for (kk = 0; kk < comp; kk++) {
                           for (uu = 0; uu < comp; uu++) {
                              for (vv = 0; vv < comp; vv++) {
                                 for (hh = 0; hh < comp; hh++) {
                                    for (ll = 0; ll < comp; ll++) {
				       sig[i][k][h][ind] = (vec[ii][a1] - vec[ii][a])*(vec[jj][a1] - vec[jj][a])*(vec[kk][a1] - vec[kk][a])*(vec[uu][a1] - vec[uu][a])*(vec[vv][a1] - vec[vv][a])*(vec[hh][a1] - vec[hh][a])*(vec[ll][a1] - vec[ll][a])*aux7[a][d];
                                       ind += 1;
				    }
                                 }
                              }
                           }
                        }
                     }
                  }


	       } else {
		  a1 = 2*k;
		  a = 2*k - 1;

		  ind = 0;
                  for (ii = 0; ii < comp; ii++) {
                     sig[i][k][h][ind] = sig[i+1][a][h][ind] + sig[i+1][a1][h][ind];
                     ind += 1;
                  }
                  for (ii = 0; ii < comp; ii++) {
                     for (jj = 0; jj < comp; jj++) {
                        sig[i][k][h][ind] = sig[i+1][a][h][ind] + sig[i+1][a][i+1][ii]*sig[i+1][a1][h][jj] + sig[i+1][a1][h][ind];
                        ind += 1;
                     }
                  }
                  for (ii = 0; ii < comp; ii++) {
                     for (jj = 0; jj < comp; jj++) {
                        for (kk = 0; kk < comp; kk++) {
                           sig[i][k][h][ind] = sig[i+1][a][h][ind] + sig[i+1][a][i+1][comp*(ii+1) + jj]*sig[i+1][a1][h][kk] + sig[i+1][a][i+1][ii]*sig[i+1][a1][h][comp*(jj+1) + kk] + sig[i+1][a1][h][ind];
                           ind += 1;
                        }
                     }
                  }
                  for (ii = 0; ii < comp; ii++) {
                     for (jj = 0; jj < comp; jj++) {
                        for (kk = 0; kk < comp; kk++) {
                           for (uu = 0; uu < comp; uu++) {
                              sig[i][k][h][ind] = sig[i+1][a][h][ind] + sig[i+1][a][i+1][comp*(comp*(ii+1) + (jj+1)) + kk]*sig[i+1][a1][h][uu] + sig[i+1][a][i+1][comp*(ii+1) + jj]*sig[i+1][a1][h][comp*(kk+1) + uu] + sig[i+1][a][i+1][ii]*sig[i+1][a1][h][comp*(comp*(jj+1) + (kk+1)) + uu] + sig[i+1][a1][h][ind];
                              ind += 1;
                           }
                        }
                     }
                  }

		  for (ii = 0; ii < comp; ii++) {
                     for (jj = 0; jj < comp; jj++) {
                        for (kk = 0; kk < comp; kk++) {
                           for (uu = 0; uu < comp; uu++) {
                              for (vv = 0; vv < comp; vv++) {
                                 sig[i][k][h][ind] = sig[i+1][a][h][ind] + sig[i+1][a][i+1][comp*(comp*(comp*(ii+1) + (jj+1)) + (kk + 1)) + uu]*sig[i+1][a1][h][vv] + sig[i+1][a][i+1][comp*(comp*(ii+1) + (jj+1)) + kk]*sig[i+1][a1][h][comp*(uu+1) + vv] + sig[i+1][a][i+1][comp*(ii+1) + jj]*sig[i+1][a1][h][comp*(comp*(kk+1) + (uu+1)) + vv] + sig[i+1][a][i+1][ii]*sig[i+1][a1][h][comp*(comp*(comp*(jj+1) + (kk+1)) + (uu+1)) + vv] + sig[i+1][a1][h][ind];
                                 ind += 1;
                              }
                           }
                        }
                     }
                  }

                  for (ii = 0; ii < comp; ii++) {
                     for (jj = 0; jj < comp; jj++) {
                        for (kk = 0; kk < comp; kk++) {
                           for (uu = 0; uu < comp; uu++) {
                              for (vv = 0; vv < comp; vv++) {
                                 for (hh = 0; hh < comp; hh++) {
                                    sig[i][k][h][ind] = sig[i+1][a][h][ind] + sig[i+1][a][i+1][comp*(comp*(comp*(comp*(ii+1) + (jj+1)) + (kk + 1)) + (uu+1)) + vv]*sig[i+1][a1][h][hh] + sig[i+1][a][i+1][comp*(comp*(comp*(ii+1) + (jj+1)) + (kk+1)) + uu]*sig[i+1][a1][h][comp*(vv+1) + hh] + sig[i+1][a][i+1][comp*(comp*(ii+1) + (jj+1)) + kk]*sig[i+1][a1][h][comp*(comp*(uu+1) + (vv+1)) + hh] + sig[i+1][a][i+1][comp*(ii+1) + jj]*sig[i+1][a1][h][comp*(comp*(comp*(kk+1) + (uu+1)) + (vv+1)) + hh] + sig[i+1][a][i+1][ii]*sig[i+1][a1][h][comp*(comp*(comp*(comp*(jj+1) + (kk+1)) + (uu+1)) + (vv+1)) + hh] + sig[i+1][a1][h][ind];

                                    ind += 1;
                                 }
                              }
                           }
                        }
                     }
                  }

                  for (ii = 0; ii < comp; ii++) {
                     for (jj = 0; jj < comp; jj++) {
                        for (kk = 0; kk < comp; kk++) {
                           for (uu = 0; uu < comp; uu++) {
                              for (vv = 0; vv < comp; vv++) {
                                 for (hh = 0; hh < comp; hh++) {
                                    for (ll = 0; ll < comp; ll++) {
                                       sig[i][k][h][ind] = sig[i+1][a][h][ind] + sig[i+1][a][i+1][comp*(comp*(comp*(comp*(comp*(ii+1) + (jj+1)) + (kk + 1)) + (uu+1)) + (vv+1)) + hh]*sig[i+1][a1][h][ll] + sig[i+1][a][i+1][comp*(comp*(comp*(comp*(ii+1) + (jj+1)) + (kk+1)) + (uu+1)) + vv]*sig[i+1][a1][h][comp*(hh+1) + ll] + sig[i+1][a][i+1][comp*(comp*(comp*(ii+1) + (jj+1)) + (kk+1)) + uu]*sig[i+1][a1][h][comp*(comp*(vv+1) + (hh+1)) + ll] + sig[i+1][a][i+1][comp*(comp*(ii+1) + (jj+1)) + kk]*sig[i+1][a1][h][comp*(comp*(comp*(uu+1) + (vv+1)) + (hh+1)) + ll] + sig[i+1][a][i+1][comp*(ii+1) + jj]*sig[i+1][a1][h][comp*(comp*(comp*(comp*(kk+1) + (uu+1)) + (vv+1)) + (hh+1)) + ll] + sig[i+1][a][i+1][ii]*sig[i+1][a1][h][comp*(comp*(comp*(comp*(comp*(jj+1) + (kk+1)) + (uu+1)) + (vv+1)) + (hh+1)) + ll] + sig[i+1][a1][h][ind];
                                       ind += 1;
                                    }
                                 }
                              }
                           }
                        }
                     }
                  }
	       }
	    }
	 }
      }

      j = (j+1)/2;
   } 

/* Here we print the results. */

   for (i = 0; i < cum; i++) {
      printf ("%lf ", sig[0][1][0][i]);
   }

   printf("\n"); 


/* Finally, we free the allocated memory and end the execution. */

   for (i = 0; i < comp; i++) {
      free(vec[i]);
   }

   free(vec);

   for (i = 0; i < n-1; i++) {
      free(aux1[i]);
      free(aux2[i]);
      free(aux3[i]);
      free(aux4[i]);
      free(aux5[i]);
      free(aux6[i]);
      free(aux7[i]);

   }

   free(aux1);
   free(aux2);
   free(aux3);
   free(aux4);
   free(aux5);
   free(aux6);
   free(aux7);


   j = 2;
   for (i = 0; i < m; i++) {
      free(mat[i]);
      for (k = 1; k < j; k++) {
	 free(mati[i][k]);
	 for (h = 0; h <= i; h++) {
	    free(sig[i][k][h]);
	 }
	 free(sig[i][k]);
      }

      j = 2*j -1;
      free(mati[i]);
      free(sig[i]);
   }

   free(sig);
   free(mat);
   free(mati);


   return 0;

}
