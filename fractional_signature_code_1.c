/* 
fractional_signature_code_1.c
Copyright (C) 2025  Rubén, José Manuel

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

/* For this program to work correctly, among other things, one needs to have the GNU Scientific Library installed, which is used to evaluate certain mathematical functions. */

/* The aim of this program is, given a .csv file containing by rows the values of the 784 pixels of each image of a handwritten digit, to consider the paths given by piecewise linear interpolation of the sequence of pixel values together with their coordinates in the image, as explained in the paper, and to calculate for each of them its discrete fractional signature of parameter alpha. Subsequently, the program also saves in the appropriate .csv format these signatures by rows in a file that is indicated, so that the corresponding machine learning models can later be applied. */

/* When compiling the program, if the gcc compiler is used, it is recommended to use the -O2 option to increase the execution speed, as this is a program that can take a long time to run. */

/* We should also note that although this program allows us to use the value of alpha = 1, for which the discrete fractional signature coincides with the original signature, it is not advisable to use it, it being preferable to consider other already available functions and programs for the calculation of the original signature. This is due to the fact that small numerical errors are naturally and inevitably introduced in the calculation, which for other values of the parameter are negligible but for alpha = 1 cause dependencies that the original signature does not present. These can alter the results significantly enough, especially if we subsequently standardise the signature terms, to affect predictions and accuracy rates. */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<gsl/gsl_sf.h>
#include<string.h>
#include<omp.h>

void calcula_sig(int, int, int, int, double **, double ***, int **, double ***, double ****);

int main(void) {
   char str[50000];
   int i, a, d, k, h, j, l, comp = 3, m, n = 784, num_paths = 60000, aux, auxp, lensig = 0, depth = 7;
   int **mat;
   double alpha = 1.15;
   double **matsig, ***matvec, ***mat1, ****sig, ***auxvec;
   FILE *file;

/* To begin with, we reserve memory for the variable matsig, which we will use to store the discrete fractional signatures of each of the paths we consider, and then print them in the indicated file. As will be seen later, we proceed in this way because in this program we will use parallelisation to carry out the calculation of the discrete fractional signatures, which has led us to save the already computed signatures in memory before printing them in order to have a better control of the order in which they are printed in the file, which is essential. */

/* Note that, after each dynamic memory allocation, in the event of an error, we release all the memory we had previously allocated and close all the files we are working with. This will be a common procedure throughout the code and therefore, we will not highlight it in the comments that follow. */
   
   matsig = (double **) malloc(num_paths*sizeof(double *));

   if (matsig == NULL) {
      printf ("Error in the memory allocation. We put an end to the execution\n");

      return 1;
   }

   for (i = 1; i <= depth; i++) {
      aux = 1;
      for (j = 0; j < i; j++) {
         aux *= comp;
      }
      lensig += aux;
   }

   for (i = 0; i < num_paths; i++) {
      matsig[i] = (double *) malloc(lensig*sizeof(double));

      if (matsig[i] == NULL) {
	 printf ("Error in the memory allocation. We put an end to the execution\n");

	 for (j = 0; j < i; j++) {
            free(matsig[j]);
	 }

	 free(matsig);

	 return 1;
      }
   }

/* From here, we open the given file from which the data sequences corresponding to each of the images of the handwritten digits will be extracted. We also allocate memory for the variable mat, which we will use to store these sequences. */

   file = fopen("data_train.csv", "r");

   if (file == NULL) {
      for (i = 0; i < num_paths; i++) {
	 free(matsig[i]);
      }
      
      free(matsig);
      printf ("Error opening a file. We end the execution\n");

      return 1;
   }

   mat = (int **) malloc(num_paths*sizeof(int *));

   if (mat == NULL) {

      for (i = 0; i < num_paths; i++) {
         free(matsig[i]);
      }
      
      free(matsig);
      fclose(file);

      printf("Error in the memory allocation. We put an end to the execution\n");

      return 1;
   }

   for (i = 0; i < num_paths; i++) {
      mat[i] = (int *) malloc(n*sizeof(int));

      if (mat[i] == NULL) {
	 for (k = 0; k < i; k++) {
	    free(mat[k]);
	 }

	 for (k = 0; k < num_paths; k++) {
            free(matsig[k]);
         }

	 free(matsig);
	 free(mat);
	 fclose(file);

	 printf ("Error in the memory allocation. We put an end to the execution\n");

	 return 1;
      } 
   }


   for (i = 0; i < num_paths; i++) {

      if (fgets(str, sizeof(str), file) == NULL) {
	 printf ("Error reading the file. We end the execution\n");


         for (i = 0; i < num_paths; i++) {
            free(mat[i]);
            free(matsig[i]);
         }

         free(mat);
         free(matsig);
         fclose(file);

	 return 1;
      }

      for (j = 0; j < n; j++) {
         sscanf(str, "%d,", &mat[i][j]);
         if (mat[i][j]/10 == 0) {
            memmove(str, str + 2, strlen(str));
         } else{
            if(mat[i][j]/100 == 0) {
               memmove(str, str + 3, strlen(str));
            } else{
               memmove(str, str + 4, strlen(str));
            }
         }
      }
   }

   fclose(file);

/* Once the necessary data has been read, we allocate memory for the variable matvec, which will be used to store each of the above sequences but augmented with the coordinates of each of the pixels in the corresponding image. We proceed in this way because we will reuse the variable mat later and therefore need to be able to free it (note that, although it is not necessary in this example, because the pixel values are integers, matvec is of type double and mat is not, which could be relevant when applying this program to other problems). */

   matvec = (double ***) malloc(num_paths*sizeof(double **));

   if (matvec == NULL) {
      printf ("Error in the memory allocation. We put an end to the execution\n");
      
      for (i = 0; i < num_paths; i++) {
	 free(mat[i]);
	 free(matsig[i]);
      }

      free(mat);
      free(matsig);

      return 1;
   }

   for (i = 0; i < num_paths; i++) {
      matvec[i] = (double **) malloc(comp*sizeof(double *));

      if (matvec[i] == NULL) {
	 printf ("Error in the memory allocation. We put an end to the execution\n");

	 for (k = 0; k < i; k++) {
	    for (a = 0; a < comp; a++) {
	       free(matvec[k][a]);
	    }

	    free(matvec[k]);
	 }

	 free(matvec);
	 for (k = 0; k < num_paths; k++) {
            free(mat[k]);
	    free(matsig[k]);
         }

         free(mat);
         free(matsig);

	 return 1;

      }

      for (j = 0; j < comp; j++) {
         matvec[i][j] = (double *) malloc(n*sizeof(double));

	 if (matvec[i][j] == NULL) {
	    printf ("Error in the memory allocation. We put an end to the execution\n");

            for (k = 0; k < i; k++) {
               for (a = 0; a < comp; a++) {
                  free(matvec[k][a]);
               }

               free(matvec[k]);
            }

	    for (a = 0; a < j; a++) {
	       free(matvec[i][a]);
	    }

	    free(matvec[i]);
            free(matvec);
            for (k = 0; k < num_paths; k++) {
               free(mat[k]);
	       free(matsig[k]);
            }

            free(mat);
            free(matsig);

            return 1;

         }
      }
   }

   for (i = 0; i < num_paths; i++) {
      for (j = 0; j < n; j++) {
         aux = j/28;
         matvec[i][0][j] = (double) aux;
         aux = j%28;
         matvec[i][1][j] = (double) aux;
         matvec[i][2][j] = (double) mat[i][j];
      }
   }


   for (i = 0; i < num_paths; i++) {
      free(mat[i]);
   }

   free(mat);

/* Once the augmented sequences have been saved using the variable matvec and the memory given to mat has been freed, we start preparing to carry out the discrete fractional sigature calculations. To do this, we will use the variables mat and mat1 to store auxiliary information that will guide us throughout the calculation algorithm, indicating which calculations we have to perform at any given moment. To obtain the discrete fractional signatures, although we could program the algorithm given by its generalised Chen identity, this is very slow and not feasible. Therefore, we have opted for a non-recursive algorithm, which first calculates the discrete fractional signatures of the corresponding linear paths and then uses them to progressively calculate other intermediate signatures until we reach the desired one. Therefore, we will need the auxiliary data that we store in to mat and mat1 to guide us and to know which discrete fractional signatures we have to calculate at each step. */

   m = floor(log(n)/log(2)) + 2;

   
   mat = (int **) malloc(m*sizeof(int *));

   if (mat == NULL) {
      printf ("Error in the memory allocation. We put an end to the execution\n");
      
      for (i = 0; i < num_paths; i++) {
         for (j = 0; j < comp; j++) {
            free(matvec[i][j]);
         }

         free(matvec[i]);
	 free(matsig[i]);
      }

      free(matsig);
      free(matvec);
      
      return 1;
   }

   j = 2;
   for (i = 0; i < m; i++) {
      mat[i] = (int *) malloc (j*sizeof(int));

      if (mat[i] == NULL) {
	 printf ("Error in the memory allocation. We put an end to the execution\n");

	 for (a = 0; a < i; a++) {
	    free(mat[a]);
	 }

	 for (a = 0; a < num_paths; a++) {
            for (k = 0; k < comp; k++) {
               free(matvec[a][k]);
            }

            free(matvec[a]);
	    free(matsig[a]);
         }

         free(matsig);
         free(matvec);
	 free(mat);

	 return 1;
      }

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
   mat1 = (double ***) malloc (m*sizeof(double **));

   if (mat1 == NULL) {
      printf ("Error in the memory allocation. We put an end to the execution\n");

      for (i = 0; i < m; i++) {
         free(mat[i]);
      }

      free(mat);

      for (i = 0; i < num_paths; i++) {
         for (l = 0; l < comp; l++) {
            free(matvec[i][l]);
         }

         free(matvec[i]);
         free(matsig[i]);
      }

      free(matvec);
      free(matsig);

      return 1;
   }
   
   for (i = 0; i < m; i++) {
      mat1[i] = (double **) malloc(j*sizeof(double *));
      
      if (mat1[i] == NULL) {
         printf ("Error in the memory allocation. We put an end to the execution\n");

	 l = 2; 
	 for (a = 0; a < i; a++) {
	    for (k = 1; k < l; k++) {
	       free(mat1[a][k]);
	    }

	    l = 2*l-1;
	    free(mat1[a]);
	 }

         for (i = 0; i < m; i++) {
            free(mat[i]);
         }

         free(mat);
         free(mat1);

         for (i = 0; i < num_paths; i++) {
            for (l = 0; l < comp; l++) {
               free(matvec[i][l]);
            }

            free(matvec[i]);
            free(matsig[i]);
         }

         free(matvec);
         free(matsig);

         return 1;
      }
      
      for (k = 1; k < j; k++) {
         mat1[i][k] = (double *) malloc((i+1)*sizeof(double));


	 if (mat1[i][k] == NULL) {
            printf ("Error in the memory allocation. We put an end to the execution\n");

	    for (l = 1; l < k; l++) {
	       free(mat1[i][l]);
	    }

	    free(mat1[i]);

            l = 2;
            for (a = 0; a < i; a++) {
               for (k = 1; k < l; k++) {
                  free(mat1[a][k]);
               }

               l = 2*l-1;
               free(mat1[a]);
            }

            for (i = 0; i < m; i++) {
               free(mat[i]);
            }

            free(mat);
            free(mat1);

            for (i = 0; i < num_paths; i++) {
               for (l = 0; l < comp; l++) {
                  free(matvec[i][l]);
               }

               free(matvec[i]);
               free(matsig[i]);
            }

            free(matvec);
            free(matsig);

            return 1;
         }

	 for (h = 0; h < i; h++) {
            mat1[i][k][h] = mat1[i-1][k/2 + k%2][h];
         }
         if (i == 0) {
            mat1[i][k][i] = n-1;
         } else {
            if (k%2 == 1) {
              mat1[i][k][i] = (mat[i][k] + mat[i][k+1])/2.;
              if (mat1[i][k][i] == (double) mat[i][k]) {
                 mat1[i][k][i] = 0.;
              }
            } else {
              mat1[i][k][i] = 0.;
            }
         }
      }

      j = 2*j -1;
   }
   

/* Finally, to conclude the preparations for the calculation of the discrete fractional signatures, we allocate dynamic memory for the auxvec variable, which we will again use to pre-calculate and store values that we will need during the algorithm. In this case, we will store the coefficients by which we have to multiply the different terms of the discrete fractional signatures of the linear paths to calculate them. In this way, the computations are only performed once and the total execution time is therefore reduced. Note that, if instead of using auxvec, which is a triple pointer, we would use 7 (depth of the signature) double pointers aux1, ..., aux7, we could slightly reduce the execution time (as done in the provided isolated code). However, this alternative is more cumbersome and would increase the code by about 700 lines. */

   auxvec = (double ***) malloc(depth*sizeof(double **));

   if (auxvec == NULL) {
      printf ("Error in the memory allocation. We put an end to the execution\n");

      j = 2;
      for (i = 0; i < m; i++) {
         free(mat[i]);
         for (k = 1; k < j; k++) {
            free(mat1[i][k]);
         }

         j = 2*j -1;
         free(mat1[i]);
      }

      free(mat);
      free(mat1);

      for (i = 0; i < num_paths; i++) {
         for (j = 0; j < comp; j++) {
            free(matvec[i][j]);
         }

         free(matvec[i]);
         free(matsig[i]);
      }

      free(matvec);
      free(matsig);

      return 1;
   }  


   for (i = 0; i < depth; i++) {
      auxvec[i] = (double **) malloc((n-1)*sizeof(double *));

      if (auxvec[i] == NULL) {
         printf ("Error in the memory allocation. We put an end to the execution\n");

         for (j = 0; j < i; j++) {
            free(auxvec[j]);
         }

         j = 2;
         for (i = 0; i < m; i++) {
            free(mat[i]);
            for (k = 1; k < j; k++) {
               free(mat1[i][k]);
            }

            j = 2*j -1;
            free(mat1[i]);
         }

         free(mat);
         free(mat1);

         for (i = 0; i < num_paths; i++) {
            for (j = 0; j < comp; j++) {
               free(matvec[i][j]);
            }

            free(matvec[i]);
	    free(matsig[i]);
         }

	 free(matsig);
         free(matvec);
         free(auxvec);

         return 1;
      }

   }


   for (l = 0; l < depth; l++) {
      for (a = 0; a < n-1; a++) {

         auxvec[l][a] = (double *) malloc ((2*n-1)*sizeof(double));

         if (auxvec[l][a] == NULL) {

            printf("Error in the memory allocation. We put an end to the execution\n");


	    for (i = 0; i < a; i++) {
               free(auxvec[l][i]);
            }

            for (i = 0; i < l; i++) {
               for (j = 0; j < n-1; j++) {
                  free(auxvec[i][j]);
               }
            }

            for (i = 0; i < depth; i++) {
               free(auxvec[i]);
            }

            free(auxvec);


            j = 2;
            for (i = 0; i < m; i++) {
               free(mat[i]);
               for (k = 1; k < j; k++) {
                  free(mat1[i][k]);
               }

               j = 2*j -1;
               free(mat1[i]);
            }

            free(mat);
            free(mat1);

            for (i = 0; i < num_paths; i++) {
               for (j = 0; j < comp; j++) {
                  free(matvec[i][j]);
               }
               
	       free(matvec[i]);
	       free(matsig[i]);
            }

            free(matvec);
	    free(matsig);

            return 1;
         }

         for (d = 2*a + 2; d < 2*n-1; d++) {
            auxvec[l][a][d] = (pow(d/2.-a, alpha*(l+1))*gsl_sf_beta_inc(l*alpha + 1, alpha, 1./(d/2. - a))*gsl_sf_beta(l*alpha + 1, alpha))/(gsl_sf_gamma(alpha)*gsl_sf_gamma(l*alpha + 1));
         }
      }
   }


/* From here, once we have pre-calculated all the values and auxiliary data we need, we start a parallel session using the OpenMp library (7 threads in this example). In it, using the variable sig, we allocate dynamic memory to store all the discrete fractional signatures that we calculate throughout the algorithm. Note that we use the aux and auxp variables to control if an error occurs in the memory reservation and then act appropriately. */   

   aux = 0;

#pragma omp parallel num_threads(7) private(sig, i, h, j, k, a, d, l, auxp)
{

   auxp = 0;	
   
   j = 2;

   sig = (double ****) malloc(m*sizeof(double ***));

   if (sig == NULL) {

      #pragma omp critical
      {
         aux = 1;
      }

      auxp = 1;
   }


   for (i = 0; i < m && auxp == 0; i++) {
      sig[i] = (double ***) malloc(j*sizeof(double *));

      if (sig[i] == NULL) {

         l = 2;
         for (a = 0; a < i; a++) {
            for (k = 1; k < l; k++) {
               for (h = 0; h <= a; h++) {
                  free(sig[a][k][h]);
               }

               free(sig[a][k]);
            }

            l = 2*l-1;
            free(sig[a]);
         }

         free(sig);

         #pragma omp critical
         {
            aux = 1;
         }

         auxp = 1;
      }

      for (k = 1; k < j && auxp == 0; k++) {
         sig[i][k] = (double **) malloc((i+1)*sizeof(double *));

         if (sig[i][k] == NULL) {

            for (l = 1; l < k; l++) {
               for (h = 0; h <= i; h++) {
                  free(sig[i][l][h]);
               }

               free(sig[i][l]);
            }

            free(sig[i]);

            l = 2;
            for (a = 0; a < i; a++) {
               for (k = 1; k < l; k++) {
                  for (h = 0; h <= a; h++) {
                     free(sig[a][k][h]);
                  }

                  free(sig[a][k]);
               }

               l = 2*l-1;
               free(sig[a]);
            }

            free(sig);

            #pragma omp critical
            {
               aux = 1;
            }

            auxp = 1;
         }

	 for (h = 0; h <= i && auxp == 0; h++) {
            sig[i][k][h] = (double *) malloc(lensig*sizeof(double));


            if (sig[i][k][h] == NULL) {

               for (l = 0; l < h; l++) {
                  free(sig[i][k][l]);
               }

               free(sig[i][k]);

               for (l = 1; l < k; l++) {
                  for (h = 0; h <= i; h++) {
                     free(sig[i][l][h]);
                  }

                  free(sig[i][l]);
               }

               free(sig[i]);

               l = 2;
               for (a = 0; a < i; a++) {
                  for (k = 1; k < l; k++) {
                     for (h = 0; h <= a; h++) {
                        free(sig[a][k][h]);
                     }

                     free(sig[a][k]);
                  }

                  l = 2*l-1;
                  free(sig[a]);
               }

               free(sig);

               #pragma omp critical
               {
                  aux = 1;
               }

               auxp = 1;
            }
         }
      }

      j = 2*j - 1;
   }


   #pragma omp barrier 

   if (aux == 0) {
      #pragma omp for schedule(static)

      for (i = 0; i < num_paths; i++) {
         calcula_sig(n, m, comp, depth, matvec[i], auxvec, mat, mat1, sig);
         for (j = 0; j < lensig; j++) {
	    matsig[i][j] = sig[0][1][0][j];
         }
         printf ("%d\n", i);
      }
   }
   
   #pragma omp barrier

   if (auxp == 0) {

      j = 2;
      for (i = 0; i < m; i++) {
         for (k = 1; k < j; k++) {
            for (h = 0; h <= i; h++) {
               free(sig[i][k][h]);
            }
            free(sig[i][k]);
         }

         j = 2*j -1;
         free(sig[i]);
      }

      free(sig);
   }

}


/* Once the calculations of the discrete fractional signatures have been completed, we free all the memory we have allocated and write the values obtained in the indicated file, subsequently ending the execution. */


   if (aux != 0) {
      printf ("Error in the memory allocation. We put an end to the execution\n");

      for (l = 0; l < depth; l++) {
         for (i = 0; i < n-1; i++) {
            free(auxvec[l][i]);
         }

         free(auxvec[l]);
      }

      free(auxvec);

      j = 2;
      for (i = 0; i < m; i++) {
         free(mat[i]);
         for (k = 1; k < j; k++) {
            free(mat1[i][k]);
         }

         j = 2*j -1;
         free(mat1[i]);
      }

      free(mat);
      free(mat1);

      for (i = 0; i < num_paths; i++) {
         free(matsig[i]);
      }

      free(matsig);

      for (i = 0; i < num_paths; i++) {
         for (j = 0; j < comp; j++) {
            free(matvec[i][j]);
         }

         free(matvec[i]);
      }

      free(matvec);

      return 1;
   }


   for (l = 0; l < depth; l++) {
      for (i = 0; i < n-1; i++) {
         free(auxvec[l][i]);
      }

      free(auxvec[l]);
   }

   free(auxvec);

   j = 2;
   for (i = 0; i < m; i++) {
      free(mat[i]);
      for (k = 1; k < j; k++) {
         free(mat1[i][k]);
      }

      j = 2*j -1;
      free(mat1[i]);
   }
   
   free(mat);
   free(mat1);
   

   file = fopen("results_train.csv", "w");

   if (file == NULL) {
      printf ("Error opening a file. We end the execution\n");

      for (i = 0; i < num_paths; i++) {
         free(matsig[i]);
      }

      free(matsig);

      for (i = 0; i < num_paths; i++) {
         for (j = 0; j < comp; j++) {
            free(matvec[i][j]);
         }
	 
	 free(matvec[i]);
      }

      free(matvec);

      return 1;
   }
   
   for (i = 0; i < num_paths; i++) {
      for (j = 0; j < lensig - 1; j++) {
         fprintf(file, "%le,", matsig[i][j]);
      }

      fprintf(file, "%le\n", matsig[i][lensig - 1]);

      free(matsig[i]);
   }

   free(matsig);
   fclose(file);


   for (i = 0; i < num_paths; i++) {
      for (j = 0; j < comp; j++) {
         free(matvec[i][j]);
      }

      free(matvec[i]);
   }

   free(matvec);

   return 0;

}


void calcula_sig (int n, int m, int comp, int depth, double **vec, double ***auxvec, int **mat, double ***mat1, double ****sig) {
   int i, j, k, h, a, a1, d, ii, jj, kk, uu, vv, hh, ll, ind;

/* We use this function to compute the discrete fractional signatures of the paths. To do this, as already explained, we will not use the recursive algorithm that the definition based on a generalisation of Chen's identity would suggest because it is too slow, but we opt for an alternative algorithm that first calculates the discrete fractional signatures of the linear paths that form the original one and then combines them in the appropriate way to obtain the desired discrete fractional signature in the end. */

   j = 2;
   for (i = 0; i < m-1; i++) {
      j = 2*j -1;
   }

   for (i = m-1; i >= 0; i--) {
      for (k = 1; k < j; k++) {
	 for (h = 0; h <= i; h++) {
	    if ((mat1[i][k][h] != 0.) && (mat[i][k]-mat[i][k-1] != 0)) {
	       if (mat[i][k-1] + 1 ==  mat[i][k]) {
		  d = (int)(2*mat1[i][k][h]);
		  a = mat[i][k-1];
		  a1 = mat[i][k];
		 
		  ind = 0;
                  for (ii = 0; ii < comp; ii++) {
                     sig[i][k][h][ind] = (vec[ii][a1] - vec[ii][a])*auxvec[0][a][d];
                     ind += 1;
                  }
                  for (ii = 0; ii < comp; ii++) {
                     for (jj = 0; jj < comp; jj++) {
                        sig[i][k][h][ind] = (vec[ii][a1] - vec[ii][a])*(vec[jj][a1] - vec[jj][a])*auxvec[1][a][d];
                        ind += 1;
		     }
                  }
                  for (ii = 0; ii < comp; ii++) {
                     for (jj = 0; jj < comp; jj++) {
                        for (kk = 0; kk < comp; kk++) {
                           sig[i][k][h][ind] = (vec[ii][a1] - vec[ii][a])*(vec[jj][a1] - vec[jj][a])*(vec[kk][a1] - vec[kk][a])*auxvec[2][a][d];
                           ind += 1;
                        }
                     }
                  }
                  for (ii = 0; ii < comp; ii++) {
                     for (jj = 0; jj < comp; jj++) {
                        for (kk = 0; kk < comp; kk++) {
                           for (uu = 0; uu < comp; uu++) {
                              sig[i][k][h][ind] = (vec[ii][a1] - vec[ii][a])*(vec[jj][a1] - vec[jj][a])*(vec[kk][a1] - vec[kk][a])*(vec[uu][a1] - vec[uu][a])*auxvec[3][a][d];
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
                                 sig[i][k][h][ind] = (vec[ii][a1] - vec[ii][a])*(vec[jj][a1] - vec[jj][a])*(vec[kk][a1] - vec[kk][a])*(vec[uu][a1] - vec[uu][a])*(vec[vv][a1] - vec[vv][a])*auxvec[4][a][d];
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
                                    sig[i][k][h][ind] = (vec[ii][a1] - vec[ii][a])*(vec[jj][a1] - vec[jj][a])*(vec[kk][a1] - vec[kk][a])*(vec[uu][a1] - vec[uu][a])*(vec[vv][a1] - vec[vv][a])*(vec[hh][a1] - vec[hh][a])*auxvec[5][a][d];
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
                                       sig[i][k][h][ind] = (vec[ii][a1] - vec[ii][a])*(vec[jj][a1] - vec[jj][a])*(vec[kk][a1] - vec[kk][a])*(vec[uu][a1] - vec[uu][a])*(vec[vv][a1] - vec[vv][a])*(vec[hh][a1] - vec[hh][a])*(vec[ll][a1] - vec[ll][a])*auxvec[6][a][d];
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

/* Note that we are using long expressions to compute the indices we access using the pointers and that we are repeating many calculations. However, if we try to precompute these indices to avoid these repetitions, the memory accesses are more expensive than the calculations and the execution time is longer. Therefore, we have chosen to keep this more cumbersome but better performing version of the code here. */

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

   return;

}
