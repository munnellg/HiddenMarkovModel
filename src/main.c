#include "hmm.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define N 2  /* Number of possible states */
#define M 3 /* Number of possible observation classes */

#define T 4 /* Number of Observations */

void
print_matrix ( Mat *m, int w, int h ) {
    int i;
    for(i=0; i<w*h; i++) {
        printf("%0.4f ", m[i]);
        if( (i+1)%w == 0 ) {
            printf("\n");
        }
    }
}

void
print_sequence ( Observation *O, int len ) {
    int i;
    for( i=0; i<len; i++ ) {
        printf("%d ", O[i]);
    }
    printf("\n");
}

void
print_banner ( char *msg ) {
   printf("***************************************************************\n");
   printf("* %-59s *\n", msg);
   printf("***************************************************************\n");
}

void
test1() {
    /* Observation vector.
     * Statically allocating this because when it was allocated onn the stack
     * it caused segfaults if I made it too big */
    static Observation O[T] = { 0, 1, 0, 2 };

    /* Hidden Markov model Hmm( A, B, pi ) */
    Hmm hmm;

    /* State Transition probability matrix */
    Mat A[N*N] = { 0.7, 0.3, 0.4, 0.6 };

    /* Observation Probability matrix */
    Mat B[N*M] = { 0.1, 0.4, 0.5, 0.7, 0.2, 0.1 };

    /* Initial State Probability vector */
    Mat pi[N]  = { 0.6, 0.4 };

    print_banner("Testing forward pass");
    
    print_sequence( O, T );

    /* Initialize the Hidden Markov Model using the data we have gathered */
    hmm_init( &hmm, N, M, A, B, pi );

    /* Print the probability of observing sequence O given the parameters of 
     * the Hidden Markov Model. This tests the Forward Pass algorithm.
     * Note that Pr(O | HMM) tends towards zero very quickly. For large values
     * of T, you are unlikely to get anything useful back */
    printf("%f\n\n", hmm_pr_obs( &hmm, O, T ) );
}

void
test2() {
    /* Hidden Markov model Hmm( A, B, pi ) */
    Hmm hmm;

    /* State Transition probability matrix */
    Mat A[N*N] = { 0.7, 0.3, 0.4, 0.6 };

    /* Observation Probability matrix */
    Mat B[N*M] = { 0.1, 0.4, 0.5, 0.7, 0.2, 0.1 };

    /* Initial State Probability vector */
    Mat pi[N]  = { 0.6, 0.4 };

    print_banner("Testing the training initialization function");

    hmm_init( &hmm, N, M, A, B, pi );

    hmm_prepare_train( &hmm );

    printf("A:\n");
    print_matrix( A, N, N );
    printf("\n");

    printf("B:\n");
    print_matrix( B, M, N );
    printf("\n");

    printf("pi:\n");
    print_matrix( pi, N, 1 );
    printf("\n");
}

void
test3() {
   /* Hidden Markov model Hmm( A, B, pi ) */
    Hmm hmm;

    /* State Transition probability matrix */
    Mat A[N*N] = { 0.7, 0.3, 0.4, 0.6 };

    /* Observation Probability matrix */
    Mat B[N*M] = { 0.1, 0.4, 0.5, 0.7, 0.2, 0.1 };

    /* Initial State Probability vector */
    Mat pi[N]  = { 0.6, 0.4 };

    print_banner("Testing the state prediction function");

    hmm_init( &hmm, N, M, A, B, pi );
    
    Observation O[] = { 0, 1, 0, 2 };
    State X[4];

    hmm_obs_states ( &hmm, O, X, sizeof(O)/sizeof(O[0]) );

    print_sequence( X, sizeof(X)/sizeof(X[0]) );
}

void
classify_chars(char *fname) {
    FILE *f;
    int i;
    Hmm *hmm;
    Observation O[50000] = {0};

    f = fopen(fname, "r");
    
    if(!f) {
        return;
    }
    
    printf("Reading chars... ");
    for( i=0; i<50000 && !feof(f); i++ ) {
        O[i] = fgetc(f);
        if( O[i] == ' ' ) {
            O[i] = 26;
        } else {
            O[i] -= 'a';
        }
    }

    fclose(f);
    printf("OK\n");
    /* 2 states, 27 possible observations */
    hmm = hmm_new( 2, 27 );

    hmm_fit( hmm, O, i, 100 );

    printf("pi\n");
    print_matrix ( hmm->pi, 2, 1 ); 

    printf("A\n");
    print_matrix ( hmm->A, 2, 2 ); 

    printf("B\n");
    print_matrix ( hmm->B, 27, 2 );

    hmm_free(hmm);
}

int
main( int argc, char *argv[] ) {
    /* Seed PRNG */
    srand(time(NULL));
    
    setbuf(stdout, NULL);

    if( argc > 1 ) {
        classify_chars( argv[1] );
    } else {
        printf("usage: hmm FILE1\n");
    }
    return EXIT_SUCCESS;
}
