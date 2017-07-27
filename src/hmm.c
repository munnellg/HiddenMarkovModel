#include "hmm.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>

/* 
 * ===  FUNCTION  =============================================================
 *         Name:  mat_init_train_row
 *  Description:  Training the HMM requires that the various HMM matrices be
 *                initialized to **almost** uniform values. These values must
 *                still be normalized. This function initializes a row of a
 *                matrix to be almost the same value, but to deviate by slight
 *                amounts
 *       Inputs:  m   => The matrix to be initialized
 *                row => The row we want to target
 *                w   => The width of the row
 * ============================================================================
 */
static void
mat_init_row_train ( Mat *m, int row, int w ) {
    int i;
    double scale, sum, uniform;

    /* depending on how big the width is we will want smaller and smaller
     * deviations. Use log to determine the appropriate deviation */
    scale = log10(w) * 10;

    /* sum is used for normalization later */
    sum = 0;

    /* uniform is the value we would use if we wanted row values to be equal */
    uniform = 1.0/w;

    /* initialize to **almost** uniform values */
    for( i=0; i<w; i++ ) {
        m[i + row*w] = uniform + ((double)rand()/RAND_MAX)/scale;
        sum += m[i + row*w];
    }

    /* normalize */
    for( i=0; i<w; i++ ) {
        m[i + row*w] /= sum;
    }
}

/* 
 * ===  FUNCTION  =============================================================
 *         Name:  mat_scale_row
 *  Description:  Applies a scaling factor to a single row in a matrix.
 *                Useful for normalizing the alpha matrix when performing
 *                the forward pass.
 *
 *       Inputs:  Mat   => The matrix to be scaled
 *                col   => Target column of the matrix
 *                w     => Matrix width
 *                h     => Matrix height
 *                scale => Scaling factor to be applied
 * ============================================================================
 */
static void
mat_scale_row( Mat *m, int row, int w, double scale ) {
    int i;
    
    for( i=0; i<w; i++ ) {
        m[IDX2D(i, row, w)] *= scale; 
    }
}

/* 
 * ===  FUNCTION  =============================================================
 *         Name:  log_probability
 *  Description:  Derive the log probability of a sequence of observations 
 *                based on coefficients obtained when performing the forward
 *                pass. Result will be negative and grows larger as the 
 *                model converges.
 *
 *       Inputs:  c => coefficients obtained during forward pass
 *                T => length of coefficients vector
 *
 *      Outputs:  Log probability of observation sequence from which c was
 *                obtained
 * ============================================================================
 */
static double
log_probability( Mat *c, int T ) {
    int i;
    double log_pr;
    
    /* log probability is the sum of the logs of all the coefficients obtained
     * during the forward pass */
    log_pr = 0;
    for( i=0; i<T; i++ ) {
        log_pr += log(c[i]);
    }
    
    /* return the negated log probability. This is because really we are 
     * computing the log of 1/product(c). Note that while the negation is
     * mathematically correct, for practical purposes we don't really need it.
     * Once we know how our value should change given convergence, we don't
     * really need to worry about whether it is positive or negative. */
    return -log_pr;
}

/* 
 * ===  FUNCTION  =============================================================
 *         Name:  alpha_t_i
 *  Description:  Compute alpha value of state q_i at time t given the 
 *                observation sequence O and the Hidden Markov Model hmm.
 *
 *                Alpha tells us the probability that we moved from some 
 *                previous state at time t-1 to state q_i at time t based on
 *                the relative probability that we were in each of those 
 *                previous states and based on the probability that state q_i
 *                will actually emit the observation we see at O[t].
 *
 *                By definition, the computation of alpha is recursive with 
 *                the base case at time t=0 being equal to the initial 
 *                probability of the state as defined by hmm->pi. However,
 *                it is extremely computationally inefficient to implement it
 *                as a recursive function, so we won't do that.
 *
 *       Inputs:  hmm    => The HMM on which probabilities are based
 *                alphas => A matrix of alpha values. It is assumed that 
 *                          alphas for time t-1 have already been computed
 *                O      => The sequence of observations on which the alphas 
 *                          are based.
 *                t      => The time at which we want to compute alpha
 *                i      => The state for which we want to compute alpha
 *                T      => The maximum length of the sequence O
 *
 *      Outputs:  Alpha value for state q_i at time t for observation sequence
 *                O and hidden markov model hmm.
 * ============================================================================
 */
static double
alpha_t_i ( Hmm *hmm, Mat *alphas, Observation *O, int t, int i ) {
    int j;
    double alpha;
 
    /* container for final alpha value */ 
    alpha = 0;

    if( t == 0 ) {
        /* Base case for t=0. Alpha is just the initial probability of i times
         * the probability that i would emit observation O[0] */
        alpha = hmm->pi[i]*hmm->B[IDX2D(O[t], i, hmm->M)];
    } else if ( t > 0 ) {
        /* for any time t>0, alpha is partially defined as the probability 
         * that we came from any of the possible previous states. This is a
         * product of the probability that we were actually in a given state
         * multiplied by the probability that we would transition from that
         * state to i. The total probability is the sum of these values. */
        for( j=0; j<hmm->N; j++ ) {
            alpha += alphas[IDX2D(j, t-1, hmm->N)]*hmm->A[IDX2D(i, j, hmm->N)];
        }
        /* Finally, alpha is also dependent on the probability that state q_i
         * would actually emit observation O[t]. Multiply that by the sum we
         * already obtained. */
        alpha *= hmm->B[IDX2D(O[t], i, hmm->M)];
    }

    /* All done. Return the alpha value for state q_i at time t. */
    return alpha;
}


/* 
 * ===  FUNCTION  =============================================================
 *         Name:  beta_t_i
 *  Description:  Compute the beta value for state q_i at time t. The beta
 *                value is a counterpart to the alpha value. If the alpha
 *                gives us the probable states we have come from, then
 *                the beta value gives us the probable states we are going to.
 *
 *                Like the alpha value, beta can be computed recursively, but
 *                doing so is wasteful. It makes more sense in terms of time
 *                to just build a matrix and store everything there.
 *
 *       Inputs:  hmm   => The Hidden Markov Model
 *                betas => Matrix of beta values of size T*N. It is assumed 
 *                         that beta values at time t+1 have already been 
 *                         computed
 *                O     => A sequence of observations which we will use to 
 *                         compute probabilities
 *                t     => The time at which we want to compute beta for q_i
 *                i     => The state for which we want to compute beta
 *                T     => The length of the observation sequence                         
 *
 *      Outputs:  Beta value for state q_i at time t for observations sequence
 *                O given Hidden Markov Model hmm
 * ============================================================================
 */
static double
beta_t_i ( Hmm *hmm, Mat *betas, Observation *O, int t, int i, int T ) {
    int j;
    double beta;

    /* container for final beta value */
    beta = 0;

    if( t==T-1 ) {
        /* base case for time T-1 is beta = 1 */
        beta = 1;
    } else if( t >= 0 ) {
        /* For any time T < T-1, we compute the probability that we will move
         * from the current state to one of the future states given the 
         * probability that any possible future state would emit the 
         * observation O[t+1], the probability that we would actually 
         * transition from the current state to that future state and the 
         * probability that the possible future state will later transition
         * to another desired future state */
        for( j=0; j<hmm->N; j++ ) {
            beta += hmm->A[IDX2D(j,i,hmm->N)]*hmm->B[IDX2D(O[t+1],j,hmm->M)]
                    * betas[IDX2D(j,t+1,hmm->N)];
        }
    }
    
    /* return final beta value */
    return beta;
}

static void
compute_gammas ( Hmm *hmm, Mat *gammas, Mat *digammas, Mat *alphas, 
        Mat *betas, Observation *O, int T ) {
    int t, i, j;
    double denom;

    for( t=0; t<T-1; t++ ) {
        denom = 0;
        for( i=0; i<hmm->N; i++ ) {
            for( j=0; j<hmm->N; j++ ) {
                denom += alphas[IDX2D( i, t, hmm->N )] 
                    * hmm->A[IDX2D( j, i, hmm->N )]
                    * hmm->B[IDX2D( O[t+1], j, hmm->M )]
                    * betas[IDX2D( j, t+1, hmm->N )];
            }
        }
            
        for( i=0; i<hmm->N; i++ ) {
            gammas[IDX2D(i, t, hmm->N)] = 0;
            for( j=0; j<hmm->N; j++ ) {
                
                digammas[IDX3D(j,i,t,hmm->N,hmm->N)] = 
                    alphas[IDX2D( i, t, hmm->N )] 
                    * hmm->A[IDX2D( j, i, hmm->N )]
                    * hmm->B[IDX2D( O[t+1], j, hmm->M )]
                    * betas[IDX2D( j, t+1, hmm->N )] / denom;
                
                gammas[IDX2D(i, t, hmm->N)] 
                    += digammas[IDX3D(j,i,t,hmm->N,hmm->N)];
            }
        }
    }
    
    denom = 0;
    for( i=0; i<hmm->N; i++ ) {
        denom += alphas[IDX2D(i,t,hmm->N)];
    }

    for( i=0; i<hmm->N; i++ ) {
        gammas[IDX2D(i,t,hmm->N)] = alphas[IDX2D(i,t,hmm->N)]/denom;
    }
}

/* 
 * ===  FUNCTION  =============================================================
 *         Name:  forward_pass
 *  Description:  The forward pass is an analytical step in using an HMM which
 *                starts at the beginning of an observation sequence and 
 *                works its way towards the end (hence the name forward pass).
 *
 *                The forward pass builds a probability matrix using a set of
 *                observations and a Hidden Markov Model. This matrix by itself
 *                can be used to determine the probability that we will observe
 *                the input observation sequence given the HMM. 
 *
 *                Over long sequences, the computed alphas can quickly tend 
 *                towards zero. In order to protect against underflow, we 
 *                normalize the values of the alphas at each time slice t. The 
 *                coefficient which we use to perform the normalization is 
 *                stored in the input matrix c. It can be shown that the 
 *                probability of observing the sequence up to time t is the 
 *                reciprocal of the product of these coefficients from t = 0
 *                up to the desired time.
 *
 *       Inputs:  hmm    => The Hidden Markov Model
 *                alphas => A matrix which will be used to store the computed
 *                          alphas. It should be N*T in size
 *                c      => A vector which will store the coefficients. It
 *                          should be of length T
 *                O      => An observation sequence of events
 *                T      => The length of the observation sequence
 * ============================================================================
 */
static void
forward_pass ( Hmm *hmm, Mat *alphas, Mat *c, Observation *O, int T ) {
    int t, i;
    
    /* starting at the beginning of time, compute alpha values based on 
     * observation sequence O */
    for( t=0; t<T; t++ ) {
        /* c[t] is sum of alpha values at time t. We'll use this to normalize
         * the probabilities. Also very useful further down the line. It can 
         * be shown that expressions  written in terms of c instead of alpha 
         * and beta remain exact and are protected against underflow */
        c[t] = 0;
        
        for( i=0; i<hmm->N; i++ ) {
            /* compute the alpha for state q_i at time t */
            alphas[IDX2D(i, t, hmm->N)] = alpha_t_i( hmm, alphas, O, t, i );
            /* add new alpha to running total for time t */
            c[t] += alphas[IDX2D( i, t, hmm->N)];
        }
        
        /* normalize alpha values at time t by coefficient c[t] */
        c[t] = (c[t] > 0)? 1/c[t] : 0;
        mat_scale_row( alphas, t, hmm->N, c[t] );
    }    
}


/* 
 * ===  FUNCTION  =============================================================
 *         Name:  backward_pass
 *  Description:  The backward pass is analogous to the forward pass except
 *                that it starts at the end of the observation sequence and
 *                works backwards. Like the forward pass, we need to be 
 *                careful about underflow. In order to account for this, we
 *                scale the beta values by the same coefficient that was 
 *                computed and used during the forward pass.
 *
 *                The betas computed during the backward pass can be used in
 *                conjunction with the alphas to compute gammas which yield
 *                the most likely sequence of underlying states give an
 *                observed sequence of events.
 *
 *       Inputs:  hmm   => Hidden Markov Model
 *                betas => Matrix which we will populate with beta values
 *                c     => Coefficients obtained during forward pass
 *                O     => Observation sequence of events
 *                T     => Length of the observation sequence
 * ============================================================================
 */
static void
backward_pass ( Hmm *hmm, Mat *betas, Mat *c, Observation *O, int T ) {
    int i, t;

    /* Start at the end of time and work backwards */
    for( t=T-1; t>=0; t-- ) {
        /* Compute all betas at time t */
        for( i=0; i<hmm->N; i++ ) {
           betas[IDX2D(i,t,hmm->N)] = beta_t_i(hmm, betas, O, t, i, T);
            
           /* Scale the beta at t by the same coefficient that was used during 
            * the alpha pass */
           betas[IDX2D(i,t,hmm->N)] *= c[t];
        }
    }
}

static void
reestimate_pi ( Hmm *hmm, Mat *gammas ) {
    int i, t;
    t = 0;
    for( i=0; i<hmm->N; i++ ) {
        hmm->pi[i] = gammas[IDX2D(i,t,hmm->N)];
    }
}

static void
reestimate_A ( Hmm *hmm, Mat *gammas, Mat *digammas, int T ) {
    int i,j,t;
    double denom, numer;

    for( i=0; i<hmm->N; i++ ) {
        for( j=0; j<hmm->N; j++ ) {
            denom = numer = 0;
            for( t=0; t<T-1; t++ ) {
                numer += digammas[IDX3D(j,i,t,hmm->N,hmm->N)];
                denom += gammas[IDX2D(i,t,hmm->N)];
            }
            hmm->A[IDX2D(j,i,hmm->N)] = numer/denom;
        }
    }
}

static void
reestimate_B ( Hmm *hmm, Mat *gammas, Observation *O, int T ) {
    int i,j,t;
    double denom, numer;
    for( i=0; i<hmm->N; i++ ) {
        for( j=0; j<hmm->M; j++ ) {
            numer = denom = 0;
            for( t=0; t<T; t++ ) {
                if(O[t] == j) {
                    numer += gammas[IDX2D(i,t,hmm->N)];
                }
                denom += gammas[IDX2D(i,t,hmm->N)];
            }
            hmm->B[IDX2D(j,i,hmm->M)] = numer/denom;
        }
    }
}

/* 
 * ===  FUNCTION  =============================================================
 *         Name:  hmm_new
 *  Description:  Allocate memory for a new Hidden Markov Model struct and 
 *                initialize it as appropriate.
 *
 *                This function allocates the entire HMM on the heap and hence
 *                should be followed by a corresponding call to hmm_free when
 *                the program is finished using the struct.
 *
 *       Inputs:  N => The number of possible states the model can be in
 *                M => The number of possible observations the model might see
 *
 *      Outputs:  A pointer to a newly allocated HMM struct or NULL on 
 *                allocation failure. 
 * ============================================================================
 */
Hmm*
hmm_new ( int N, int M ) {
    Hmm* hmm;
    Mat *A, *B, *pi;

    /* allocate memory for the new hmm */
    /* remember that malloc returns NULL on failed allocation */
    hmm = malloc( sizeof( Hmm ) );

    /* if hmm allocation went well, start initializing */
    if( hmm ) {
        /* allocate memory for all three matrics */
        A  = malloc( sizeof( Mat ) * N * N );
        B  = malloc( sizeof( Mat ) * N * M );
        pi = malloc( sizeof( Mat ) * N );
        
        /* don't worry about failed allocations yet */
        /* hmm_init will just point the hmm at NULLs if there was a problem */
        hmm_init( hmm, N, M, A, B, pi );
        
        /* use the fact that hmm is pointing to A, B and pi to free all */
        /* allocated memory in the event of an allocation failure       */
        if( !A || !B || !pi ) {
            hmm_free( hmm );
            hmm = NULL;
        }
    }

    /* return hmm or NULL if anything went wrong */
    return hmm;
}

/* 
 * ===  FUNCTION  =============================================================
 *         Name:  hmm_init
 *  Description:  Initialize the Hidden Markov Model using the information
 *                passed as arguments.
 *
 *                This function will do a check to ensure that hmm is not
 *                NULL before it begins to initialize the struct, so it should
 *                be reasonably safe if NULL memory accidentally gets passed
 *                as input. The function will simply return without doing
 *                anything.
 *            
 *       Inputs:  hmm => The HMM to be initialized
 *                N   => the number of possible states in the model
 *                M   => the number of symbols which the model may observe
 *                A   => state transition matrix of size N * N
 *                B   => observation probability matrix of size N * M
 *                pi  => initial state distribution vector of size N
 * ============================================================================
 */
void
hmm_init ( Hmm *hmm, int N, int M, Mat *A, Mat *B, Mat *pi ) {

    /* check to see if we have been passed a valid hmm */
    if( hmm ) {
        /* if so, initialize */
        hmm->N = N;
        hmm->M = M;
        
        /* A, B amd pi could be NULL. Don't really care right now. It's the 
         * calling function's responsibility to sort that out */
        hmm->A  = A;
        hmm->B  = B;
        hmm->pi = pi;
    }
}

void
hmm_prepare_train ( Hmm *hmm ) {
    int i;
    for( i=0; i<hmm->N; i++ ) {
        mat_init_row_train( hmm->A, i, hmm->N );
        mat_init_row_train( hmm->B, i, hmm->M );
    }
    
    mat_init_row_train( hmm->pi, 0, hmm->N );
}

double 
hmm_pr_obs ( Hmm *hmm, Observation *O, int T ) {
    int i;
    double p;
    Mat *mem, *alphas, *c;

    /* allocate enough memory for all matrices and ensure malloc went okay */
    mem = malloc( sizeof( Mat ) * ( T * (hmm->N + 1) ) );
    if(!mem) {
        return -1;
    }

    /* split allocated memory up into alphas and coefficients */
    alphas = mem; 
    c = mem + T*hmm->N;

    /* Do the forward pass algorithm */
    forward_pass( hmm, alphas, c, O, T );

    /* Compute product of coefficients. It can be shown that the reciprocal
     * of this value is equal to the probability of the observed sequence. */
    p = c[0];
    for( i=1; i<T; i++ ) {
        p *= c[i];
    }
    
    /* free the matrices */
    free( mem );

    /* Return the reciprocal. As an alternative, we could just return p which
     * would increase in value as O became less probable. This would allow us
     * to test the relative probability of events without worrying about the
     * value getting too small for the computer to store accurately. In fact,
     * during the training phase we use the log of this probability to test
     * for convergence. So in many practical ways it makes a lot more sense to 
     * not return 1/p.
     *
     * Ternary operator prevents divide by zero error. */
    return (p)? 1/p : 0;
}

void
hmm_obs_states ( Hmm *hmm, Observation *O, State *X, int T ) {
    int i, t;
    
    Mat *mem, *alphas, *betas, *gammas, *digammas, *c;

    mem = malloc(sizeof(Mat) * (3*hmm->N*T + hmm->N*hmm->N*(T-1) + T));
    
    if(!mem) {
        return;
    }

    alphas = mem;
    betas  = alphas + hmm->N*T;
    gammas = betas + hmm->N*T;
    digammas = gammas + hmm->N*T;
    c = digammas + hmm->N*hmm->N*(T-1);

    forward_pass( hmm, alphas, c, O, T );
    backward_pass( hmm, betas, c, O, T );
    compute_gammas( hmm, gammas, digammas, alphas, betas, O, T );

    for( t=0; t<T; t++ ) {
        X[t] = 0;
        for( i=1; i<hmm->N; i++ ) {
            if( gammas[IDX2D(i, t, hmm->N)]>gammas[IDX2D(X[t], t, hmm->N)] ) {
                X[t] = i;
            }
        }
    }

    free(mem);
}

void
hmm_fit( Hmm *hmm, Observation *O, int T, int max_iters ) {
    int i;
    double log_prob, old_log_prob;
    Mat *mem, *alphas, *betas, *gammas, *digammas, *c;

    mem = malloc(sizeof(Mat) * (3*hmm->N*T + hmm->N*hmm->N*(T-1) + T));
    
    if(!mem) {
        return;
    }

    alphas = mem;
    betas  = alphas + hmm->N*T;
    gammas = betas + hmm->N*T;
    digammas = gammas + hmm->N*T;
    c = digammas + hmm->N*hmm->N*(T-1);

    hmm_prepare_train( hmm );

    log_prob = -DBL_MAX;
    old_log_prob = -DBL_MAX;

    for( i=0; i<max_iters && log_prob >= old_log_prob; i++ ) {
        old_log_prob = log_prob;
        
        forward_pass( hmm, alphas, c, O, T );
        backward_pass( hmm, betas, c, O, T );
        compute_gammas( hmm, gammas, digammas, alphas, betas, O, T );
        
        reestimate_pi( hmm, gammas );
        reestimate_A ( hmm, gammas, digammas, T );
        reestimate_B ( hmm, gammas, O, T );
        
        log_prob = log_probability( c, T );
    }

    free(mem);
}

/* 
 * ===  FUNCTION  =============================================================
 *         Name:  hmm_free
 *  Description:  Release any memory allocated to the input Hidden Markov 
 *                Model. Will do a check for NULLs before attempting to free,
 *                so it is safe to pass NULL input.
 *
 *       Inputs:  hmm => The HMM to be freed
 * ============================================================================
 */
void
hmm_free ( Hmm *hmm ) {

    /* only free memory that has been allocated */
    if(hmm) {
        if( hmm->A ) {
            free( hmm->A );
        }
        
        if( hmm->B ) {
            free( hmm->B );
        }
        
        if( hmm->pi ) {
            free( hmm->pi );
        }
        
        free(hmm);
    }
}
