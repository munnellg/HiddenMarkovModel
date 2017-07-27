/*
 * ============================================================================
 *
 *       Filename:  hmm.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  07/07/17 12:40:34
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Gary Munnelly (gm), munnellg@tcd.ie
 *        Company:  Adapt Centre, Trinity College Dublin
 *
 * ============================================================================
 */


#ifndef  HMM_INC
#define  HMM_INC

#include <stdint.h>

/* useful little macro for indexing 1D array as if it were 2D of width w */
#define IDX2D(x,y,w) ((x)+(y)*(w))

/* macro for indexing 1D array as if it were 3D of width w and height h */
#define IDX3D(x,y,z,w,h) ((x)+(w)*((y)+(z)*(h)))

typedef uint32_t Observation; /* an observed output from the HMM */
typedef uint32_t State;       /* a possible state which the HMM may have */
typedef double   Mat;         /* used for creating probability matrics */ 

/* 
 * ===  STRUCT  ===============================================================
 *         Name:  Hmm
 *  Description:  Struct for maintaining the state of a Hidden Markov Model.
 *
 *                An HMM is described by three properties - the state 
 *                transition probability matrix A, the observation probability 
 *                matrix B and the initial state distribution vector pi.
 *
 *                A gives us the probability that we will transition from any
 *                given state q_i to any other possible state q_j in the model.
 *                A is right stochastic, meaning that each row should sum to 1.
 *
 *                B is the probability that we will see observation o_t if we
 *                are in state q_i at time t. It is also right stochastic.
 *
 *                Finally, pi gives the probability that we are in any 
 *                particular state q_i at time t=0.
 *
 *                In this representation, I have also included N and M, which
 *                represent the number of states the model can be in and the
 *                number of classes of observations which we may observe
 *                respectively. These values give us the dimensions of A, B and
 *                pi.
 *
 *                Notation for HMM properties is taken from "A Revealing 
 *                Introduction to Hidden Markov Models" by Mark Stamp. I have
 *                based my implementation heavily on his examples.
 * ============================================================================
 */
typedef struct Hmm {
    int  N;  /* number of states in the model */
    int  M;  /* number of observation symbols */
    Mat *A;  /* state transition probability matrix of size N*N */
    Mat *B;  /* observation probability matrix of size N*M */
    Mat *pi; /* initial state distribution vector of length N */
} Hmm;

/* 
 * ===  FUNCTION  =============================================================
 *         Name:  hmm_new
 *  Description:  Allocate memory for a new Hidden Markov Model struct and 
 *                initialize it as appropriate.
 *
 *                This function should allocate the entire HMM on the heap and 
 *                hence should be followed by a corresponding call to hmm_free 
 *                when the program is finished using the struct.
 *
 *       Inputs:  N => The number of possible states the model can be in
 *                M => The number of possible observations the model might see
 *
 *      Outputs:  A pointer to a newly allocated HMM struct or NULL on 
 *                allocation failure.
 * ============================================================================
 */
extern Hmm* hmm_new ( int N, int M );

/* 
 * ===  FUNCTION  =============================================================
 *         Name:  hmm_init
 *  Description:  Initialize the Hidden Markov Model using the information
 *                passed as arguments. Should basically just point the HMM
 *                struct at the various other values passed as input.
 *
 *                I don't like the idea of an init function dynamically 
 *                allocating memory when there is a chance that the input 
 *                struct is allocated on the stack. It leaves too much room
 *                for memory leaks. Implementations of this function should
 *                assume the calling function has taken the time to set aside 
 *                memory for the matrices and will also handle the release of 
 *                any dynamicially allocated memory.
 *
 *       Inputs:  hmm => The HMM to be initialized
 *                N   => the number of possible states in the model
 *                M   => the number of symbols which the model may observe
 *                A   => state transition matrix of size N * N
 *                B   => observation probability matrix of size N * M
 *                pi  => initial state distribution vector of size N*
 * ============================================================================
 */
extern void hmm_init ( Hmm *hmm, int N, int M, Mat *A, Mat *B, Mat *pi );

extern void hmm_prepare_train ( Hmm *hmm );

/* 
 * ===  FUNCTION  =============================================================
 *         Name:  hmm_pr_obs
 *  Description:  Gives the probability that we will observe the sequence of
 *                observations O given the parameters of the Hidden Markov
 *                Model. This is a difficult thing to manage as P( O | Hmm )
 *                tends towards zero very quickly. There are ways around this,
 *                one of which is discussed in the implementation of this 
 *                function in hmm.c. However if we follow these methods, then
 *                this function no longer returns a probability.
 *
 *       Inputs:  hmm => The Hidden Markov Model with respect to which we want
 *                       to compute the probability of O
 *                O   => A sequence of observations
 *                T   => The length of the sequence of observations
 * ============================================================================
 */
extern double hmm_pr_obs ( Hmm *hmm, Observation *O, int T );

extern void hmm_obs_states ( Hmm *hmm, Observation *O, State *X, int T );

extern void hmm_fit ( Hmm *hmm, Observation *O, int T, int max_iters );

/* 
 * ===  FUNCTION  =============================================================
 *         Name:  hmm_free
 *  Description:  Release any memory allocated to the input Hidden Markov 
 *                Model. 
 *
 *       Inputs:  hmm => The HMM to be freed
 * ============================================================================
 */
extern void hmm_free ( Hmm *hmm );

#endif   /* ----- #ifndef HMM_INC  ----- */
