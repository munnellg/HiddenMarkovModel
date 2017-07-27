# Hidden Markov Models

An implementation of a simple Hidden Markov Model based on pseudo code provided
by [Mark Stamp][Stamp] in his paper "[A Revealing Introduction to Hidden 
Markov Models][Revealing]".

So far, the outputs of my program seem to more or less correlate with the 
expected outputs provided by Mark in his paper. I haven't done an in-depth
debugging and I'm sure there are numerous places where the code could
be improved, but for now it is provided as-is.

The main function doesn't really do anything useful or interesting. It just 
contains some badly written tests I wrote to make sure everything worked as I
wanted it to The most important thing to look at here is `src/hmm.h` and 
`src/hmm.c` which are derived from Mark's pseudocode
