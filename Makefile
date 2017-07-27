OBJS = $(wildcard src/*.c)

CC = clang

COMPILER_FLAGS = -Wall -g

LINKER_FLAGS = -lm

BINARY = hmm

all : $(OBJS)
	$(CC) $(COMPILER_FLAGS) $(OBJS) -o $(BINARY) $(LINKER_FLAGS)
