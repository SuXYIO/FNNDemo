CC = gcc
CFLAGS = -Wall
CDBFLAGS = -g

TARG = LR.out
DBTARG = LR.db.out
TARGS = $(shell find . -name '*.out')

DEP = $(shell find . -name '*.h')
CCODE = $(shell find . -name '*.c')
DSYM = $(shell find . -name '*.dSYM')
OBJ = $(CCODE:%.c=%.o)

$(TARG): $(OBJ)
	$(CC) $(CFLAGS) -o $(TARG) $(OBJ)

%.o: %.c $(DEP)
	$(CC) $(CFLAGS) -c $< -o $@

debug: $(CCODE) $(DEP)
	$(CC) $(CDBFLAGS) -o $(DBTARG) $(CCODE)

clean: 
	rm -rf $(TARGS) $(OBJ) $(DSYM)
