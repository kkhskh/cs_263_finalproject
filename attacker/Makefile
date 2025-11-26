# attacker/Makefile

CC = gcc
CFLAGS = -O2 -Wall -march=native

all: flush_reload

flush_reload: flush_reload.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f flush_reload
