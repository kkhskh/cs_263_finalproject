# attacker/Dockerfile
# Build this on the x86 VM later.

FROM gcc:13

WORKDIR /attacker

COPY flush_reload.c Makefile ./

RUN make

CMD ["./flush_reload", "500000"]
