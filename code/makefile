python_version_full := $(wordlist 2,4,$(subst ., ,$(shell python3 --version 2>&1)))
python_version_major := $(word 1,${python_version_full})
python_version_minor := $(word 2,${python_version_full})
python_version_patch := $(word 3,${python_version_full})


CC = gcc
PYVERSION=3.6
FLAGS = -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.${python_version_minor} -o



default: part0 part1 part2 part3 clean

part0:
	rm __init__.py

part1:
	cython agent_domain_2.pyx
	${CC} ${FLAGS} agent_domain_2.so agent_domain_2.c
      
part2:
	cython ccea_2.pyx
	${CC} ${FLAGS} ccea_2.so ccea_2.c
      
part3:
	cython reward_2.pyx
	${CC} ${FLAGS} reward_2.so reward_2.c

clean:
	rm *.c
	echo -n > __init__.py

	      
clear:
	rm *.so
