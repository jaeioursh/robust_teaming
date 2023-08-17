
python_version_full := $(wordlist 2,4,$(subst ., ,$(shell python3 --version 2>&1)))
python_version_major := $(word 1,${python_version_full})
python_version_minor := $(word 2,${python_version_full})
python_version_patch := $(word 3,${python_version_full})


CC = gcc
PYVERSION=3.6
FLAGS = -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python3.${python_version_minor} -o

default: part0 compile clean full_make

part0:
	rm __init__.py
	echo ${python_version_full}

compile:
	echo 'start'
	#cython mod_funcs.pyx
	#${CC} ${FLAGS} mod_funcs.so mod_funcs.c

clean:
	echo -n > __init__.py
	#rm mod_funcs.c


	

video:
	ffmpeg -r 12 -i ims/test%d.png -c:v libx264 -vf fps=12 -pix_fmt yuv420p out.mp4

full_make:
	cd code && make
	

