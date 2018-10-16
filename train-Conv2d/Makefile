.PHONY: build
.PHONY: clean

PLOTTER = true
SNAPSHOT = true

ifeq ($(PLOTTER),true)
    ifeq ($(SNAPSHOT),true)
        CFLAGS = -DOPENGL -DNPY --std=c++11 -L/usr/local/lib -lcuSNN -lcnpy -lz -lGL -lGLU -lX11 -lglut
    else
        CFLAGS = -DOPENGL --std=c++11 -L/usr/local/lib -lcuSNN -lGL -lGLU -lX11 -lglut
    endif
else ifeq ($(SNAPSHOT),true)
    CFLAGS = -DNPY --std=c++11 -L/usr/local/lib -lcuSNN -lcnpy -lz
else
    CFLAGS = --std=c++11 -L/usr/local/lib -lcuSNN
endif

BUILD = build/main
OBJ_DIR = obj
SRC_DIR = src

ifeq ($(PLOTTER),true)
    HEADERS = plotter.h data.h
    OBJECTS = plotter.o data.o main.o
else
    HEADERS = data.h
    OBJECTS = data.o main.o
endif

OBJS = $(patsubst %,$(OBJ_DIR)/%,$(OBJECTS))
HDRS = $(patsubst %,$(SRC_DIR)/%,$(HEADERS))

build: ${OBJS}
	nvcc ${OBJS} -o ${BUILD} $(CFLAGS)

clean:
	rm -f ${BUILD}
	rm -f ${OBJ_DIR}/*.o

${OBJ_DIR}/%.o: ${SRC_DIR}/%.cpp ${HDRS}
	nvcc $(CFLAGS) -c -o $@ $<
