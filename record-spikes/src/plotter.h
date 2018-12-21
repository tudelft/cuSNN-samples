#ifndef H_PLOTTER
#define H_PLOTTER

#ifdef OPENGL
#include <GL/freeglut.h>
#endif
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include "cusnn.cuh"


/* LAYER COLORS */
class Colors {
public:

    int cnt_kernels;
    int *h_kernels;
    int *h_selected;
    float *h_colors_r;
    float *h_colors_g;
    float *h_colors_b;

    /* FUNCTIONS */
    Colors(int cnt_kernel, std::vector<int> kernels);
    ~Colors();
};


/* KERNEL WEIGHTS */
class Weights {
public:

    #ifdef OPENGL
    GLint WindowID;
    #endif

    /* FUNCTIONS */
    Weights(Network *SNN, int idx, int l, int d);
};


/* INTERNAL ACTIVITY */
class Spikes {
public:

    #ifdef OPENGL
    GLint WindowID;
    #endif

    /* FUNCTIONS */
    Spikes(Network *SNN, int idx);
};


/* POSTSYNAPTIC TRACE */
class Trace {
public:

    #ifdef OPENGL
    GLint WindowID;
    #endif

    /* FUNCTIONS */
    Trace(Network *SNN, int idx);
};


/* PLOTTER CLASS */
class plotterGL {
public:

    Network *SNN;
    int sim_step;

    int cnt_snapshots;
    int cnt_trace_windows;
    int cnt_weights_windows;
    int cnt_spikes_windows;
    bool enable_snapshot;
    std::string snapshots_dir;

    Colors** h_colors;
    Trace** h_trace;
    Weights** h_weights;
    Spikes** h_spikes;

    plotterGL(Network *SNN, std::vector<int> kernels, std::string snapshots_dir);
    ~plotterGL();

    static void display();
    static void keyboard(unsigned char key, int x, int y);
    static void timerEvent(int value);

    void update(Network *SNN, int sim_step);
    void display_input();
    void display_weights();
    void display_spikes();
    void display_trace();

    void keyboard_input(int state);
    void timer_weights();
    void keyboard_weights(int state);
    void timer_spikes();
    void keyboard_spikes(int state);
    void timer_trace();
    void keyboard_trace(int state);

    void add_spikes_window(Network *SNN);
    void add_weights_window(Network *SNN, int l, int d);
    void add_trace_window(Network *SNN);
    void add_colors(int layer, int cnt_kernel, std::vector<int> kernels);
};


// additional functions
plotterGL *plotter_init(Network *SNN, std::vector<int> kernels, std::string snapshots_dir);
void save_image(std::string filename);
void ppm_writer(unsigned char *in, const char *name, int dimx, int dimy);


#endif