#ifndef H_PLOT
#define H_PLOT


#include <GL/gl.h>
#include <GL/freeglut.h>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>
#include "cusnn.cuh"


/* LAYER COLORS */
class Layer_colors {
public:

    int cnt_kernels;
    int *h_kernels;
    int *h_selected;
    float *h_colors_r;
    float *h_colors_g;
    float *h_colors_b;

    /* FUNCTIONS */
    Layer_colors(int cnt_kernel, std::vector<int> kernels);
    ~Layer_colors();
};


/* KERNEL WEIGHTS */
class KernelWeights {
public:

    GLint WindowID;

    /* FUNCTIONS */
    KernelWeights(int idx, int l, int d);
};


/* INTERNAL ACTIVITY */
class NetActivity {
public:

    GLint WindowID;

    /* FUNCTIONS */
    NetActivity(int idx);
};


/* POSTSYNAPTIC TRACE */
class PostTrace {
public:

    GLint WindowID;

    /* FUNCTIONS */
    PostTrace(int idx);
};


/* PLOTTER CLASS */
class plotterGL {
public:

    int cnt_snapshots;
    int cnt_posttrace_window;
    int cnt_kernel_window;
    int cnt_internal_window;
    bool enable_snapshot;
    std::string snapshots_dir;

    Layer_colors** h_layer_colors;
    PostTrace** h_posttrace_window;
    KernelWeights** h_kernel_window;
    NetActivity** h_internal_window;

    plotterGL(Network *SNN, std::vector<int> kernels, std::string snapshots_dir);
    ~plotterGL();

    virtual void update(Network *SNN);
    virtual void display_input();
    virtual void display_kernels();
    virtual void display_internal();
    virtual void display_posttrace();

    virtual void keyboard_input(int state);
    virtual void timer_kernel();
    virtual void keyboard_kernel(int state);
    virtual void timer_internal();
    virtual void keyboard_internal(int state);
    virtual void timer_posttrace();
    virtual void keyboard_posttrace(int state);

    void addIntWindow();
    void addKernelWindow(int l, int d);
    void addPostTraceWindow();
    void addLayerColors(int layer, int cnt_kernel, std::vector<int> kernels);
};


// additional functions
plotterGL *plotter_init(Network *SNN, std::vector<int> kernels, std::string snapshots_dir);
void save_image(std::string filename);
void PPM_writer(unsigned char *in, const char *name, int dimx, int dimy);


#endif