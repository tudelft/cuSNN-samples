#include "plotter.h"


// initialize pointer to classes
plotterGL *plotter;
Network *SNN_aux;

// flag
bool plot = true;
bool plot_input     = true;
bool plot_kernels   = true;
bool plot_internal  = true;
bool plot_posttrace = false;

// windows ID
GLint WindowID1;

// window size
unsigned int window_width  = 200;
unsigned int window_height = 200;
int tostring_precision = 0;

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void timerEvent(int value);


/* Layer_colors */
// constructor
Layer_colors::Layer_colors(int cnt_kernel, std::vector<int> kernels) {

    // kernel colors
    this->h_colors_r = (float *) malloc(sizeof(float) * cnt_kernel);
    this->h_colors_g = (float *) malloc(sizeof(float) * cnt_kernel);
    this->h_colors_b = (float *) malloc(sizeof(float) * cnt_kernel);
    for (int i = 0; i < cnt_kernel; i++) {
        this->h_colors_r[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        this->h_colors_g[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        this->h_colors_b[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    // kernel for visualization
    this->cnt_kernels = (int) kernels.size();
    this->h_kernels = (int *) malloc(sizeof(int) * this->cnt_kernels);
    for (int i = 0; i < this->cnt_kernels; i++)
        this->h_kernels[i] = kernels.at((unsigned long) i);

    // selected kernels
    this->h_selected = (int *) malloc(sizeof(int) * cnt_kernel);
    for (int i = 0; i < cnt_kernel; i++) {
        this->h_selected[i] = -1;
        for (int j = 0; j < this->cnt_kernels; ++j) {
            if (i == this->h_kernels[j]) {
                this->h_selected[i] = j;
                break;
            }
        }
    }
}


// destructor
Layer_colors::~Layer_colors(){
    free(this->h_colors_r);
    free(this->h_colors_g);
    free(this->h_colors_b);
    free(this->h_kernels);
    free(this->h_selected);
}


// add layer data
void plotterGL::addLayerColors(int layer, int cnt_kernel, std::vector<int> kernelsl) {
    this->h_layer_colors[layer] = new Layer_colors(cnt_kernel, kernelsl);
}


/* POSTSYNAPTIC TRACE */
// constructor
PostTrace::PostTrace(int idx) {
    glutInitWindowPosition(window_width * idx, 125 + 2*window_height);
    glutInitWindowSize(window_width, window_height);
    std::string title = "Layer " + std::to_string(idx) + ": " + SNN_aux->h_layers[idx]->layer_type_str.c_str() +
            " -- Post-synaptic trace";
    const char* title_char = title.c_str();
    this->WindowID = glutCreateWindow(title_char);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
}


// destructor
PostTrace::~PostTrace(){}


// add window
void plotterGL::addPostTraceWindow() {
    this->h_posttrace_window[this->cnt_posttrace_window] = new PostTrace(this->cnt_posttrace_window);
    this->cnt_posttrace_window++;
}


/* KernelWeights */
// constructor
KernelWeights::KernelWeights(int idx, int l, int d) {

    int row = idx / 4;
    int col = idx % 4;
    glutInitWindowPosition(window_width * (col+1), 100 + (window_height + 25) * row);
    glutInitWindowSize(window_width, window_height);
    std::string title = "Layer " + std::to_string(l) + ": " + SNN_aux->h_layers[l]->layer_type_str.c_str() +
            " -- Conv. kernels (" + std::to_string(SNN_aux->h_layers[l]->h_delay_indices[d]) + "ms delay)";
    const char* title_char = title.c_str();
    this->WindowID = glutCreateWindow(title_char);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
}


// destructor
KernelWeights::~KernelWeights(){}


// add window
void plotterGL::addKernelWindow(int l, int d) {
    this->h_kernel_window[this->cnt_kernel_window] = new KernelWeights(this->cnt_kernel_window, l, d);
    this->cnt_kernel_window++;
}


/* INTERNAL ACTIVITY */
// constructor
NetActivity::NetActivity(int idx) {
    glutInitWindowPosition(0, 100 + (window_height + 25) * (idx + 1));
    glutInitWindowSize(window_width, window_height);
    std::string title = "Layer " + std::to_string(idx) + ": " + SNN_aux->h_layers[idx]->layer_type_str.c_str() +
            " -- Activity";
    const char* title_char = title.c_str();
    this->WindowID = glutCreateWindow(title_char);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
}


// destructor
NetActivity::~NetActivity(){}


// add window
void plotterGL::addIntWindow() {
    this->h_internal_window[this->cnt_internal_window] = new NetActivity(this->cnt_internal_window);
    this->cnt_internal_window++;
}


// initialize class
plotterGL *plotter_init(Network *SNN, std::vector<int> kernels, std::string snapshots_dir) {
    plotter = new plotterGL(SNN, kernels, snapshots_dir);
    return plotter;
}


// constructor
plotterGL::plotterGL(Network *SNN, std::vector<int> kernels, std::string snapshots_dir) {

    SNN_aux = SNN;
    this->cnt_snapshots = 0;
    this->snapshots_dir = snapshots_dir;

    // kernel colors
    this->h_layer_colors = (Layer_colors **) malloc(sizeof(Layer_colors*) * SNN->cnt_layers);
    for (int l = 0; l < SNN->cnt_layers; l++)
        this->addLayerColors(l, SNN->h_layers[l]->cnt_kernels, kernels);

    // initialize GL
    int argc = 1;
    char *argv[1] = {(char*)""};
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA);

    // initialize window 1
    if (plot_input) {
        glutInitWindowPosition(0, 100);
        glutInitWindowSize(window_width, window_height);
        WindowID1 = glutCreateWindow("Input");
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    // initialize windows for kernels
    if (plot_kernels) {
        int num_windows = 0;
        for (int l = 0; l < SNN->cnt_layers; l++){
            for (int d = 0; d < SNN->h_layers[l]->num_delays_synapse; d++)
                num_windows++;
        }
        this->cnt_kernel_window = 0;
        this->h_kernel_window = (KernelWeights **) malloc(sizeof(KernelWeights*) * num_windows);
        for (int l = 0; l < SNN->cnt_layers; l++){
            for (int d = 0; d < SNN->h_layers[l]->num_delays_synapse; d++) {
                this->addKernelWindow(l, d);
            }
        }
    }

    // initialize windows for internal activity
    if (plot_internal) {
        this->cnt_internal_window = 0;
        this->h_internal_window = (NetActivity **) malloc(sizeof(NetActivity*) * SNN->cnt_layers);
        for (int l = 0; l < SNN->cnt_layers; l++)
            this->addIntWindow();
    }

    // initialize windows for post-synaptic trace
    if (plot_posttrace) {
        this->cnt_posttrace_window = 0;
        this->h_posttrace_window = (PostTrace **) malloc(sizeof(PostTrace*) * SNN->cnt_layers);
        for (int l = 0; l < SNN->cnt_layers; l++)
            this->addPostTraceWindow();
    }

    // activate alpha
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);

    // register callbacks
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutTimerFunc(0, timerEvent, 0);
}


// destructor
plotterGL::~plotterGL(){
    free(this->h_internal_window);
    free(this->h_kernel_window);
    free(this->h_layer_colors);
}


// update the plot
void plotterGL::update(Network *SNN) {
    if (plot) {
        SNN_aux = SNN;
        glutMainLoopEvent();
    }
}


//calbacks
void display() {
    if (plot_input) plotter->display_input();
    if (plot_kernels) plotter->display_kernels();
    if (plot_internal) plotter->display_internal();
    if (plot_posttrace) plotter->display_posttrace();
}


//calbacks
void timerEvent(int value) {

    if (plot_input) {
        glutSetWindow(WindowID1);
        glutPostRedisplay();
    }
    if (plot_kernels) plotter->timer_kernel();
    if (plot_internal) plotter->timer_internal();
    if (plot_posttrace) plotter->timer_posttrace();
    glutTimerFunc(0, timerEvent, 0);
}


//calbacks
void keyboard(unsigned char key, int /*x*/, int /*y*/) {
    switch (key) {

        // hide all the windows
        case (27):
            if (plot) {
                if (plot_input) plotter->keyboard_input(27);
                if (plot_kernels) plotter->keyboard_kernel(27);
                if (plot_internal) plotter->keyboard_internal(27);
                if (plot_posttrace) plotter->keyboard_posttrace(27);
                plot = false;
            }
            return;

        // store one image per window
        case(32):
            if (plot) {
                if (plot_input) plotter->keyboard_input(32);
                if (plot_kernels) plotter->keyboard_kernel(32);
                if (plot_internal) plotter->keyboard_internal(32);
                if (plot_posttrace) plotter->keyboard_posttrace(32);
                plotter->cnt_snapshots++;
            }
            return;
    }
}


//calbacks
void plotterGL::display_input() {

    glutSetWindow(WindowID1);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    float alpha = 1.0;

    // input data
    for (int row = 0; row < SNN_aux->inp_size[1]; row++) {
        for (int col = 0; col < SNN_aux->inp_size[2]; col++) {
            int node_idx = col * SNN_aux->inp_size[1] + row;
            int ch0_idx = node_idx * SNN_aux->h_layers[0]->length_delay_inp;
            int ch1_idx = SNN_aux->h_layers[0]->inp_size[1] * SNN_aux->h_layers[0]->inp_size[2] *
                    SNN_aux->h_layers[0]->length_delay_inp + node_idx * SNN_aux->h_layers[0]->length_delay_inp;

            int col_idx = col - SNN_aux->inp_size[2] / 2;
            int row_Idx = SNN_aux->inp_size[1] / 2 - row - 1;
            glColor4f(0.5, 0.5, 0.5, alpha);
            if (SNN_aux->h_inputs[ch0_idx]) glColor4f(0.0, 0.0, 0.0, alpha); // OFF -> black
            if (SNN_aux->h_inputs[ch1_idx]) glColor4f(1.0, 1.0, 1.0, alpha); // ON -> white

            glRectf((GLfloat) (col_idx / (SNN_aux->inp_size[2] / 2.0)),
                    (GLfloat) (row_Idx / (SNN_aux->inp_size[1] / 2.0)),
                    (GLfloat) ((col_idx + 1) / (SNN_aux->inp_size[2] / 2.0)),
                    (GLfloat) ((row_Idx + 1) / (SNN_aux->inp_size[1] / 2.0)));
        }
    }
    glFlush();
}


//calbacks
void plotterGL::display_kernels() {

    int idx = 0;
    for (int l = 0; l < SNN_aux->cnt_layers; l++) {
        for (int d = 0; d < SNN_aux->h_layers[l]->num_delays_synapse; d++) {

            glutSetWindow(this->h_kernel_window[idx]->WindowID);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable(GL_BLEND);
            glClearColor(0.f, 0.f, 0.f, 1.f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            float rows_max = floor(sqrt((float) SNN_aux->h_layers[l]->cnt_kernels));
            if ((int) rows_max % 2 == 1 && rows_max != 1.f) rows_max--;
            float cols_max = (float) SNN_aux->h_layers[l]->cnt_kernels / rows_max;

            int x_offset = 1 + SNN_aux->h_layers[l]->rf_side + 1 + 1 + 2;
            int y_offset = 1 + SNN_aux->h_layers[l]->rf_side + 2;
            float x_scaler = (float) x_offset * cols_max / 2.f;
            float y_scaler = (float) y_offset * rows_max / 2.f;

            int col_kernel = 0;
            int row_kernel = 0;
            for (int k = 0; k < SNN_aux->h_layers[l]->cnt_kernels; k++) {

                // black box for kernel
                for (int rows = 0; rows < SNN_aux->h_layers[l]->rf_side - SNN_aux->h_layers[l]->rf_side_limits[0]; rows++) {
                    for (int cols = 0; cols < SNN_aux->h_layers[l]->rf_side - SNN_aux->h_layers[l]->rf_side_limits[1]; cols++) {

                        int col_idx = 1 + cols - (int) x_scaler + col_kernel * x_offset;
                        int row_idx = (int) y_scaler - 1 - rows - 1 - row_kernel * y_offset;
                        int syn_index = cols * SNN_aux->h_layers[l]->rf_side + rows;

                        float accum_weight = 0.f;
                        float red = 0.f, green = 0.f, blue = 0.f;
                        for (int ch = 0; ch < SNN_aux->h_layers[l]->kernel_channels; ch++) {

                            int syn_delay_index = ch * SNN_aux->h_layers[l]->rf_side * SNN_aux->h_layers[l]->rf_side *
                                    SNN_aux->h_layers[l]->num_delays_synapse +
                                    syn_index * SNN_aux->h_layers[l]->num_delays_synapse + d;
                            float weight = SNN_aux->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index];
                            if (weight > 1.f) weight = 1.f;
                            else if (weight < 0.f) weight = 0.f;

                            // Event-based-sensors specific
                            if (!l) {
                                if (SNN_aux->h_layers[l]->kernel_channels == 1) {
                                    if (!ch) {
                                        red += weight;
                                        green += weight;
                                        blue += weight;
                                    } else if (ch == 1) {
                                        red += weight;
                                        green += weight;
                                        blue += weight;
                                    }
                                } else {
                                    if (!ch) red += weight;
                                    else if (ch == 1) green += weight;
                                }
                            } else {
                                if (SNN_aux->h_layers[l]->kernel_channels == 1) {
                                    red += weight;
                                    green += weight;
                                    blue += weight;
                                } else {
                                    if (SNN_aux->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index] > 0.f &&
                                        SNN_aux->h_layers[l-1]->h_stdp_converged[ch]) {
                                        if (SNN_aux->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index] > 1.f)
                                            SNN_aux->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index] = 1.f;
                                        accum_weight +=
                                                SNN_aux->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index];
                                    }
                                }
                            }
                        }

                        if (SNN_aux->h_layers[l]->out_maps == 1 && l != 0) {
                            for (int ch = 0; ch < SNN_aux->h_layers[l]->kernel_channels; ch++) {

                                int syn_delay_index = ch *
                                        SNN_aux->h_layers[l]->rf_side * SNN_aux->h_layers[l]->rf_side *
                                        SNN_aux->h_layers[l]->num_delays_synapse +
                                        syn_index * SNN_aux->h_layers[l]->num_delays_synapse + d;

                                if (SNN_aux->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index] > 0.f &&
                                    SNN_aux->h_layers[l-1]->h_stdp_converged[ch]) {
                                    if (SNN_aux->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index] > 1.f)
                                        SNN_aux->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index] = 1.f;
                                    red += plotter->h_layer_colors[l-1]->h_colors_r[ch] *
                                            SNN_aux->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index] *
                                            SNN_aux->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index] /
                                            accum_weight;
                                    green += plotter->h_layer_colors[l-1]->h_colors_g[ch] *
                                            SNN_aux->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index] *
                                            SNN_aux->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index] /
                                            accum_weight;
                                    blue  += plotter->h_layer_colors[l-1]->h_colors_b[ch] *
                                            SNN_aux->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index] *
                                            SNN_aux->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index] /
                                            accum_weight;
                                }
                            }
                        }

                        glColor4f(red, green, blue, 1.f);
                        glRectf((GLfloat) (col_idx / x_scaler),
                                (GLfloat) (row_idx / y_scaler),
                                (GLfloat) ((col_idx + 1) / x_scaler),
                                (GLfloat) ((row_idx + 1) / y_scaler));
                    }
                }

                // box around the filter
                int col_begin = 1 - (int) x_scaler + col_kernel * x_offset;
                int col_end = 1 - (int) x_scaler + col_kernel * x_offset + SNN_aux->h_layers[l]->rf_side;
                int row_begin = (int) y_scaler - 1 - row_kernel * y_offset;
                int row_end = (int) y_scaler - 1 - row_kernel * y_offset - SNN_aux->h_layers[l]->rf_side;

                if (SNN_aux->h_layers[l]->h_stdp_converged[k]) glColor4f(0.f, 1.f, 0.f, 1.f);
                else glColor4f(1.f, 0.f, 0.f, 1.f);

                glBegin(GL_LINE_LOOP) ;
                glVertex3f((GLfloat) (col_begin / x_scaler),
                           (GLfloat) (row_begin / y_scaler), 0.f);
                glVertex3f((GLfloat) (col_end / x_scaler),
                           (GLfloat) (row_begin / y_scaler), 0.f);
                glVertex3f((GLfloat) (col_end / x_scaler),
                           (GLfloat) (row_end / y_scaler), 0.f);
                glVertex3f((GLfloat) (col_begin / x_scaler),
                           (GLfloat) (row_end / y_scaler), 0.f);
                glEnd();

                // color box
                if (this->h_layer_colors[l]->h_selected[k] != -1 && d == 0) {
                    glColor4f(this->h_layer_colors[l]->h_colors_r[this->h_layer_colors[l]->h_selected[k]],
                              this->h_layer_colors[l]->h_colors_g[this->h_layer_colors[l]->h_selected[k]],
                              this->h_layer_colors[l]->h_colors_b[this->h_layer_colors[l]->h_selected[k]], 1.f);
                    glRectf((GLfloat) ((col_end + 1) / x_scaler),
                            (GLfloat) (row_begin / y_scaler),
                            (GLfloat) ((col_end + 2) / x_scaler),
                            (GLfloat) (row_end / y_scaler));
                }

                row_kernel++;
                if (row_kernel >= rows_max) {
                    row_kernel = 0;
                    col_kernel++;
                }

                // display kernel name and STDP convergence data
                glColor3f(1.f, 1.f, 1.f);
                glRasterPos2f(col_begin / x_scaler, (row_end-1) / y_scaler);
                std::string stdp_convg;
                stdp_convg += std::string("k") + std::to_string(k);
                if (SNN_aux->h_layers[l]->enable_stdp && d == 0)
                    stdp_convg += std::string(" = ") +
                            std::to_string(SNN_aux->h_layers[l]->h_kernels[k]->stdp_objective_avg);
                int len, i;
                len = (int) stdp_convg.length();
                if (SNN_aux->h_layers[l]->enable_stdp) len -= tostring_precision;
                for (i = 0; i < len; i++)
                    glutBitmapCharacter(GLUT_BITMAP_8_BY_13, stdp_convg[i]);
            }
            idx++;
        }
    }
    glFlush();
}


//calbacks
void plotterGL::display_internal() {

    for (int l = 0; l < SNN_aux->cnt_layers; l++) {

        glutSetWindow(this->h_internal_window[l]->WindowID);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);

        float alpha = 1.f;
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (SNN_aux->h_layers[l]->strides != 0) {
            for (int rows = 0; rows < SNN_aux->h_layers[l]->out_size[1]; rows++) {
                for (int cols = 0; cols < SNN_aux->h_layers[l]->out_size[2]; cols++) {

                    int cnt = 0;
                    float red = 0.f, green = 0.f, blue = 0.f;
                    int node_index = cols * SNN_aux->h_layers[l]->out_size[1] + rows;
                    int begin = node_index * SNN_aux->h_layers[l]->length_delay_out;

                    for (int k = 0; k < this->h_layer_colors[l]->cnt_kernels; k++) {
                        int kernel = this->h_layer_colors[l]->h_kernels[k];
                        if (kernel >= SNN_aux->h_layers[l]->cnt_kernels)
                            continue;

                        // if neuron spikes
                        if (SNN_aux->h_layers[l]->h_kernels[kernel]->h_node_train[begin]) {
                            red += this->h_layer_colors[l]->h_colors_r[k];
                            green += this->h_layer_colors[l]->h_colors_g[k];
                            blue += this->h_layer_colors[l]->h_colors_b[k];
                            cnt++;
                        }
                    }
                    red   /= (float) cnt;
                    green /= (float) cnt;
                    blue  /= (float) cnt;

                    int cols_aux = cols - SNN_aux->h_layers[l]->out_size[2] / 2;
                    int rows_aux = SNN_aux->h_layers[l]->out_size[1] / 2 - rows - 1;
                    glColor4f(red, green, blue, alpha);
                    glRectf((GLfloat) (cols_aux / (SNN_aux->h_layers[l]->out_size[2] / 2.f)),
                            (GLfloat) (rows_aux / (SNN_aux->h_layers[l]->out_size[1] / 2.f)),
                            (GLfloat) ((cols_aux+1) / (SNN_aux->h_layers[l]->out_size[2] / 2.f)),
                            (GLfloat) ((rows_aux+1) / (SNN_aux->h_layers[l]->out_size[1] / 2.f)));
                }
            }
        } else {
            float rows_max = 0.f, cols_max = 0.f;
            rows_max = floor(sqrtf(SNN_aux->h_layers[l]->cnt_kernels));
            if ((int) rows_max % 2 == 1 && rows_max != 1.f) rows_max--;
            cols_max = (float) SNN_aux->h_layers[l]->cnt_kernels / rows_max;

            for (int k = 0; k < this->h_layer_colors[l]->cnt_kernels; k++) {
                if (this->h_layer_colors[l]->h_kernels[k] >= SNN_aux->h_layers[l]->cnt_kernels)
                    continue;

                // if neuron spikes
                if (SNN_aux->h_layers[l]->h_kernels[this->h_layer_colors[l]->h_kernels[k]]->h_node_train[0]) {
                    int row = k % (int) rows_max;
                    int col = k / (int) rows_max;
                    float cols_aux = col - cols_max / 2;
                    float rows_aux = rows_max / 2 - row - 1;

                    glColor4f(this->h_layer_colors[l]->h_colors_r[k],
                              this->h_layer_colors[l]->h_colors_g[k],
                              this->h_layer_colors[l]->h_colors_b[k], alpha);
                    glRectf((GLfloat) (cols_aux / (cols_max / 2.f)),
                            (GLfloat) (rows_aux / (rows_max / 2.f)),
                            (GLfloat) ((cols_aux+1) / (cols_max / 2.f)),
                            (GLfloat) ((rows_aux+1) / (rows_max / 2.f)));
                }
            }
        }
    }
    glFlush();
}


//calbacks
void plotterGL::display_posttrace() {

    for (int l = 0; l < SNN_aux->cnt_layers; l++) {

        glutSetWindow(this->h_posttrace_window[l]->WindowID);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);

        float alpha = 1.f;
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        for (int rows = 0; rows < SNN_aux->h_layers[l]->out_size[1]; rows++) {
            for (int cols = 0; cols < SNN_aux->h_layers[l]->out_size[2]; cols++) {

                int node_index = cols * SNN_aux->h_layers[l]->out_size[1] + rows;
                int cols_aux = cols - SNN_aux->h_layers[l]->out_size[2] / 2;
                int rows_aux = SNN_aux->h_layers[l]->out_size[1] / 2 - rows - 1;

                int aux_cnt = 0;
                float red = 0.f, green = 0.f, blue = 0.f;
                for (int k = 0; k < this->h_layer_colors[l]->cnt_kernels; k++) {
                    int kernel = this->h_layer_colors[l]->h_kernels[k];
                    if (kernel >= SNN_aux->h_layers[l]->cnt_kernels)
                        continue;

                    float value = SNN_aux->h_layers[l]->h_kernels[kernel]->h_node_posttrace[node_index];
                    value *= SNN_aux->h_layers[l]->neuron_decay / SNN_aux->h_sim_step[0];
                    red   += this->h_layer_colors[l]->h_colors_r[k] * value;
                    green += this->h_layer_colors[l]->h_colors_g[k] * value;
                    blue  += this->h_layer_colors[l]->h_colors_b[k] * value;
                    aux_cnt++;
                }
                red   /= (float) aux_cnt;
                green /= (float) aux_cnt;
                blue  /= (float) aux_cnt;

                glColor4f(red, green, blue, alpha);
                glRectf((GLfloat) (cols_aux / (SNN_aux->h_layers[l]->out_size[2] / 2.f)),
                        (GLfloat) (rows_aux / (SNN_aux->h_layers[l]->out_size[1] / 2.f)),
                        (GLfloat) ((cols_aux+1) / (SNN_aux->h_layers[l]->out_size[2] / 2.f)),
                        (GLfloat) ((rows_aux+1) / (SNN_aux->h_layers[l]->out_size[1] / 2.f)));
            }
        }
    }
    glFlush();
}


//calbacks
void plotterGL::keyboard_input(int state) {

    glutSetWindow(WindowID1);

    // delete window
    if (state == 27) {
        glutHideWindow();
        glutDestroyWindow(glutGetWindow());

    // store window
    } else if (state == 32) {
        std::string filename;
        filename += this->snapshots_dir + "/input_" + std::to_string(this->cnt_snapshots) + ".ppm";
        save_image(filename);
    }
}


//calbacks
void plotterGL::timer_kernel() {
    int idx = 0;
    for (int l = 0; l < SNN_aux->cnt_layers; l++) {
        for (int d = 0; d < SNN_aux->h_layers[l]->num_delays_synapse; d++) {
            glutSetWindow(this->h_kernel_window[idx]->WindowID);
            glutPostRedisplay();
            idx++;
        }
    }
}


//calbacks
void plotterGL::keyboard_kernel(int state) {
    int idx = 0;
    for (int l = 0; l < SNN_aux->cnt_layers; l++) {
        for (int d = 0; d < SNN_aux->h_layers[l]->num_delays_synapse; d++) {
            glutSetWindow(this->h_kernel_window[idx]->WindowID);

            // delete window
            if (state == 27) {
                glutHideWindow();
                glutDestroyWindow(glutGetWindow());

            // store window
            } else if (state == 32) {
                std::string filename;
                filename += this->snapshots_dir + "/kernel_l" + std::to_string(l) + "_d" + std::to_string(d) + "_" +
                        std::to_string(this->cnt_snapshots) + ".ppm";
                save_image(filename);
            }
            idx++;
        }
    }
}


//calbacks
void plotterGL::timer_internal() {
    for (int l = 0; l < SNN_aux->cnt_layers; l++) {
        glutSetWindow(this->h_internal_window[l]->WindowID);
        glutPostRedisplay();
    }
}


//calbacks
void plotterGL::keyboard_internal(int state) {
    for (int l = 0; l < SNN_aux->cnt_layers; l++) {
        glutSetWindow(this->h_internal_window[l]->WindowID);

        // delete window
        if (state == 27) {
            glutHideWindow();
            glutDestroyWindow(glutGetWindow());

        // store window
        } else if (state == 32) {
            std::string filename;
            filename += this->snapshots_dir + "/internal" + std::to_string(l) + "_" +
                    std::to_string(this->cnt_snapshots) + ".ppm";
            save_image(filename);
        }
    }
}


//calbacks
void plotterGL::timer_posttrace() {
    for (int l = 0; l < SNN_aux->cnt_layers; l++) {
        glutSetWindow(this->h_posttrace_window[l]->WindowID);
        glutPostRedisplay();
    }
}


//calbacks
void plotterGL::keyboard_posttrace(int state) {
    for (int l = 0; l < SNN_aux->cnt_layers; l++) {
        glutSetWindow(this->h_posttrace_window[l]->WindowID);

        // delete window
        if (state == 27) {
            glutHideWindow();
            glutDestroyWindow(glutGetWindow());

        // store window
        } else if (state == 32) {
            std::string filename;
            filename += this->snapshots_dir + "/posttrace" + std::to_string(l) + "_" +
                    std::to_string(this->cnt_snapshots) + ".ppm";
            save_image(filename);
        }
    }
}


// save current window
void save_image(std::string filename) {
    unsigned char* image = (unsigned char*) malloc(sizeof(unsigned char) * 3 * window_width * window_height);
    glReadPixels(0, 0, window_width, window_height, GL_RGB, GL_UNSIGNED_BYTE, image);
    PPM_writer(image, filename.c_str(), window_width, window_height);
}


// save current window
void PPM_writer(unsigned char *in, const char *name, int dimx, int dimy) {
    FILE *fp = fopen(name, "wb");
    (void) fprintf(fp, "P6\n%d %d\n255\n", dimx, dimy);
    for (int j = dimy-1; j >= 0; j--) {
        for (int i = 0; i < dimx; i++) {
            static unsigned char color[3];
            color[0] = in[3*i+3*j*dimx];
            color[1] = in[3*i+3*j*dimx+1];
            color[2] = in[3*i+3*j*dimx+2];
            (void) fwrite(color, 1, 3, fp);
        }
    }
    (void) fclose(fp);
}
