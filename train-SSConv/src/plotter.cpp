#include "plotter.h"


// static plotter pointer
static plotterGL* plotter_static;

// flag
bool plot = true;
bool plot_input = true;
bool plot_weights = true;
bool plot_spikes = true;

// input window
GLint input_window;
bool input_text = false;

// window size
unsigned int window_width = 500;
unsigned int window_height = 500;
int tostring_precision = 2;


/* Colors */
// constructor
Colors::Colors(int cnt_kernel, std::vector<int> kernels) {

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
Colors::~Colors(){
    free(this->h_colors_r);
    free(this->h_colors_g);
    free(this->h_colors_b);
    free(this->h_kernels);
    free(this->h_selected);
}


// add color data
void plotterGL::add_colors(int layer, int cnt_kernel, std::vector<int> kernels) {
    this->h_colors[layer] = new Colors(cnt_kernel, kernels);
}


/* Weights */
// constructor
Weights::Weights(Network *SNN, int idx, int l, int d) {

    int num_columns = 4;
    int row = idx / num_columns;
    int col = idx % num_columns;
    int loc_width = 1, loc_height = 0;

    glutInitWindowPosition(window_width * (col+loc_width), (window_height + 30) * (row+loc_height));
    glutInitWindowSize(window_width, window_height);
    std::string title = "Layer " + std::to_string(l) + ": " + SNN->h_layers[l]->layer_type_str +
            " -- Weights (" + std::to_string(SNN->h_layers[l]->h_delay_indices[d]) + "ms delay)";
    const char* title_char = title.c_str();
    this->WindowID = glutCreateWindow(title_char);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
}


// add window
void plotterGL::add_weights_window(Network *SNN, int l, int d) {
    this->h_weights[this->cnt_weights_windows] = new Weights(SNN, this->cnt_weights_windows, l, d);
    this->cnt_weights_windows++;
}


/* SPIKES */
// constructor
Spikes::Spikes(Network *SNN, int idx) {
    glutInitWindowPosition(0, (window_height + 30) * (idx + 1));
    glutInitWindowSize(window_width, window_height);
    std::string title = "Layer " + std::to_string(idx) + ": " + SNN->h_layers[idx]->layer_type_str +
            " -- Activity";
    const char* title_char = title.c_str();
    this->WindowID = glutCreateWindow(title_char);
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
}


// add window
void plotterGL::add_spikes_window(Network *SNN) {
    this->h_spikes[this->cnt_spikes_windows] = new Spikes(SNN, this->cnt_spikes_windows);
    this->cnt_spikes_windows++;
}


// initialize class
plotterGL *plotter_init(Network *SNN, std::vector<int> kernels, std::string snapshots_dir) {
    return new plotterGL(SNN, kernels, snapshots_dir);
}


// constructor
plotterGL::plotterGL(Network *SNN, std::vector<int> kernels, std::string snapshots_dir) {

    this->SNN = SNN;
    this->cnt_snapshots = 0;
    this->enable_snapshot = true;
    this->snapshots_dir = snapshots_dir;

    struct stat sb;
    if (stat(this->snapshots_dir.c_str(), &sb) != 0)
        this->enable_snapshot = false;

    // kernel colors
    this->h_colors = (Colors **) malloc(sizeof(Colors*) * SNN->h_cnt_layers[0]);
    for (int l = 0; l < SNN->h_cnt_layers[0]; l++)
        this->add_colors(l, SNN->h_layers[l]->cnt_kernels, kernels);

    // initialize GL
    int argc = 1;
    char *argv[1] = {(char*)""};
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA);

    // initialize input window
    if (plot_input) {
        glutInitWindowPosition(0, 0);
        glutInitWindowSize(window_width, window_height);
        input_window = glutCreateWindow("Input");
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT);
    }

    // initialize windows for weights
    if (plot_weights) {
        int num_windows = 0;
        for (int l = 0; l < SNN->h_cnt_layers[0]; l++){
            for (int d = 0; d < SNN->h_layers[l]->num_delays; d++)
                num_windows++;
        }
        this->cnt_weights_windows = 0;
        this->h_weights = (Weights **) malloc(sizeof(Weights*) * num_windows);
        for (int l = 0; l < SNN->h_cnt_layers[0]; l++){
            for (int d = 0; d < SNN->h_layers[l]->num_delays; d++) {
                this->add_weights_window(this->SNN, l, d);
            }
        }
    }

    // initialize windows for spikes
    if (plot_spikes) {
        this->cnt_spikes_windows = 0;
        this->h_spikes = (Spikes **) malloc(sizeof(Spikes*) * SNN->h_cnt_layers[0]);
        for (int l = 0; l < SNN->h_cnt_layers[0]; l++)
            this->add_spikes_window(this->SNN);
    }

    // activate alpha
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);

    // register callbacks
    plotter_static = this;
    glutDisplayFunc(plotterGL::display);
    glutKeyboardFunc(plotterGL::keyboard);
    glutTimerFunc(0, plotterGL::timerEvent, 0);
}


// destructor
plotterGL::~plotterGL(){
    free(this->h_spikes);
    free(this->h_weights);
    free(this->h_colors);
}


// update the plot
void plotterGL::update(Network *SNN, int sim_step) {
    if (plot) {
        this->SNN = SNN;
        this->sim_step = sim_step;
        glutMainLoopEvent();
    }
}


//calbacks
void plotterGL::display() {
    if (plot_input) plotter_static->display_input();
    if (plot_weights) plotter_static->display_weights();
    if (plot_spikes) plotter_static->display_spikes();
}


//calbacks
void plotterGL::timerEvent(int value) {

    if (plot_input) {
        glutSetWindow(input_window);
        glutPostRedisplay();
    }
    if (plot_weights) plotter_static->timer_weights();
    if (plot_spikes) plotter_static->timer_spikes();
    glutTimerFunc(0, timerEvent, 0);
}


//calbacks
void plotterGL::keyboard(unsigned char key, int /*x*/, int /*y*/) {
    switch (key) {

        // hide all the windows
        case (27):
            if (plot) {
                if (plot_input) plotter_static->keyboard_input(27);
                if (plot_weights) plotter_static->keyboard_weights(27);
                if (plot_spikes) plotter_static->keyboard_spikes(27);
                plot = false;
            }
            return;

        // store one image per window
        case(32):
            if (plot) {
                if (!plotter_static->enable_snapshot) printf("\nWarning: snapshots_dir does not exist\n");
                else {
                    if (plot_input) plotter_static->keyboard_input(32);
                    if (plot_weights) plotter_static->keyboard_weights(32);
                    if (plot_spikes) plotter_static->keyboard_spikes(32);
                    plotter_static->cnt_snapshots++;
                }
            }
            return;

        // no action if other key
        default: return;
    }
}


//calbacks
void plotterGL::display_input() {

    glutSetWindow(input_window);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    float alpha = 1.0;

    if (this->SNN->h_inp_size[0] == 2) {
        for (int row = 0; row < this->SNN->h_inp_size[1]; row++) {
            for (int col = 0; col < this->SNN->h_inp_size[2]; col++) {
                int node_idx = col * this->SNN->h_inp_size[1] + row;
                int ch0_idx = node_idx * this->SNN->h_length_delay_inp[0];
                int ch1_idx = this->SNN->h_inp_size[1] * this->SNN->h_inp_size[2] *
                              this->SNN->h_length_delay_inp[0] + node_idx * this->SNN->h_length_delay_inp[0];

                int col_idx = col - this->SNN->h_inp_size[2] / 2;
                int row_Idx = this->SNN->h_inp_size[1] / 2 - row - 1;
                glColor4f(0.f, 0.f, 0.f, alpha);
                if (this->SNN->h_inputs[ch0_idx] && !this->SNN->h_inputs[ch1_idx])
                    glColor4f(1.f, 0.f, 0.f, alpha); // OFF -> red
                else if (!this->SNN->h_inputs[ch0_idx] && this->SNN->h_inputs[ch1_idx])
                    glColor4f(0.f, 1.f, 0.f, alpha); // ON -> green
                else if (this->SNN->h_inputs[ch0_idx] && this->SNN->h_inputs[ch1_idx])
                    glColor4f(1.f, 1.f, 0.f, alpha); // OFF-ON -> yellow

                glRectf((GLfloat) (col_idx / (this->SNN->h_inp_size[2] / 2.0)),
                        (GLfloat) (row_Idx / (this->SNN->h_inp_size[1] / 2.0)),
                        (GLfloat) ((col_idx + 1) / (this->SNN->h_inp_size[2] / 2.0)),
                        (GLfloat) ((row_Idx + 1) / (this->SNN->h_inp_size[1] / 2.0)));
            }
        }
    } else if (this->SNN->h_inp_size[0] == 1) {
        for (int row = 0; row < this->SNN->h_inp_size[1]; row++) {
            for (int col = 0; col < this->SNN->h_inp_size[2]; col++) {
                int node_idx = col * this->SNN->h_inp_size[1] + row;
                int ch0_idx = node_idx * this->SNN->h_length_delay_inp[0];

                int col_idx = col - this->SNN->h_inp_size[2] / 2;
                int row_Idx = this->SNN->h_inp_size[1] / 2 - row - 1;
                glColor4f(0.0, 0.0, 0.0, alpha);
                if (this->SNN->h_inputs[ch0_idx]) glColor4f(1.0, 1.0, 1.0, alpha);

                glRectf((GLfloat) (col_idx / (this->SNN->h_inp_size[2] / 2.0)),
                        (GLfloat) (row_Idx / (this->SNN->h_inp_size[1] / 2.0)),
                        (GLfloat) ((col_idx + 1) / (this->SNN->h_inp_size[2] / 2.0)),
                        (GLfloat) ((row_Idx + 1) / (this->SNN->h_inp_size[1] / 2.0)));
            }
        }
    }

    // display simulation step
    if (input_text) {
        glColor3f(1.f, 1.f, 1.f);
        glRasterPos2f(0.55f, 0.85f);
        std::string sim_step;
        sim_step += std::to_string(this->sim_step) + std::string("ms");
        int len = (int) sim_step.length();
        for (int i = 0; i < len; i++)
            glutBitmapCharacter(GLUT_BITMAP_8_BY_13, sim_step[i]);
    }

    glFlush();
}


//calbacks
void plotterGL::display_weights() {

    int idx = 0;
    for (int l = 0; l < this->SNN->h_cnt_layers[0]; l++) {
        for (int d = 0; d < this->SNN->h_layers[l]->num_delays; d++) {

            glutSetWindow(this->h_weights[idx]->WindowID);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable(GL_BLEND);
            glClearColor(0.f, 0.f, 0.f, 1.f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            float rows_max = floorf(sqrtf((float) this->SNN->h_layers[l]->cnt_kernels));
            if ((int) rows_max % 2 == 1 && rows_max != 1.f) rows_max--;
            float cols_max = (float) this->SNN->h_layers[l]->cnt_kernels / rows_max;

            int x_offset = 1 + this->SNN->h_layers[l]->rf_side + 1 + 1 + 2;
            int y_offset = 1 + this->SNN->h_layers[l]->rf_side + 2;
            float x_scaler = (float) x_offset * cols_max / 2.f;
            float y_scaler = (float) y_offset * rows_max / 2.f;

            int col_kernel = 0;
            int row_kernel = 0;
            for (int k = 0; k < this->SNN->h_layers[l]->cnt_kernels; k++) {

                // black box for kernel
                int effective_rf_rows = this->SNN->h_layers[l]->rf_side - this->SNN->h_layers[l]->rf_side_limits[0];
                int effective_rf_cols = this->SNN->h_layers[l]->rf_side - this->SNN->h_layers[l]->rf_side_limits[1];
                for (int rows = 0; rows < effective_rf_rows; rows++) {
                    for (int cols = 0; cols < effective_rf_cols; cols++) {

                        int col_idx = 1 + cols - (int) x_scaler + col_kernel * x_offset;
                        int row_idx = (int) y_scaler - 1 - rows - 1 - row_kernel * y_offset;
                        int syn_index = cols * this->SNN->h_layers[l]->rf_side + rows;

                        float accum_weight = 0.f;
                        float red = 0.f, green = 0.f, blue = 0.f;
                        if (this->SNN->h_layers[l]->h_kernels[k]->h_delay_active[d]) {
                            for (int ch = 0; ch < this->SNN->h_layers[l]->kernel_channels; ch++) {
                                int syn_delay_index = ch * this->SNN->h_layers[l]->rf_side *
                                        this->SNN->h_layers[l]->rf_side * this->SNN->h_layers[l]->num_delays +
                                        syn_index * this->SNN->h_layers[l]->num_delays + d;
                                float weight = this->SNN->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index] +
                                        this->SNN->h_layers[l]->synapse_inh_scaling *
                                        this->SNN->h_layers[l]->h_kernels[k]->h_weights_inh[syn_delay_index];
                                if (weight > 1.f) weight = 1.f;
                                else if (weight < 0.f) weight = 0.f;

                                // Event-based-sensors specific
                                if (!l) {
                                    if (this->SNN->h_layers[l]->kernel_channels > 1) {
                                        if (!ch) red += weight;
                                        else if (ch == 1) green += weight;
                                    } else {
                                        red += weight;
                                        green += weight;
                                        blue += weight;
                                    }
                                } else {
                                    if (this->SNN->h_layers[l-1]->h_kernels_cnvg[ch])
                                        accum_weight += weight;
                                }
                            }

                            if (l != 0) {
                                for (int ch = 0; ch < this->SNN->h_layers[l]->kernel_channels; ch++) {
                                    int syn_delay_index = ch * this->SNN->h_layers[l]->rf_side *
                                            this->SNN->h_layers[l]->rf_side *
                                            this->SNN->h_layers[l]->num_delays +
                                            syn_index * this->SNN->h_layers[l]->num_delays + d;
                                    float weight = this->SNN->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index] +
                                            this->SNN->h_layers[l]->synapse_inh_scaling *
                                            this->SNN->h_layers[l]->h_kernels[k]->h_weights_inh[syn_delay_index];
                                    if (weight > 1.f) weight = 1.f;
                                    else if (weight < 0.f) weight = 0.f;

                                    if (this->SNN->h_layers[l-1]->h_kernels_cnvg[ch]) {
                                        if (this->SNN->h_layers[l]->out_maps == 1) {
                                            red += this->h_colors[l-1]->h_colors_r[ch] * weight * weight / accum_weight;
                                            green += this->h_colors[l-1]->h_colors_g[ch] * weight * weight / accum_weight;
                                            blue += this->h_colors[l-1]->h_colors_b[ch] * weight * weight / accum_weight;
                                        } else {
                                            red += weight;
                                            green += weight;
                                            blue += weight;
                                        }
                                    }
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
                int col_end = 1 - (int) x_scaler + col_kernel * x_offset + this->SNN->h_layers[l]->rf_side;
                int row_begin = (int) y_scaler - 1 - row_kernel * y_offset;
                int row_end = (int) y_scaler - 1 - row_kernel * y_offset - this->SNN->h_layers[l]->rf_side;

                if (this->SNN->h_layers[l]->h_kernels[k]->h_delay_active[d]) {
                    if (this->SNN->h_layers[l]->h_kernels_cnvg[k]) glColor4f(0.f, 1.f, 0.f, 1.f);
                    else glColor4f(1.f, 0.f, 0.f, 1.f);
                } else glColor4f(1.f, 1.f, 1.f, 1.f);

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
                if (this->h_colors[l]->h_selected[k] != -1 && d == 0) {
                    glColor4f(this->h_colors[l]->h_colors_r[this->h_colors[l]->h_selected[k]],
                              this->h_colors[l]->h_colors_g[this->h_colors[l]->h_selected[k]],
                              this->h_colors[l]->h_colors_b[this->h_colors[l]->h_selected[k]], 1.f);
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
                if (this->SNN->h_layers[l]->enable_learning && !d && this->SNN->h_layers[l]->learning_type == 1)
                    stdp_convg += std::string(" = ") +
                            std::to_string(this->SNN->h_layers[l]->h_kernels[k]->stdp_paredes_objective_avg);
                int len = (int) stdp_convg.length();
                if (this->SNN->h_layers[l]->enable_learning && !d && this->SNN->h_layers[l]->learning_type == 1)
                    len -= tostring_precision;
                for (int i = 0; i < len; i++)
                    glutBitmapCharacter(GLUT_BITMAP_8_BY_13, stdp_convg[i]);
            }
            idx++;
        }
    }
    glFlush();
}


//calbacks
void plotterGL::display_spikes() {

    for (int l = 0; l < this->SNN->h_cnt_layers[0]; l++) {

        glutSetWindow(this->h_spikes[l]->WindowID);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_BLEND);

        float alpha = 1.f;
        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int width = this->SNN->h_layers[l]->out_size[2];
        if (width % 2 != 0) width++;
        int height = this->SNN->h_layers[l]->out_size[1];
        if (height % 2 != 0) height++;

        if (this->SNN->h_layers[l]->strides != 0) {
            for (int rows = 0; rows < this->SNN->h_layers[l]->out_size[1]; rows++) {
                for (int cols = 0; cols < this->SNN->h_layers[l]->out_size[2]; cols++) {

                    int cnt = 0;
                    float red = 0.f, green = 0.f, blue = 0.f;
                    int node_index = cols * this->SNN->h_layers[l]->out_size[1] + rows;
                    int begin = node_index * this->SNN->h_layers[l]->length_delay_out;

                    for (int k = 0; k < this->h_colors[l]->cnt_kernels; k++) {
                        int kernel = this->h_colors[l]->h_kernels[k];
                        if (kernel >= this->SNN->h_layers[l]->cnt_kernels)
                            continue;

                        if (this->SNN->h_layers[l]->h_kernels[kernel]->h_node_train[begin]) {
                            red += this->h_colors[l]->h_colors_r[k];
                            green += this->h_colors[l]->h_colors_g[k];
                            blue += this->h_colors[l]->h_colors_b[k];
                            cnt++;
                        }
                    }
                    red /= (float) cnt;
                    green /= (float) cnt;
                    blue /= (float) cnt;

                    int cols_aux = cols - width / 2;
                    int rows_aux = height / 2 - rows - 1;
                    glColor4f(red, green, blue, alpha);
                    glRectf((GLfloat) (cols_aux / ((float) width / 2.f)),
                            (GLfloat) (rows_aux / (height / 2.f)),
                            (GLfloat) ((cols_aux+1) / ((float) width / 2.f)),
                            (GLfloat) ((rows_aux+1) / (height / 2.f)));
                }
            }
        } else {
            float rows_max, cols_max;
            rows_max = floorf(sqrtf(this->SNN->h_layers[l]->cnt_kernels));
            if ((int) rows_max % 2 == 1 && rows_max != 1.f) rows_max--;
            cols_max = (float) this->SNN->h_layers[l]->cnt_kernels / rows_max;

            for (int k = 0; k < this->h_colors[l]->cnt_kernels; k++) {
                if (this->h_colors[l]->h_kernels[k] >= this->SNN->h_layers[l]->cnt_kernels)
                    continue;

                if (this->SNN->h_layers[l]->h_kernels[this->h_colors[l]->h_kernels[k]]->h_node_train[0]) {
                    int row = k % (int) rows_max;
                    int col = k / (int) rows_max;
                    float cols_aux = col - cols_max / 2;
                    float rows_aux = rows_max / 2 - row - 1;

                    glColor4f(this->h_colors[l]->h_colors_r[k],
                              this->h_colors[l]->h_colors_g[k],
                              this->h_colors[l]->h_colors_b[k], alpha);
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
void plotterGL::keyboard_input(int state) {

    glutSetWindow(input_window);

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
void plotterGL::timer_weights() {
    int idx = 0;
    for (int l = 0; l < this->SNN->h_cnt_layers[0]; l++) {
        for (int d = 0; d < this->SNN->h_layers[l]->num_delays; d++) {
            glutSetWindow(this->h_weights[idx]->WindowID);
            glutPostRedisplay();
            idx++;
        }
    }
}


//calbacks
void plotterGL::keyboard_weights(int state) {
    int idx = 0;
    for (int l = 0; l < this->SNN->h_cnt_layers[0]; l++) {
        for (int d = 0; d < this->SNN->h_layers[l]->num_delays; d++) {
            glutSetWindow(this->h_weights[idx]->WindowID);

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
void plotterGL::timer_spikes() {
    for (int l = 0; l < this->SNN->h_cnt_layers[0]; l++) {
        glutSetWindow(this->h_spikes[l]->WindowID);
        glutPostRedisplay();
    }
}


//calbacks
void plotterGL::keyboard_spikes(int state) {
    for (int l = 0; l < this->SNN->h_cnt_layers[0]; l++) {
        glutSetWindow(this->h_spikes[l]->WindowID);

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


// save current window
void save_image(std::string filename) {
    auto* image = (unsigned char*) malloc(sizeof(unsigned char) * 3 * window_width * window_height);
    glReadPixels(0, 0, window_width, window_height, GL_RGB, GL_UNSIGNED_BYTE, image);
    ppm_writer(image, filename.c_str(), window_width, window_height);
}


// save current window
void ppm_writer(unsigned char *in, const char *name, int dimx, int dimy) {
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
