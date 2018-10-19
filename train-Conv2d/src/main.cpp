#include <signal.h>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

#include "cusnn.cuh"
#include "plotter.h"
#include "data.h"


// handle simulation interruption
bool isInterrupted = false;
void handleInterrupt(int sig){
    isInterrupted = true;
}


int main(int argc, char** argv){

    // timers
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<float> fsec;

    // printing options
    std::cout << std::fixed;
    std::cout << std::setprecision(4);

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // SIMULATION CONFIG
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    // dataset
    const std::string dataset_dir = "../data/DVSsim_checkerboard";
    const int inp_size[] = {2, 128, 128}; // (channels, height, width)
    const float inp_scale[] = {2.f, 2.f}; // (height, width)

    // simulation settings
    const int runs = 10000;
    const float sim_time = 150.f; // ms
    const float sim_step = 1.f; // ms

    const bool openGL = true;
    const bool load_model = false;
    const bool use_gt = false;
    const bool snapshots = false;
    int snapshots_freq = 5; // timesteps
    int snapshots_layer = 0;
    std::vector<int> kernels_display_idx = {0, 128};
    std::string weights_dir = "weights/weights_10122018_133510";
    std::string snapshots_dir = "../cuSNN_pics";

    // neuron and synapse models
    float neuron_refrac = 3.f; // ms
    float synapse_trace_init = 0.15f;

    // STDP settings
    int stdp_stats_window = 250;
    int stdp_limit_updates = -1;
    float stdp_rate = 0.001f;
    float stdp_convg_th = 0.075f;

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // SIMULATION PREP
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    // filename for weight file
    time_t t = time(nullptr);
    struct tm * now = localtime(&t);
    char buffer [80];
    std::stringstream ss;
    std::string weights_out;
    strftime (buffer, sizeof(buffer), "weights_%m%d%Y_%H%M%S", now);
    ss << buffer;
    ss >> weights_out;

    // create GPU network
    Network *SNN = nullptr;
    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);
    if (cudaDeviceCount) {
        cudaSetDevice(0);
        printf("\n");
        SNN = new Network(inp_size, inp_scale, sim_step, neuron_refrac, synapse_trace_init, stdp_rate,
                          stdp_stats_window, stdp_convg_th, stdp_limit_updates);
    } else {
        printf("Error: No CUDA devices found.\n");
        return 0;
    }

    /* NETWORK STRUCTURE */
    /* void add_layer(std::string layer_type, bool stdp, bool load_weights, bool homeostasis, float Vth, float decay,
                      float alpha, float max_delay = 1.f, int num_delays_synapse = 1, float synapse_inh_scaling = 0.f,
                      int rf_side = 7, int out_channels = 8, std::string padding = "none", float w_init = 0.5f,
                      float stdp_scale_a = 0.f); */
    SNN->h_layers = (Layer **) malloc(sizeof(Layer*) * 4);
    SNN->add_layer("Conv2d", true, true, true, 1.f, 5.f, 0.3f, 1.f, 1, 0.5, 7, 2, "half", 0.5f, 0.f);
    SNN->create_network(isInterrupted);

    // network visualization
    plotterGL *plotter = nullptr;
    #ifdef OPENGL
    if (openGL && !isInterrupted) {
        std::vector<int> kernels_display;
        if (kernels_display_idx.size() == 2) {
            for (int k = kernels_display_idx[0]; k < kernels_display_idx[1]; k++)
                kernels_display.push_back(k);
        } else {
            for (int k = 0; k < kernels_display_idx.size(); k++) {
                if (kernels_display_idx[k] != -1)
                    kernels_display.push_back(kernels_display_idx[k]);
            }
        }
        SNN->copy_to_host();
        plotter = plotter_init(SNN, kernels_display, snapshots_dir);
    }
    #endif

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // SIMULATION RUN
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    int ms_end = 0, ms_init = 0, num_convg = 0;
    float avgTime = 0.f;
    std::string file;
    std::vector<int> ets, ex, ey, ep, events_step;
    std::vector<float> gt_wx, gt_wy;
    srand((unsigned int) time(nullptr));
    signal(SIGINT, handleInterrupt);

    // load weights
    if (load_model && !isInterrupted) {
        csv_to_weights(weights_dir, SNN);
        SNN->weights_to_device();
    }

    // dataset to buffer
    std::vector<std::string> data_indices;
    indices2buffer(dataset_dir, data_indices, isInterrupted);

    // training loop
    printf("\n");
    for (int r = 0; r < runs && !isInterrupted; r++) {
        auto t0 = Time::now();
        int sim_nsteps = (int) (sim_time / sim_step);

        // load input
        data2buffer_DVSsim(dataset_dir, data_indices, ets, ex, ey, ep, events_step,
                           ms_init, ms_end, file, sim_step, true, isInterrupted);
        if (use_gt) gt2buffer_DVSsim(dataset_dir, gt_wx, gt_wy, ms_init, ms_end, file, sim_step, isInterrupted);

        // feed the network
        feed_network(ets, ex, ey, ep, events_step, gt_wx, gt_wy, sim_nsteps, SNN, plotter, openGL,
                     snapshots, snapshots_dir, snapshots_freq, snapshots_layer, use_gt, false, isInterrupted);

        // print progress
        auto t1 = Time::now();
        fsec fs = t1 - t0;
        if (!r) avgTime = fs.count();
        else avgTime = avgTime * ((float) runs - 1.f) / (float) runs + fs.count() / (float) runs;
        std::cout << "Runs " << r+1 << "/" << runs << " -> ";
        std::cout << "ETA [s]: " << avgTime * (runs - r);
        if (num_convg != -1) num_convg = SNN->check_kernel_convg();
        if (num_convg != -1) std::cout  << " -> kernels converged: " << num_convg <<  "\r";
        else std::cout << "\r";
        if (r != runs - 1) std::cout.flush();
        else std::cout << "\n";

        ets.clear();
        ex.clear();
        ey.clear();
        ep.clear();
        events_step.clear();
        gt_wx.clear();
        gt_wy.clear();
    }
    printf("\n");

    // store weights
    if (num_convg != -1) {
        SNN->copy_to_host();
        weights_to_csv(weights_out, SNN);
    }

    if (SNN != nullptr) delete SNN;
    #ifdef OPENGL
    if (plotter != nullptr) delete plotter;
    #endif
    return 0;
}
