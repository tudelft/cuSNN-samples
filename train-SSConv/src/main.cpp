#include <signal.h>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cuda_runtime.h>

#include "cusnn.cuh"
#include "plotter.h"
#include "data.h"


// handle simulation interruption
bool break_sim = false;
void interrupt(int sig){
    break_sim = true;
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
    const std::string dataset_dir = "../data/roadmap";
    const int inp_size[] = {2, 264, 320}; // (channels, height, width)
    const float inp_scale[] = {2.f, 2.f}; // (height, width)

    // simulation settings
    const int runs = 1000000;
    const float sim_time = 150.f; // ms
    const float sim_step = 1.f; // ms
    const float sim_int = 1.f; // sim_steps input integration
    const float scale_ets = 1.f;

    const bool openGL = true;
    const bool load_model = false;
    const bool store_model_it = true;
    const bool record_activity = false;
    const bool data_augmentation = true;
    const int store_model_it_gap = -1;
    std::vector<int> kernels_display_idx = {0, 128};
    std::string weights_dir = "../weights/roadmap";
    std::string snapshots_dir = "../cuSNN_snapshots";

    // neuron and synapse models
    float neuron_refrac = 1.f; // ms
    float synapse_trace_init = 0.15f;
    
    bool inhibition = true;
    bool drop_delays = false;
    float drop_delays_th = 0.5f;

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    // SIMULATION PREP
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    // create GPU network
    Network *SNN = nullptr;
    int cudaDeviceCount;
    cudaGetDeviceCount(&cudaDeviceCount);
    if (cudaDeviceCount) {
        cudaSetDevice(0);
        printf("\n");
        SNN = new Network(inp_size, inp_scale, sim_step, neuron_refrac, synapse_trace_init, inhibition,
                          drop_delays, drop_delays_th);
    } else {
        printf("Error: No CUDA devices found.\n");
        return 0;
    }

    /* NETWORK STRUCTURE */
    /* void add_layer(std::string layer_type, bool learning, bool load_weights, bool homeostasis, float Vth,
                       float decay, float alpha, float max_delay = 1.f, int num_delays = 1,
                       float synapse_inh_scaling = 0.f, int rf_side = 7, int out_channels = 8,
                       std::string padding = "none", float w_init = 0.5f); */
    SNN->h_layers = (Layer **) malloc(sizeof(Layer*) * 5);
    SNN->add_layer("Conv2d", true, true, true, 0.4f, 5.f, 0.25f, 1.f, 1, 0.f, 5, 16, "half");
    SNN->create_network(break_sim);

    /* LEARNING RULE */
    SNN->enable_stdp_paredes(0.001f, 0.f, 0.075f, 10, true, 250, break_sim);
//    SNN->enable_stdp_shrestha(0.0001f, 5.f, false, 1.f, 10, true, break_sim);
//    SNN->enable_stdp_kheradpisheh(0.0001f, 10, true, break_sim);

    // visualization
    plotterGL *plotter = nullptr;
    #ifdef OPENGL
    if (openGL && !break_sim) {
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

    float avg_time = 0.f;
    srand((unsigned int) time(nullptr));
    signal(SIGINT, interrupt);

    time_t t = time(nullptr);
    struct tm *now = localtime(&t);
    char buffer[80];
    std::stringstream ss;
    std::string weights_out;
    strftime(buffer, sizeof(buffer), "weights_%m%d%Y_%H%M%S", now);
    ss << buffer;
    ss >> weights_out;

    // load weights
    if (load_model && !break_sim) {
        csv_to_weights(weights_dir, SNN, plotter);
        SNN->weights_to_device();
    }

    // model summary
    if (!break_sim) {
        SNN->summary();
        if (SNN->learning)
            std::cout << "Weights to be stored at " << std::string("weights/weights_") +
            weights_out.substr(8,23) << "\n\n";
    }

    // dataset to buffer
    std::vector<std::string> data_indices;
    indices2buffer(dataset_dir, data_indices, break_sim);

    // training loop
    int sim_num_steps = (int) (sim_time / sim_step);
    for (int r = 0; r < runs && !break_sim; r++) {
        auto t0 = Time::now();

        // feed the network
        feed_network(dataset_dir, data_indices, sim_int, sim_step, sim_num_steps, scale_ets, SNN, plotter, openGL,
                     data_augmentation, record_activity, snapshots_dir, weights_out, true, break_sim);

        // store weights
        if (SNN->learning && store_model_it) {
            SNN->copy_to_host();
            weights_to_csv(weights_out, SNN);
        }
        if (SNN->learning && r % store_model_it_gap == 0 && store_model_it_gap > 0) {
            SNN->copy_to_host();
            std::string weights_out_aux = weights_out + std::string("_r") + std::to_string(r);
            weights_to_csv(weights_out_aux, SNN);
            std::cout << "\nRun " << r << ": Weights stored at " << std::string("weights/") + weights_out_aux << "\n";
        }

        // print progress
        auto t1 = Time::now();
        fsec fs = t1 - t0;
        if (!r) avg_time = fs.count();
        else avg_time = avg_time * ((float) runs - 1.f) / (float) runs + fs.count() / (float) runs;
        std::cout << "Runs " << r+1 << "/" << runs << " -> ";
        std::cout << "ETA [s]: " << avg_time * (runs - r) << "\r";
        if (r != runs - 1) std::cout.flush();
        else std::cout << "\n";
    }
    printf("\n\n");

    // store weights
    if (SNN->learning) {
        SNN->copy_to_host();
        weights_to_csv(weights_out, SNN);
    }

    if (SNN != nullptr) delete SNN;
    #ifdef OPENGL
    if (plotter != nullptr) delete plotter;
    #endif
    return 0;
}