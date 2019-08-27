#include <iomanip>
#include <sstream>

#include "data.h"


// counters
int cnt_data = 0;
int cnt_runs = 0;

// file selection: generator for uniform random numbers
unsigned int seed = (unsigned int) std::chrono::system_clock::now().time_since_epoch().count();
std::default_random_engine random_folder(seed);
std::default_random_engine random_event(seed);
auto random_bool = std::bind(std::uniform_int_distribution<>(0,1),std::default_random_engine());

// handle simulation interruption
bool break_sim_data = false;
void interrupt_data(int sig){
    break_sim_data = true;
}


void indices2buffer(const std::string& dataset_dir, std::vector<std::string>& data_indices, bool& break_sim) {

    if (break_sim) return;

    std::string filename, type, folder;
    filename += dataset_dir + std::string("/data_file.csv");
    std::ifstream indices(filename);
    if (!indices.good()) {
        printf("\nError: data_file.csv does not exist.\n");
        break_sim = true;
        return;
    }

    while(indices.good()) {
        getline(indices, folder, '\n');
        if (folder.length() != 0) {
            data_indices.push_back(folder);
        }
    }
    indices.close();
}


void feed_network(const std::string& dataset_dir, std::vector<std::string>& dataset, const float sim_int,
                  const float sim_step, const int sim_step_range[2], int sim_num_steps, const float scale_ets,
                  Network *&SNN, plotterGL *plotter, const bool openGL, const bool data_augmentation,
                  const bool record_activity, std::string snapshots_dir, std::string weights_out, bool random, bool& break_sim) {

    if (break_sim) return;

    int idx_folder, ex, ey, ep;
    int folder_num_steps = 0, idx_step = 0, cnt_step = 0, ref_time = 0;
    float ets;
    bool break_feed = false;
    bool invert_polarity = false, horizontal_flip = false, vertical_flip = false, rotate = false;
    std::string folder, filename, metadata;
    std::vector<float> inputs_ets((unsigned long) SNN->h_inp_size[0] * SNN->h_inp_size[1] * SNN->h_inp_size[2]);

    // select event sequence
    if (random) {
        std::uniform_int_distribution<int> dist(0, (int) dataset.size() - 1);
        idx_folder = dist(random_folder);
    } else {
        idx_folder = cnt_data;
        cnt_data++;
        if (cnt_data >= dataset.size()) {
            printf("\nDataset limit reached.\n");
            break_sim_data = true;
        }
    }
    folder = dataset[idx_folder];
    if (!folder.empty() && folder[folder.length()-1] == '\r') folder.erase(folder.length()-1);
    filename = dataset_dir + '/' + folder + "/events.csv";

    // read metadata
    metadata = dataset_dir + '/' + folder + "/num_steps.csv";
    if (FILE *fp = fopen(metadata.c_str(), "r")) {
        while (fscanf(fp, "%d", &folder_num_steps) == 1) {}
        fclose(fp);
    }

    // fix sim_num_steps
    if (folder_num_steps - sim_step_range[1] <= sim_num_steps || sim_num_steps <= 0.f)
        sim_num_steps = folder_num_steps - sim_step_range[1] - 1;

    // select random starting point
    if (random) {
        std::uniform_int_distribution<int> dist(sim_step_range[0],
                                                folder_num_steps - sim_num_steps - sim_step_range[1] + sim_step_range[0]);
        idx_step = dist(random_event);
    } else {
        idx_step = sim_step_range[0];
    }

    // scale indices based on sim step
    idx_step /= (int) sim_step;
    sim_num_steps /= (int) sim_step;

    // data augmentation
    if (data_augmentation) {
        invert_polarity = (bool) random_bool();
        horizontal_flip = (bool) random_bool();
        vertical_flip = (bool) random_bool();
        if (SNN->h_inp_size[1] == SNN->h_inp_size[2])
            rotate = (bool) random_bool();
    }

    // feed the network
    SNN->update_input();
    signal(SIGINT, interrupt_data);
    if (FILE *fp = fopen(filename.c_str(), "r")) {
        while (fscanf(fp, "%f,%d,%d,%d", &ets, &ex, &ey, &ep) == 4 && !break_sim_data) {

            // check if polarity is considered
            if (SNN->h_inp_size[0] == 1) ep = 0;

            // assign events
            ets *= scale_ets;
            if (cnt_step >= idx_step) {
                ey = (int) ((float) ey / SNN->inp_scale[0]);
                ex = (int) ((float) ex / SNN->inp_scale[1]);

                if (horizontal_flip)
                    ex = ex + 2 * (SNN->h_inp_size[2] / 2 - ex) - 1;
                if (vertical_flip)
                    ey = ey + 2 * (SNN->h_inp_size[1] / 2 - ey) - 1;
                if (invert_polarity)
                    ep = ep ? 0 : 1;
                if (rotate) {
                    int ex_aux = ex;
                    ex = ey;
                    ey = ex_aux;
                }

                if (ex < 0) ex = 0;
                if (ey < 0) ey = 0;

                if (ep < 0 || ep >= SNN->h_inp_size[0] ||
                    ey < 0 || ey >= SNN->h_inp_size[1] ||
                    ex < 0 || ex >= SNN->h_inp_size[2]) std::cout << "Error: event location out of range.\n";
                else {
                    int idx_node = ex * SNN->h_inp_size[1] + ey;
                    int idx = ep * SNN->h_inp_size[1] * SNN->h_inp_size[2] * SNN->h_length_delay_inp[0] +
                            idx_node * SNN->h_length_delay_inp[0];
                    SNN->h_inputs[idx]++;
                    SNN->h_inputs[idx] = 1;

                    int idx_ets = ep * SNN->h_inp_size[1] * SNN->h_inp_size[2] + idx_node;
                    inputs_ets[idx_ets] = ets;
                }
            }

            if (ets > ref_time) {
                if (!cnt_step) ref_time = (int) (ets + sim_step * 1000.f);
                else ref_time += (int) (sim_step * 1000.f);

                if (cnt_step >= idx_step) {

                    // feed the network
                    SNN->feed(break_feed);
                    if (break_feed) break;
                    #ifdef OPENGL
                    if (openGL && !break_sim_data) {
                        SNN->copy_to_host();
                        plotter->update(SNN, cnt_step-idx_step);
                    }
                    #endif
                    #ifdef NPY
                    if (record_activity)
                        SNN->copy_to_host();
                        activity_to_npy(SNN, snapshots_dir, weights_out, cnt_runs, cnt_step-idx_step, plotter, true);
                    #endif
                    SNN->update_input();
                }
                cnt_step++;
                if (cnt_step - idx_step > sim_num_steps) break;
            }
        }
        fclose(fp);
    }

    // re-init network's internal state
    SNN->init();
    break_sim = break_sim_data;
    cnt_runs++;
}


void weights_to_csv(std::string& foldername, Network *&SNN, std::string dir) {

    std::string folder;
    struct stat sb;
    folder += dir + std::string("weights/");
    if (stat(folder.c_str(), &sb) != 0) {
        const int dir_err = mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err) {
            printf("Error: weights_dir could not be created\n");
            return;
        }
    }
    folder += foldername;
    if (stat(folder.c_str(), &sb) != 0) {
        const int dir_err = mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err) {
            printf("Error: weights_dir could not be created\n");
            return;
        }
    }

    // file per layer
    std::string filename;
    std::ofstream file;
    for (int l = 0; l < SNN->h_cnt_layers[0]; l++) {

        // excitatory weights
        filename += folder + std::string("/") + std::string("layer_") + std::to_string(l) +
                std::string("_excweights.csv");
        file.open(filename);

        for (int k = 0; k < SNN->h_layers[l]->cnt_kernels; k++) {
            for (int ch = 0; ch < SNN->h_layers[l]->kernel_channels; ch++) {
                for (int syn = 0; syn < SNN->h_layers[l]->rf_side * SNN->h_layers[l]->rf_side; syn++) {
                    for (int d = 0; d < SNN->h_layers[l]->num_delays; d++) {
                        int syn_delay_index = ch * SNN->h_layers[l]->rf_side * SNN->h_layers[l]->rf_side *
                                SNN->h_layers[l]->num_delays + syn * SNN->h_layers[l]->num_delays + d;
                        file << SNN->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index] << "\n";
                    }
                }
            }
        }
        file.close();
        filename.clear();

        // inhibitory weights
        filename += folder + std::string("/") + std::string("layer_") + std::to_string(l) +
                std::string("_inhweights.csv");
        file.open(filename);

        for (int k = 0; k < SNN->h_layers[l]->cnt_kernels; k++) {
            for (int ch = 0; ch < SNN->h_layers[l]->kernel_channels; ch++) {
                for (int syn = 0; syn < SNN->h_layers[l]->rf_side * SNN->h_layers[l]->rf_side; syn++) {
                    for (int d = 0; d < SNN->h_layers[l]->num_delays; d++) {
                        int syn_delay_index = ch * SNN->h_layers[l]->rf_side * SNN->h_layers[l]->rf_side *
                                SNN->h_layers[l]->num_delays + syn * SNN->h_layers[l]->num_delays + d;
                        file << SNN->h_layers[l]->h_kernels[k]->h_weights_inh[syn_delay_index] << "\n";
                    }
                }
            }
        }

        file.close();
        filename.clear();

        // STDP convergence flag
        filename += folder + std::string("/") + std::string("layer_") + std::to_string(l) + std::string("_convg.csv");
        file.open(filename);

        for (int k = 0; k < SNN->h_layers[l]->cnt_kernels; k++)
            file << SNN->h_layers[l]->h_kernels_cnvg[k] << "\n";

        file.close();
        filename.clear();

        // delays
        filename += folder + std::string("/") + std::string("layer_") + std::to_string(l) + std::string("_delays.csv");
        file.open(filename);

        for (int k = 0; k < SNN->h_layers[l]->cnt_kernels; k++)
            file << SNN->h_layers[l]->h_kernels[k]->num_delays_active << "\n";

        file.close();
        filename.clear();

        // layer structure
        filename += folder + std::string("/") + std::string("layer_") + std::to_string(l) + std::string("_struct.csv");
        file.open(filename);
        file << SNN->h_layers[l]->layer_type << "\n";
        file << SNN->h_layers[l]->cnt_kernels << "\n";
        file << SNN->h_layers[l]->kernel_channels << "\n";
        file << SNN->h_layers[l]->rf_side << "\n";
        file << SNN->h_layers[l]->num_delays << "\n";
        file << SNN->h_layers[l]->length_delay_inp << "\n";
        file.close();
        filename.clear();
    }
}


void csv_to_weights(std::string& folder, Network *&SNN, plotterGL *&plotter) {

    std::ifstream file;
    std::string filename, w;
    int structure[7];
    bool error = false;
    for (int l = 0; l < SNN->h_cnt_layers[0]; l++) {

        if (!SNN->h_layers[l]->load_weights)
            continue;

        // check layer structure
        filename += folder + std::string("/") + std::string("layer_") + std::to_string(l) + std::string("_struct.csv");
        file.open(filename);
        if (!file.good()) {
            error = true;
            break;
        }

        int cnt = 0;
        while (file.good()) {
            getline(file, w, '\n');
            structure[cnt] = atoi(w.c_str());
            cnt++;
        }

        file.close();
        file.clear();
        filename.clear();

        if (structure[0] != SNN->h_layers[l]->layer_type ||
            structure[1] != SNN->h_layers[l]->cnt_kernels ||
            structure[2] != SNN->h_layers[l]->kernel_channels ||
            structure[3] != SNN->h_layers[l]->rf_side ||
            structure[4] != SNN->h_layers[l]->num_delays ||
            structure[5] != SNN->h_layers[l]->length_delay_inp) {
            structure[6] = l;
            error = true;
            break;
        }
    }

    if (error) {
        std::string layer_data, layer_config;
        if (structure[0] == 0) layer_data += "Conv2d";
        else if (structure[0] == 1) layer_data += "Conv2dSep";
        else if (structure[0] == 2) layer_data += "Dense";
        else if (structure[0] == 3) layer_data += "Pooling";
        else if (structure[0] == 4) layer_data += "Merge";
        if (SNN->h_layers[structure[6]]->layer_type == 0) layer_config += "Conv2d";
        else if (SNN->h_layers[structure[6]]->layer_type == 1) layer_config += "Conv2dSep";
        else if (SNN->h_layers[structure[6]]->layer_type == 2) layer_config += "Dense";
        else if (SNN->h_layers[structure[6]]->layer_type == 3) layer_config += "Pooling";
        else if (SNN->h_layers[structure[6]]->layer_type == 4) layer_config += "Merge";

        printf("\nError: Weights could not be loaded.\nLayer %i expected: %s (%i, %i, %i, %i) - (%i, %i), "
               "current configuration: %s (%i, %i, %i, %i) - (%i, %i)\n",
               structure[6], layer_data.c_str(), structure[1], structure[2], structure[3], structure[3], structure[4],
               structure[5], layer_config.c_str(), SNN->h_layers[structure[6]]->cnt_kernels,
               SNN->h_layers[structure[6]]->kernel_channels, SNN->h_layers[structure[6]]->rf_side,
               SNN->h_layers[structure[6]]->rf_side, SNN->h_layers[structure[6]]->num_delays,
               SNN->h_layers[structure[6]]->length_delay_inp);
    } else {

        for (int l = 0; l < SNN->h_cnt_layers[0]; l++) {

            if (!SNN->h_layers[l]->load_weights)
                continue;

            // excitatory weights
            filename += folder + std::string("/") + std::string("layer_") + std::to_string(l) +
                    std::string("_excweights.csv");
            file.open(filename);

            for (int k = 0; k < SNN->h_layers[l]->cnt_kernels; k++) {
                for (int ch = 0; ch < SNN->h_layers[l]->kernel_channels; ch++) {
                    for (int syn = 0; syn < SNN->h_layers[l]->rf_side * SNN->h_layers[l]->rf_side; syn++) {
                        for (int d = 0; d < SNN->h_layers[l]->num_delays; d++) {
                            getline(file, w, '\n');
                            int syn_delay_index = ch * SNN->h_layers[l]->rf_side * SNN->h_layers[l]->rf_side *
                                    SNN->h_layers[l]->num_delays + syn * SNN->h_layers[l]->num_delays +
                                    d;
                            SNN->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index] = (float) atof(w.c_str());
                        }
                    }
                }
            }
            file.close();
            file.clear();
            filename.clear();

            // inhibitory weights
            filename += folder + std::string("/") + std::string("layer_") + std::to_string(l) +
                    std::string("_inhweights.csv");
            file.open(filename);

            for (int k = 0; k < SNN->h_layers[l]->cnt_kernels; k++) {
                for (int ch = 0; ch < SNN->h_layers[l]->kernel_channels; ch++) {
                    for (int syn = 0; syn < SNN->h_layers[l]->rf_side * SNN->h_layers[l]->rf_side; syn++) {
                        for (int d = 0; d < SNN->h_layers[l]->num_delays; d++) {
                            getline(file, w, '\n');
                            int syn_delay_index = ch * SNN->h_layers[l]->rf_side * SNN->h_layers[l]->rf_side *
                                    SNN->h_layers[l]->num_delays + syn * SNN->h_layers[l]->num_delays +
                                    d;
                            SNN->h_layers[l]->h_kernels[k]->h_weights_inh[syn_delay_index] = (float) atof(w.c_str());
                        }
                    }
                }
            }
            file.close();
            file.clear();
            filename.clear();

            // STDP convergence flags
            filename += folder + std::string("/") + std::string("layer_") + std::to_string(l) +
                    std::string("_convg.csv");
            file.open(filename);

            for (int k = 0; k < SNN->h_layers[l]->cnt_kernels; k++) {
                getline(file, w, '\n');
                SNN->h_layers[l]->h_kernels_cnvg[k] = (bool) atoi(w.c_str());
            }
            file.close();
            file.clear();
            filename.clear();

            // delays
            filename += folder + std::string("/") + std::string("layer_") + std::to_string(l) +
                        std::string("_delays.csv");
            file.open(filename);

            for (int k = 0; k < SNN->h_layers[l]->cnt_kernels; k++) {
                getline(file, w, '\n');
                for (int d = 0; d < SNN->h_layers[l]->num_delays; d++) {
                    if (d < atoi(w.c_str()))
                        SNN->h_layers[l]->h_kernels[k]->h_delay_active[d] = true;
                    else
                        SNN->h_layers[l]->h_kernels[k]->h_delay_active[d] = false;
                }
            }
            file.close();
            file.clear();
            filename.clear();

            // local motion color code
            #ifdef OPENGL
            filename += folder + std::string("/") + std::string("layer_") + std::to_string(l) +
                        std::string("_colors.csv");
            file.open(filename);

            if (file.good()) {
                for (int k = 0; k < SNN->h_layers[l]->cnt_kernels; k++) {
                    getline(file, w, '\n');
                    plotter->h_colors[l]->h_colors_r[k] = (float) atof(w.c_str());
                    getline(file, w, '\n');
                    plotter->h_colors[l]->h_colors_g[k] = (float) atof(w.c_str());
                    getline(file, w, '\n');
                    plotter->h_colors[l]->h_colors_b[k] = (float) atof(w.c_str());
                }
            }

            file.close();
            file.clear();
            filename.clear();
            #endif
        }
        std::cout << "Weights loaded.\n";
    }
}


// store network internal data to numpy files
void activity_to_npy(Network *&SNN, std::string snapshot_dir, std::string folder, int run, int step,
                     plotterGL *plotter, bool store_weights) {

    struct stat sb;
    if (stat(snapshot_dir.c_str(), &sb) != 0) {
        printf("\nWarning: snapshots_dir does not exist\n");
        return;
    }

    std::string foldername, foldername_sub, filename, run_string, step_string;
    foldername = snapshot_dir + std::string("/") + folder + std::string("/");
    if (stat(foldername.c_str(), &sb) != 0) {
        const int dir_err = mkdir(foldername.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err) {
            std::cout << foldername;
            printf("Error: snapshots_dir could not be created\n");
            return;
        }
    }

    std::stringstream ss1, ss2;
    ss1 << std::setw(5) << std::setfill('0') << run;
    ss1 >> run_string;
    ss2 << std::setw(5) << std::setfill('0') << step;
    ss2 >> step_string;
    foldername_sub = foldername + run_string + std::string("_") + step_string + std::string("/");
    if (stat(foldername_sub.c_str(), &sb) != 0) {
        const int dir_err = mkdir(foldername_sub.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err) {
            std::cout << foldername_sub;
            printf("Error: snapshots_dir could not be created\n");
            return;
        }
    }

    // input data
    std::vector<float> data((unsigned long) SNN->h_inp_size[0] * SNN->h_inp_size[1] * SNN->h_inp_size[2]);
    for (int ch = 0; ch < SNN->h_inp_size[0]; ch++) {
        for (int rows = 0; rows < SNN->h_inp_size[1]; rows++) {
            for (int cols = 0; cols < SNN->h_inp_size[2]; cols++) {
                int idx_SNN = ch * SNN->h_inp_size[1] * SNN->h_inp_size[2] * SNN->h_length_delay_inp[0] +
                        (cols * SNN->h_inp_size[1] + rows) * SNN->h_length_delay_inp[0];
                int idx_npy = ch * SNN->h_inp_size[1] * SNN->h_inp_size[2] + rows * SNN->h_inp_size[2] + cols;
                data[idx_npy] = SNN->h_inputs[idx_SNN];
            }
        }
    }

    filename += foldername_sub + std::string("input.npy");
    #ifdef NPY
    cnpy::npy_save(filename, &data[0],
                   {(unsigned long) SNN->h_inp_size[0],
                    (unsigned long) SNN->h_inp_size[1],
                    (unsigned long) SNN->h_inp_size[2]}, "w");
    #endif
    filename.clear();

    // network data
    for (int l = 0; l < SNN->h_cnt_layers[0]; l++) {
        std::vector<float> data((unsigned long) SNN->h_layers[l]->cnt_kernels * SNN->h_layers[l]->out_node_kernel);

        for (int k = 0; k < SNN->h_layers[l]->cnt_kernels; k++) {
            for (int rows = 0; rows < SNN->h_layers[l]->out_size[1]; rows++) {
                for (int cols = 0; cols < SNN->h_layers[l]->out_size[2]; cols++) {
                    int idx_SNN = (cols * SNN->h_layers[l]->out_size[1] + rows) * SNN->h_layers[l]->length_delay_out;
                    int idx_npy = k * SNN->h_layers[l]->out_node_kernel + rows * SNN->h_layers[l]->out_size[2] + cols;
                    data[idx_npy] = SNN->h_layers[l]->h_kernels[k]->h_node_train[idx_SNN];
                }
            }
        }

        filename += foldername_sub + std::string("layer_") + std::to_string(l) + std::string(".npy");
        #ifdef NPY
        cnpy::npy_save(filename, &data[0],
               {(unsigned long) SNN->h_layers[l]->cnt_kernels,
                (unsigned long) SNN->h_layers[l]->out_size[1],
                (unsigned long) SNN->h_layers[l]->out_size[2]}, "w");
        #endif
        filename.clear();

        // color data
        #ifdef OPENGL
        if (!run && !step) {
            std::vector<float> data((unsigned long) SNN->h_layers[l]->cnt_kernels * 3);
            for (int k = 0; k < SNN->h_layers[l]->cnt_kernels; k++) {
                int idx_npy = k * 3;
                data[idx_npy + 0] = plotter->h_colors[l]->h_colors_r[k];
                data[idx_npy + 1] = plotter->h_colors[l]->h_colors_g[k];
                data[idx_npy + 2] = plotter->h_colors[l]->h_colors_b[k];
            }

            filename += foldername + std::string("layer_") + std::to_string(l) + std::string("_color.npy");
            #ifdef NPY
            cnpy::npy_save(filename, &data[0],
               {(unsigned long) SNN->h_layers[l]->cnt_kernels,
                (unsigned long) 3}, "w");
            #endif
            filename.clear();
        }
        #endif
    }

    // weights
    if (store_weights) {
        std::string weights_dir, weights_subdir;
        weights_dir += foldername_sub + std::string("weights/");
        if (stat(weights_dir.c_str(), &sb) != 0) {
            const int dir_err = mkdir(weights_dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
            if (-1 == dir_err) {
                std::cout << weights_dir;
                printf("Error: snapshots_dir could not be created\n");
                return;
            }
        }
        weights_to_csv(folder, SNN, foldername_sub);
    }
}