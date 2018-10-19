#include "data.h"


// counters
int cnt_data = 0;
int cnt_snaps = 0;

// file selection: generator for uniform random numbers
std::default_random_engine random_folder(static_cast<unsigned int>(time(nullptr)));
std::default_random_engine random_event(static_cast<unsigned int>(time(nullptr)));


// handle simulation interruption
bool isInterrupted_data = false;
void handleInterrupt_data(int sig){
    isInterrupted_data = true;
}


void indices2buffer(const std::string& dataset_dir, std::vector<std::string>& data_indices, bool& isInterrupted) {
    if (isInterrupted) return;

    std::string filename, type, folder;
    filename += dataset_dir + std::string("/data_file.csv");
    std::ifstream indices(filename);
    if (!indices.good()) {
        printf("\nError: data_file.csv does not exist.\n");
        isInterrupted = true;
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


void data2buffer_DVSsim(const std::string& dataset_dir, std::vector<std::string>& dataset, std::vector<int>& ets,
                        std::vector<int>& ex, std::vector<int>& ey, std::vector<int>& ep, std::vector<int>& events_step,
                        int& ms_init, int& ms_end, std::string& file, const float sim_step, bool random,
                        bool& isInterrupted) {
    if (isInterrupted) return;

    std::string filename;
    bool init, start;
    int ets_aux, ex_aux, ey_aux, ep_aux, idx;
    int event_cnt, ref_time = 0;

    // select random data folder
    if (random) {
        std::uniform_int_distribution<int> dist(0, (int) dataset.size() - 1);
        idx = dist(random_folder);
    } else {
        idx = cnt_data;
        cnt_data++;
    }
    file = dataset[idx];
    if (!file.empty() && file[file.length()-1] == '\r') file.erase(file.length()-1);
    filename = dataset_dir + '/' + file + '/' + file + ".csv";

    // dataset to buffer (events)
    init = true;
    start = true;
    event_cnt = 0;
    if (FILE *fp = fopen(filename.c_str(), "r")) {
        while (fscanf(fp, "%d,%d,%d,%d", &ets_aux, &ex_aux, &ey_aux, &ep_aux) == 4) { // (t, x, y, p)

            if (ets_aux) {
                ets.push_back(ets_aux / 1000);
                ey.push_back(ey_aux);
                ex.push_back(ex_aux);
                ep.push_back(ep_aux);
            }

            if (ets_aux && ets_aux > ref_time)
                init = true;

            if (init) {
                init = false;
                ref_time += (int) (sim_step * 1000.0);
                events_step.push_back(event_cnt-1);
                if (start) {
                    ms_init = ets_aux / 1000;
                    ref_time = ms_init * 1000 + (int) (sim_step * 1000.0);
                    start = false;
                } else {
                    ms_end = ets_aux / 1000;
                }
            }
            event_cnt++;
        }
        fclose(fp);
    } else {
        std::cout << "\nError: " << filename << " does not exist.\n";
        isInterrupted = true;
        return;
    }
}


void gt2buffer_DVSsim(const std::string& dataset_dir, std::vector<float>& gt_wx, std::vector<float>& gt_wy,
                      int& ms_init, int& ms_end, std::string& file, const float sim_step, bool& isInterrupted) {
    if (isInterrupted) return;

    int ms;
    float wx, wy, d;
    std::vector<float> gt_wx_aux, gt_wy_aux;
    std::string filename;
    filename += dataset_dir + "/" + file + "/ventral_flow.csv";

    // dataset to buffer (ground truth)
    if (FILE *fp = fopen(filename.c_str(), "r")) {
        while (fscanf(fp, "%d,%f,%f,%f", &ms, &wx, &wy, &d) == 4) {
            if (ms > ms_init && ms <= ms_end + 1) {
                gt_wx_aux.push_back(wx);
                gt_wy_aux.push_back(wy);
            }
        }
        fclose(fp);
    } else {
        std::cout << "\nError: " << filename << " does not exist.\n";
        isInterrupted = true;
        return;
    }

    // linearly interpolate data
    for (int i = 0; i < (int) ((float) gt_wx_aux.size() / sim_step); i++) {
        float ms_idx = (float) i * sim_step;
        int ms_up = (int) ceilf(ms_idx);
        int ms_down = (int) floorf(ms_idx);
        if (ms_down != ms_up && ms_up <= ms_end-2) {
            gt_wx.push_back((gt_wx_aux[ms_down]*(ms_up-ms_idx)+gt_wx_aux[ms_up]*(ms_idx-ms_down))/(ms_up-ms_down));
            gt_wy.push_back((gt_wy_aux[ms_down]*(ms_up-ms_idx)+gt_wy_aux[ms_up]*(ms_idx-ms_down))/(ms_up-ms_down));
        } else {
            gt_wx.push_back(gt_wx_aux[ms_down]);
            gt_wy.push_back(gt_wy_aux[ms_down]);
        }
    }
}


void feed_network(std::vector<int>& ets, std::vector<int>& ex, std::vector<int>& ey, std::vector<int>& ep,
                  std::vector<int>& events_step, std::vector<float>& gt_wx, std::vector<float>& gt_wy, int sim_nsteps,
                  Network *SNN, plotterGL *plotter, const bool openGL, const bool snapshots, std::string snapshots_dir,
                  int snapshots_freq, int snapshots_layer, const bool use_gt, bool inference, bool& isInterrupted) {
    if (isInterrupted) return;

    int idx_step;
    if (events_step.size() <= sim_nsteps) sim_nsteps = (int) events_step.size() - 1;
    if (!inference) {
        std::uniform_int_distribution<int> dist(0, (int) events_step.size() - sim_nsteps);
        idx_step = dist(random_event);
    } else idx_step = 0;

    int event_cnt = 1;
    int warmup_cnt = 0;
    SNN->update_input();
    signal(SIGINT, handleInterrupt_data);
    for (int i = events_step[idx_step]; i <= events_step[idx_step + sim_nsteps] && !isInterrupted_data; i++) {

        ey[i] = (int) ((float) ey[i] / SNN->inp_scale[0]);
        ex[i] = (int) ((float) ex[i] / SNN->inp_scale[1]);

        // update input vector
        if (i < events_step[idx_step + event_cnt]) {
            if (ep[i] < 0 || ep[i] >= SNN->inp_size[0] ||
                ey[i] < 0 || ey[i] >= SNN->inp_size[1] ||
                ex[i] < 0 || ex[i] >= SNN->inp_size[2]) std::cout << "Error: event location out of range.\n";
            else {
                int idx_node = ex[i] * SNN->inp_size[1] + ey[i];
                int idx = ep[i] * SNN->inp_size[1] * SNN->inp_size[2] * SNN->h_layers[0]->length_delay_inp +
                        idx_node * SNN->h_layers[0]->length_delay_inp;
                SNN->h_inputs[idx] = 1.f;
            }
        } else {

            // compute one step
            SNN->feed();
            warmup_cnt++;

            // network visualization
            #ifdef OPENGL
            if (openGL && !isInterrupted_data) {
                SNN->copy_to_host();
                plotter->update(SNN);
            }
            #endif

            // take snapshot
            #ifdef NPY
            if (snapshots && !isInterrupted_data &&
                warmup_cnt % snapshots_freq == 0 &&
                SNN->h_layers[snapshots_layer]->stdp_warmup_time < warmup_cnt) {
                SNN->copy_to_host();
                snapshot_to_file(SNN, snapshots_dir, snapshots_layer, use_gt, gt_wx, gt_wy, idx_step, event_cnt);
            }
            #endif

            // reset
            event_cnt++;
            SNN->update_input();

            // update input vector
            if (ep[i] < 0 || ep[i] >= SNN->inp_size[0] ||
                ey[i] < 0 || ey[i] >= SNN->inp_size[1] ||
                ex[i] < 0 || ex[i] >= SNN->inp_size[2]) std::cout << "Error: event location out of range.\n";
            else {
                int idx_node = ex[i] * SNN->inp_size[1] + ey[i];
                int idx = ep[i] * SNN->inp_size[1] * SNN->inp_size[2] * SNN->h_layers[0]->length_delay_inp +
                        idx_node * SNN->h_layers[0]->length_delay_inp;
                SNN->h_inputs[idx] = 1.f;
            }
        }
    }

    // re-init network's internal state
    SNN->init();
    isInterrupted = isInterrupted_data;
}


void weights_to_csv(std::string& foldername, Network *SNN) {

    std::string folder;
    struct stat sb;
    folder += std::string("weights/");
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
    std::cout << "\n\n";
    for (int l = 0; l < SNN->cnt_layers; l++) {

        // excitatory weights
        filename += folder + std::string("/") + std::string("layer_") + std::to_string(l) +
                std::string("_excweights.csv");
        file.open(filename);

        for (int k = 0; k < SNN->h_layers[l]->cnt_kernels; k++) {
            for (int ch = 0; ch < SNN->h_layers[l]->kernel_channels; ch++) {
                for (int syn = 0; syn < SNN->h_layers[l]->rf_side * SNN->h_layers[l]->rf_side; syn++) {
                    for (int d = 0; d < SNN->h_layers[l]->num_delays_synapse; d++) {
                        int syn_delay_index = ch * SNN->h_layers[l]->rf_side * SNN->h_layers[l]->rf_side *
                                SNN->h_layers[l]->num_delays_synapse + syn * SNN->h_layers[l]->num_delays_synapse + d;
                        file << SNN->h_layers[l]->h_kernels[k]->h_weights_exc[syn_delay_index] << "\n";
                    }
                }
            }
        }

        if (!l) std::cout << "Weights stored at " << std::string("weights/") + filename.substr(8,23) << "\n";
        file.close();
        filename.clear();

        // inhibitory weights
        filename += folder + std::string("/") + std::string("layer_") + std::to_string(l) +
                std::string("_inhweights.csv");
        file.open(filename);

        for (int k = 0; k < SNN->h_layers[l]->cnt_kernels; k++) {
            for (int ch = 0; ch < SNN->h_layers[l]->kernel_channels; ch++) {
                for (int syn = 0; syn < SNN->h_layers[l]->rf_side * SNN->h_layers[l]->rf_side; syn++) {
                    for (int d = 0; d < SNN->h_layers[l]->num_delays_synapse; d++) {
                        int syn_delay_index = ch * SNN->h_layers[l]->rf_side * SNN->h_layers[l]->rf_side *
                                SNN->h_layers[l]->num_delays_synapse + syn * SNN->h_layers[l]->num_delays_synapse + d;
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
            file << SNN->h_layers[l]->h_stdp_converged[k] << "\n";

        file.close();
        filename.clear();

        // layer structure
        filename += folder + std::string("/") + std::string("layer_") + std::to_string(l) + std::string("_struct.csv");
        file.open(filename);
        file << SNN->h_layers[l]->layer_type << "\n";
        file << SNN->h_layers[l]->cnt_kernels << "\n";
        file << SNN->h_layers[l]->inp_size[0] << "\n";
        file << SNN->h_layers[l]->rf_side << "\n";
        file << SNN->h_layers[l]->num_delays_synapse << "\n";
        file << SNN->h_layers[l]->length_delay_inp << "\n";
        file.close();
        filename.clear();
    }
}


void csv_to_weights(std::string& folder, Network *SNN) {

    std::ifstream file;
    std::string filename, w;
    int structure[7];
    bool error = false;
    for (int l = 0; l < SNN->cnt_layers; l++) {

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
            structure[2] != SNN->h_layers[l]->inp_size[0] ||
            structure[3] != SNN->h_layers[l]->rf_side ||
            structure[4] != SNN->h_layers[l]->num_delays_synapse ||
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
        if (SNN->h_layers[structure[6]]->layer_type == 0) layer_config += "Conv2d";
        else if (SNN->h_layers[structure[6]]->layer_type == 1) layer_config += "Conv2dSep";
        else if (SNN->h_layers[structure[6]]->layer_type == 2) layer_config += "Dense";
        else if (SNN->h_layers[structure[6]]->layer_type == 3) layer_config += "Pooling";

        printf("\nError: Weights could not be loaded.\nLayer %i expected: %s (%i, %i, %i, %i) - (%i, %i), "
               "current configuration: %s (%i, %i, %i, %i) - (%i, %i)\n",
               structure[6], layer_data.c_str(), structure[1], structure[2], structure[3], structure[3], structure[4],
               structure[5], layer_config.c_str(), SNN->h_layers[structure[6]]->cnt_kernels,
               SNN->h_layers[structure[6]]->inp_size[0], SNN->h_layers[structure[6]]->rf_side,
               SNN->h_layers[structure[6]]->rf_side, SNN->h_layers[structure[6]]->num_delays_synapse,
               SNN->h_layers[structure[6]]->length_delay_inp);
    } else {

        for (int l = 0; l < SNN->cnt_layers; l++) {

            if (!SNN->h_layers[l]->load_weights)
                continue;

            // excitatory weights
            filename += folder + std::string("/") + std::string("layer_") + std::to_string(l) +
                    std::string("_excweights.csv");
            file.open(filename);

            for (int k = 0; k < SNN->h_layers[l]->cnt_kernels; k++) {
                for (int ch = 0; ch < SNN->h_layers[l]->kernel_channels; ch++) {
                    for (int syn = 0; syn < SNN->h_layers[l]->rf_side * SNN->h_layers[l]->rf_side; syn++) {
                        for (int d = 0; d < SNN->h_layers[l]->num_delays_synapse; d++) {
                            getline(file, w, '\n');
                            int syn_delay_index = ch * SNN->h_layers[l]->rf_side * SNN->h_layers[l]->rf_side *
                                    SNN->h_layers[l]->num_delays_synapse + syn * SNN->h_layers[l]->num_delays_synapse +
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
                        for (int d = 0; d < SNN->h_layers[l]->num_delays_synapse; d++) {
                            getline(file, w, '\n');
                            int syn_delay_index = ch * SNN->h_layers[l]->rf_side * SNN->h_layers[l]->rf_side *
                                    SNN->h_layers[l]->num_delays_synapse + syn * SNN->h_layers[l]->num_delays_synapse +
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
                SNN->h_layers[l]->h_stdp_converged[k] = (bool) atoi(w.c_str());
            }
            file.close();
            file.clear();
            filename.clear();
        }
        std::cout << "Weights loaded.\n";
    }
}


// store network internal data to numpy files
void snapshot_to_file(Network *SNN, std::string foldername, int layer, bool use_gt, std::vector<float>& gt_wx,
                      std::vector<float>& gt_wy, int idx_step, int event_cnt) {

    std::string folder, filename;
    struct stat sb;
    folder += foldername + std::string("/");
    if (stat(folder.c_str(), &sb) != 0) {
        const int dir_err = mkdir(folder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err) {
            std::cout << folder;
            printf("Error: snapshots_dir could not be created\n");
            return;
        }
    }

    // save postsynaptic trace
    std::vector<float> data((unsigned long) SNN->h_layers[layer]->out_nodesep_kernel);
    for (int k = 0; k < SNN->h_layers[layer]->cnt_kernels; k++) {
        for (int ch = 0; ch < SNN->h_layers[layer]->out_maps; ch++) {
            for (int rows = 0; rows < SNN->h_layers[layer]->out_size[1]; rows++) {
                for (int cols = 0; cols < SNN->h_layers[layer]->out_size[2]; cols++) {
                    int idx_SNN = ch * SNN->h_layers[layer]->out_node_kernel +
                            cols * SNN->h_layers[layer]->out_size[1] + rows;
                    int idx_npy = ch * SNN->h_layers[layer]->out_node_kernel +
                            rows * SNN->h_layers[layer]->out_size[2] + cols;
                    data[idx_npy] = SNN->h_layers[layer]->h_kernels[k]->h_node_posttrace[idx_SNN];
                }
            }
        }
    }

    filename += folder + std::to_string(cnt_snaps) + std::string(".npy");
    #ifdef NPY
    cnpy::npy_save(filename, &data[0],
                   {(unsigned long) SNN->h_layers[layer]->cnt_kernels,
                    (unsigned long) SNN->h_layers[layer]->out_maps,
                    (unsigned long) SNN->h_layers[layer]->out_size[1],
                    (unsigned long) SNN->h_layers[layer]->out_size[2]}, "w");
    #endif
    filename.clear();
    cnt_snaps++;

    // save ground truth data (DVS specific)
    if (use_gt) {
        filename += folder + std::to_string(cnt_snaps) + std::string("_gt.csv");
        std::ofstream file;
        file.open(filename);
        file << gt_wx[idx_step + event_cnt] << "," << gt_wy[idx_step + event_cnt] << "\n";
        file.close();
        filename.clear();
    }
}
