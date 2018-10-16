#ifndef DATA_H
#define DATA_H


#include <signal.h>
#include <random>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#ifdef NPY
#include "cnpy.h"
#endif
#include "plotter.h"


void indices2buffer(const std::string& dataset_dir, std::vector<std::string>& data_indices, bool& isInterrupted);
void feed_network(std::vector<int>& ets, std::vector<int>& ex, std::vector<int>& ey, std::vector<int>& ep,
                  std::vector<int>& events_step, std::vector<float>& gt_wx, std::vector<float>& gt_wy, int sim_nsteps,
                  Network *SNN, plotterGL *plotter, bool openGL, bool snapshots, std::string snapshots_dir,
                  int snapshots_freq, int snapshots_layer, bool use_gt, bool inference, bool& isInterrupted);
void weights_to_csv(std::string& foldername, Network *SNN);
void csv_to_weights(std::string& folder, Network *SNN);
void snapshot_to_file(std::string foldername, Network *SNN, float gt_wx, float gt_wy, int layer, int warmup_cnt, bool use_gt);
void snapshot_to_file(Network *SNN, std::string foldername, int layer, bool use_gt, std::vector<float>& gt_wx,
                      std::vector<float>& gt_wy, int idx_step, int event_cnt);

// DVSsim-specific functions
void data2buffer_DVSsim(const std::string& dataset_dir, std::vector<std::string>& dataset, std::vector<int>& ets,
                        std::vector<int>& ex, std::vector<int>& ey, std::vector<int>& ep, std::vector<int>& events_step,
                        int& ms_init, int& ms_end, std::string& file, float sim_step, bool random,
                        bool& isInterrupted);
void gt2buffer_DVSsim(const std::string& dataset_dir, std::vector<float>& gt_wx, std::vector<float>& gt_wy,
                      int& ms_init, int& ms_end, std::string& file, float sim_step, bool& isInterrupted);


#endif