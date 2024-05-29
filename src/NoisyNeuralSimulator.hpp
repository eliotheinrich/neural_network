#pragma once

#include <Frame.h>

#include "BooleanNeuralNetwork.hpp"

template <class T>
static std::vector<T> n_choose_k(const std::vector<T>& vals, size_t k, std::minstd_rand& rng) {
  std::vector<T> copy = vals;
  std::shuffle(copy.begin(), copy.end(), rng);
  copy.resize(k);
  return copy;
}

#define NNS_UNIFORM 0

class NoisyNeuralSimulator: public dataframe::Simulator {
	private:
    BooleanNeuralNetwork state;
		uint32_t system_size;

    double eta;
    size_t k;

    int connection_distribution;

    double sample_connection_strength() {
      if (connection_distribution == NNS_UNIFORM) {
        return 1.0;
      } else {
        return 0.0;
      }
    }

	public:
    NoisyNeuralSimulator(dataframe::Params &params, uint32_t num_threads) : dataframe::Simulator(params) {
      system_size = dataframe::utils::get<int>(params, "system_size");
      k = dataframe::utils::get<int>(params, "k");
      eta = dataframe::utils::get<double>(params, "eta");
      connection_distribution = dataframe::utils::get<int>(params, "connection_distribution", NNS_UNIFORM);

      state = BooleanNeuralNetwork(system_size);
      state.randomize(rng);

      std::vector<size_t> sites(system_size);
      std::iota(sites.begin(), sites.end(), 0);
      for (size_t i = 0; i < system_size; i++) {
        auto connected_sites = n_choose_k(sites, k, rng);
        for (auto const j : connected_sites) {
          state.connections.add_directed_edge(j, i, sample_connection_strength());
        }
      }
    }

		virtual void equilibration_timesteps(uint32_t num_steps) override {
			timesteps(num_steps);
		}

    virtual void timesteps(uint32_t num_steps) override {
      for (uint32_t t = 0; t < num_steps; t++) {
        state.update(eta, rng);
      }
    }

    void add_order_sample(dataframe::data_t& samples) {
      double s = state.get_order();
      dataframe::utils::emplace(samples, "order", s);
      dataframe::utils::emplace(samples, "order_abs", std::abs(s));
    }

		virtual dataframe::data_t take_samples() override {
      dataframe::data_t samples;

      add_order_sample(samples);

      return samples;
    }
};
