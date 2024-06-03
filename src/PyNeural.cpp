#include "NoisyNeuralSimulator.hpp"
#include "LatticeNeuralSimulator.hpp"
#include "NonlocalNeuralSimulator.hpp"

#include <PyDataFrame.hpp>

NB_MODULE(neural_network_bindings, m) {
  EXPORT_SIMULATOR_DRIVER(NoisyNeuralSimulator);
  EXPORT_SIMULATOR_DRIVER(LatticeNeuralSimulator);
  EXPORT_SIMULATOR_DRIVER(NonlocalNeuralSimulator);
}

