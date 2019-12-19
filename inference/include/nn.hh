#ifndef NN_HH_
#define NN_HH_

// C++
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <assert.h>
#include <experimental/filesystem>

// TensorFlow
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

class NN {
	/* Class for loading and running a trained neural network */

private:
	unsigned int _n_threads;
	std::string _input_name, _output_name;
    bool _verbose;
    tensorflow::MetaGraphDef* _model;


public:
    // Methods
	NN(std::string, unsigned int, bool);
	~NN();
	float predict(tensorflow::Tensor);
    bool load_model(std::string);
};

#endif /* NN_HH_ */