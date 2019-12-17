 
#ifndef NN_HH_
#define NN_HH_

// C++
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <assert.h>

// TensorFlow
#include "DNN/TensorFlow/interface/TensorFlow.h"

class NN {
	/* Class for loading and running a trained neural network */

private:
	unsigned int _input_sz, _n_threads;
	std::string _input_name, _output_name;
    bool _verbose;

public:
    // Methods
	NN(std::string, unsigned int, bool);
	~NN();
	float predict(std::vector<float>);
    bool load_model(std::string);

    // Properties
    tensorflow::MetaGraphDef* model;
};

#endif /* NN_HH_ */