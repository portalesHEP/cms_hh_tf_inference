#ifndef PREPROC_HH_
#define PREPROC_HH_

// C++
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <assert.h>
#include "boost/filesystem.hpp"

// TensorFlow
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

class Preproc {
	/* Class to handle preprocessing of inpute features*/

private:
    std::vector<float> _means, _stdevs;
    bool _verbose;

public:
    // Methods
	Preproc(std::string, bool);
	~Preproc();
	tensorflow::Tensor process(std::vector<float>);
    bool load_preproc(std::string);
};

#endif /* PREPROC_HH_ */