 
#ifndef PIPELINE_HH_
#define PIPELINE_HH_

// C++
#include <iostream>
#include <string>

// TensorFlow
#include "DNN/TensorFlow/interface/TensorFlow.h"

// Local
#include "preproc.hh"
#include "ensemble.hh"
#include "nn.hh"

class Pipeline {
	/* Class to handle all steps of processing and predictions */

private:
    Preproc _preproc;
    Ensemble _ensemble;
    bool _verbose;

public:
    // Methods
	Pipeline(Preproc, Ensemble, bool);
	~Pipeline();
	float predict(std::vector<float>);
};

#endif /* PIPELINE_HH_ */