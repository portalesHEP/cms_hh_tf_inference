#ifndef INF_WRAPPER_HH_
#define INF_WRAPPER_HH_

// C++
#include <iostream>
#include <string>

// Local
#include "pipeline.hh"

class InfWrapper {
	/* Wrapper for instanciating prediction pipelines and assigning events to correct ensemble */

private:
    std::vector<Pipeline> _pipes;
    bool _verbose;

public:
    // Methods
	InfWrapper(std::string, unsigned int, bool);
	~InfWrapper();
	float predict(std::vector<float>, unsigned long int);
    void load_pipeline(std::string, unsigned int);
};

#endif /* INF_WRAPPER_HH_ */

