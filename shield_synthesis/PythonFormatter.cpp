#include "PythonFormatter.h"

std::string PythonFormatter::get_extension()
{
    return ".py";
}


std::string PythonFormatter::assign(std::string left, std::string right) 
{
    std::string res = left + " = " + right;
    assignments_.push_back(res);
    return res;
}

std::string PythonFormatter::multiplexer(std::string cond, std::string then_value, std::string else_value)
{
    return "(" + then_value + " if " + cond + " else " + else_value + ")";
}

std::string PythonFormatter::negate(std::string val)
{
    return "(1 - " + val + ")";
}

std::string PythonFormatter::constant(bool c)
{
    return c ? "1" : "0";
}

void PythonFormatter::write_to_stream(std::ostream& os)
{
    os << "class Shield:\n\n";
    os << "  def __init__(self):\n";
    encode_variables(os);
    os << "\n";
    
    os << "  def tick(self, inputs):\n";
    encode_inputs(os);
    os << "\n";
    
    for (auto assignment : assignments_) {
        os << "    " << assignment << "\n";
    }
    
    os << "\n";
    encode_update(os);
    os << "\n";
    os << "    return ";
    encode_outputs(os);
    os << "\n";
}

void PythonFormatter::encode_variables(std::ostream& os)
{
    for (auto var : variables_) {
        os << "    self." << var << " = 0\n";
    }
}

void PythonFormatter::encode_inputs(std::ostream& os)
{
    for (int i = 0; i < inputs_.size(); i++) {
        os << "    " << inputs_[i] << " = inputs[" << i << "]\n";
    }
    
    for (auto var : variables_) {
        os << "    " << var << " = self." << var << "\n";
    }
}

void PythonFormatter::encode_update(std::ostream& os)
{
    for (auto var : variables_) {
        os << "    self." << var << " = " << var << "n\n";
    }
}

void PythonFormatter::encode_outputs(std::ostream& os)
{
    os << "[";
    for (int i = 0; i < outputs_.size() - 1; i++) {
        os << " " << outputs_[i] << ","; 
    } 
    
    if (outputs_.size() > 0) {
        os << " " << outputs_.back() << "]";
    }
}
    
PythonFormatter::~PythonFormatter() {
    assignments_.clear();
}
