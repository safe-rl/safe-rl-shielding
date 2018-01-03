#ifndef OUTPUT_FORMATTER_H
#define OUTPUT_FORMATTER_H

#include <string>
#include <vector>
#include <ostream>

class OutputFormatter 
{
protected:
    std::vector<std::string> assignments_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    std::vector<std::string> variables_;
    std::vector<std::string> temporaries_;
public:
    virtual ~OutputFormatter() {
        
    }
    virtual std::string assign(std::string left, std::string right) = 0;
    virtual std::string multiplexer(std::string cond, std::string then_value, std::string else_value) = 0;
    virtual std::string negate(std::string val) = 0;
    virtual std::string constant(bool c) = 0;
    virtual void write_to_stream(std::ostream& os) = 0;
    virtual std::string get_extension() = 0;
    
    void add_input(std::string input) {
        inputs_.push_back(input);
    }
    
    void add_output(std::string output) {
        outputs_.push_back(output);
    }
    
    void add_variable(std::string variable) {
        variables_.push_back(variable);
    }
    
    void add_temporary(std::string temporary) {
        for (auto existing : temporaries_) {
            if (existing == temporary) {
                return;
            }
        }

        temporaries_.push_back(temporary);
    }
};

#endif
