#include "OutputFormatter.h"

class PythonFormatter : public OutputFormatter
{
private:
    void encode_variables(std::ostream& os);
    void encode_inputs(std::ostream& os);
    void encode_update(std::ostream& os);
    void encode_outputs(std::ostream& os);
public:
    ~PythonFormatter();
    std::string assign(std::string left, std::string right);
    std::string multiplexer(std::string cond, std::string then_value, std::string else_value);
    std::string negate(std::string val);
    std::string constant(bool c);
    void write_to_stream(std::ostream& os);
    std::string get_extension();
};
